// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Executables/ExportEquationOfStateForRotNS/DumpRotNSEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace {

std::vector<std::array<double, 4>> read_eos_output(const std::string& path) {
  std::ifstream file(path);
  std::vector<std::array<double, 4>> rows;
  double n = 0.0;
  double e = 0.0;
  double ye = 0.0;
  double p = 0.0;
  while (file >> n >> e >> ye >> p) {
    rows.push_back({n, e, ye, p});
  }
  return rows;
}

void test_interpolate_profile_polynomial_branch() {
  // log-linear T(rho): log10(T) = 0.5*log10(rho) + 1.0
  // Cubic polynomial interpolation recovers a linear function exactly.
  const size_t n_points = 20;
  DataVector log_rho(n_points);
  DataVector log_T(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    log_rho[i] = -6.0 + static_cast<double>(i) * 0.3;
    log_T[i] = 0.5 * log_rho[i] + 1.0;
  }

  for (const double target : {-4.65, -3.0, -1.5}) {
    double result = std::numeric_limits<double>::signaling_NaN();
    ExportEosForRotNS::interpolate_profile_at_log_density(
        make_not_null(&result), target, log_rho, log_T);
    CHECK(result == approx(0.5 * target + 1.0).epsilon(1.0e-12));
  }
}

void test_interpolate_profile_linear_fallback_branch() {
  // Steep T(rho): each step is 1.5 decades in T, so a 4-point stencil spans
  // 4.5 decades > 2, triggering the linear fallback. For a linear function in
  // log space, linear interpolation is exact.
  const size_t n_points = 20;
  DataVector log_rho(n_points);
  DataVector log_T(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    log_rho[i] = static_cast<double>(i);
    log_T[i] = 1.5 * static_cast<double>(i);
  }

  for (const double target : {5.5, 8.5, 11.5}) {
    double result = std::numeric_limits<double>::signaling_NaN();
    ExportEosForRotNS::interpolate_profile_at_log_density(
        make_not_null(&result), target, log_rho, log_T, 2.0);
    CHECK(result == approx(1.5 * target).epsilon(1.0e-12));
  }
}

void test_interpolate_profile_boundary_stencil() {
  // Queries near the table edges exercise the stencil clamping logic.
  const size_t n_points = 10;
  DataVector log_rho(n_points);
  DataVector log_T(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    log_rho[i] = static_cast<double>(i);
    log_T[i] = 0.5 * static_cast<double>(i) + 1.0;
  }

  for (const double target : {0.1, 8.9}) {
    double result = std::numeric_limits<double>::signaling_NaN();
    ExportEosForRotNS::interpolate_profile_at_log_density(
        make_not_null(&result), target, log_rho, log_T);
    CHECK(result == approx(0.5 * target + 1.0).epsilon(1.0e-10));
  }
}

void test_barotropic_eos() {
  // PolytropicFluid(K=100, Gamma=2): p = 100*rho^2, epsilon = 100*rho
  // Exact identity for Gamma=2: e_geometric - p_geometric = rho_geometric
  const double K = 100.0;
  const double Gamma = 2.0;
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      eos{EquationsOfState::PolytropicFluid<true>{K, Gamma}};

  const std::string output_file = "test_barotropic_eos_output.dat";
  if (file_system::check_if_file_exists(output_file)) {
    file_system::rm(output_file, false);
  }
  const size_t num_points = 10;
  const double lower_rho_cgs = 1.0e13;
  const double upper_rho_cgs = 1.0e15;

  ExportEosForRotNS::dump_eos(eos, num_points, output_file, lower_rho_cgs,
                              upper_rho_cgs, std::nullopt);
  const auto rows = read_eos_output(output_file);
  file_system::rm(output_file, false);

  CHECK(rows.size() == num_points);

  // Check density bounds
  const double baryon_mass_rotns_cgs =
      hydro::units::geometric::default_baryon_mass *
      hydro::units::cgs::mass_unit;
  CHECK(std::pow(10.0, rows.front()[0]) ==
        approx(lower_rho_cgs / baryon_mass_rotns_cgs).epsilon(1.0e-13));
  CHECK(std::pow(10.0, rows.back()[0]) ==
        approx(upper_rho_cgs / baryon_mass_rotns_cgs).epsilon(1.0e-13));

  for (const auto& row : rows) {
    const double n_cgs = std::pow(10.0, row[0]);
    const double e_cgs = std::pow(10.0, row[1]);
    const double p_cgs = std::pow(10.0, row[3]);

    const double rho = n_cgs * std::pow(hydro::units::cgs::length_unit, 3.0) *
                       eos.baryon_mass();
    const double e_geometric =
        e_cgs / hydro::units::cgs::rest_mass_density_unit;
    const double p_geometric = p_cgs / hydro::units::cgs::pressure_unit;

    // Direct pressure check: p = K * rho^Gamma
    CHECK(p_geometric == approx(K * rho * rho).epsilon(1.0e-13));
    // Identity: e - p = rho (for Gamma=2, holds regardless of K)
    CHECK(e_geometric - p_geometric == approx(rho).epsilon(1.0e-13));
    // PolytropicFluid returns the default Y_e = 0.1
    CHECK(row[2] == approx(0.1).epsilon(1.0e-13));
  }
}

void test_eos_with_thermal_profile() {
  // IdealFluid(Gamma=2): epsilon = T/(Gamma-1) = T, p = rho*epsilon = rho*T,
  // e = rho*(1+T). Same Gamma=2 identity: e_geometric - p_geometric = rho.
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>> eos{
      EquationsOfState::IdealFluid<true>{2.0}};

  // Generate T(rho) in geometric units: T = T0 * sqrt(rho / rho_ref)
  const std::string thermal_profile_file = "test_eos_thermal_profile.dat";
  if (file_system::check_if_file_exists(thermal_profile_file)) {
    file_system::rm(thermal_profile_file, false);
  }
  const std::string output_file = "test_equilibrium_eos_output.dat";
  if (file_system::check_if_file_exists(output_file)) {
    file_system::rm(output_file, false);
  }
  const size_t n_t_points = 30;
  const double rho_ref = 1.0e-8;
  const double T0 = 5.0e-4;
  {
    std::ofstream file(thermal_profile_file);
    file << n_t_points << " 0\n";
    for (size_t i = 0; i < n_t_points; ++i) {
      const double rho = rho_ref * std::pow(10.0, static_cast<double>(i) * 0.2);
      const double T = T0 * std::sqrt(rho / rho_ref);
      file << std::scientific << std::setprecision(14) << rho << " " << T
           << "\n";
    }
  }

  // Density range interior to the thermal profile table
  const double lower_rho_cgs =
      rho_ref * std::pow(10.0, 1.0) * hydro::units::cgs::rest_mass_density_unit;
  const double upper_rho_cgs =
      rho_ref * std::pow(10.0, 4.5) * hydro::units::cgs::rest_mass_density_unit;

  const size_t num_points = 10;

  ExportEosForRotNS::dump_eos(eos, num_points, output_file, lower_rho_cgs,
                              upper_rho_cgs, thermal_profile_file);
  const auto rows = read_eos_output(output_file);
  file_system::rm(output_file, false);
  file_system::rm(thermal_profile_file, false);

  CHECK(rows.size() == num_points);

  // Check density bounds
  const double baryon_mass_rotns_cgs =
      hydro::units::geometric::default_baryon_mass *
      hydro::units::cgs::mass_unit;
  CHECK(std::pow(10.0, rows.front()[0]) ==
        approx(lower_rho_cgs / baryon_mass_rotns_cgs).epsilon(1.0e-12));
  CHECK(std::pow(10.0, rows.back()[0]) ==
        approx(upper_rho_cgs / baryon_mass_rotns_cgs).epsilon(1.0e-12));

  for (const auto& row : rows) {
    const double n_cgs = std::pow(10.0, row[0]);
    const double e_cgs = std::pow(10.0, row[1]);
    const double p_cgs = std::pow(10.0, row[3]);

    const double rho = n_cgs * std::pow(hydro::units::cgs::length_unit, 3.0) *
                       eos.baryon_mass();
    const double e_geometric =
        e_cgs / hydro::units::cgs::rest_mass_density_unit;
    const double p_geometric = p_cgs / hydro::units::cgs::pressure_unit;

    // Identity: e - p = rho (for IdealFluid Gamma=2, independent of T)
    CHECK(e_geometric - p_geometric == approx(rho).epsilon(1.0e-13));
    // For IdealFluid Gamma=2, p = rho*T, so p/rho recovers the interpolated
    // temperature. log10(T) is linear in log10(rho), so cubic interpolation
    // in log space is exact.
    const double t_expected = T0 * std::sqrt(rho / rho_ref);
    CHECK(p_geometric / rho == approx(t_expected).epsilon(1.0e-10));
    // IdealFluid returns the default Y_e = 0.1
    CHECK(row[2] == approx(0.1).epsilon(1.0e-13));
  }
}

void test_ye_profile() {
  using TEoS = EquationsOfState::Tabulated3D<true>;
  constexpr size_t n_T = 3;
  constexpr size_t n_rho = 3;
  constexpr size_t n_ye = 3;
  constexpr size_t num_vars = TEoS::NumberOfVars;

  // Natural-log grids: T in [1e-3, 1e1], rho in [1e-11, 1e-7], Ye in [0.1, 0.5]
  std::vector<double> log_T_vals(n_T);
  std::vector<double> log_rho_vals(n_rho);
  std::vector<double> ye_vals(n_ye);
  for (size_t i = 0; i < n_T; ++i) {
    log_T_vals[i] = std::log(1.0e-3) + static_cast<double>(i) * std::log(100.0);
  }
  for (size_t i = 0; i < n_rho; ++i) {
    log_rho_vals[i] =
        std::log(1.0e-11) + static_cast<double>(i) * std::log(100.0);
  }
  for (size_t i = 0; i < n_ye; ++i) {
    ye_vals[i] = 0.1 + static_cast<double>(i) * 0.2;
  }

  // epsilon = T (no Ye dependence), pressure = rho*T*exp(Ye).
  // In table space: ln(epsilon) = ln(T), ln(pressure) = ln(rho)+ln(T)+Ye.
  // Both are linear in the table coordinates (ln_T, ln_rho, Ye), so
  // trilinear interpolation recovers them exactly.
  std::vector<double> table_data(n_T * n_rho * n_ye * num_vars, 0.0);
  for (size_t i_ye = 0; i_ye < n_ye; ++i_ye) {
    for (size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
      for (size_t i_T = 0; i_T < n_T; ++i_T) {
        const size_t ijk = i_T + n_T * (i_rho + n_rho * i_ye);
        table_data[TEoS::Epsilon + num_vars * ijk] = log_T_vals[i_T];
        table_data[TEoS::Pressure + num_vars * ijk] =
            log_rho_vals[i_rho] + log_T_vals[i_T] + ye_vals[i_ye];
        table_data[TEoS::CsSquared + num_vars * ijk] = 0.1;
      }
    }
  }
  const TEoS eos(ye_vals, log_rho_vals, log_T_vals, table_data, 0.0, 1.0);

  const std::string profile_file = "test_ye_profile.dat";
  const std::string output_file = "test_ye_profile_output.dat";
  if (file_system::check_if_file_exists(profile_file)) {
    file_system::rm(profile_file, false);
  }
  if (file_system::check_if_file_exists(output_file)) {
    file_system::rm(output_file, false);
  }

  // Profile: 5 log-spaced rho points, linear Y_e and log-linear T in
  // log10(rho). Profile rho range [5e-11, 5e-10] wraps the dump range [1e-10,
  // 2e-10].
  const size_t n_profile = 5;
  const double rho_profile_min = 5.0e-11;
  const double rho_profile_max = 5.0e-10;
  const double ye_profile_min = 0.15;
  const double ye_profile_max = 0.35;
  const double T_profile_min = 1.0e-2;
  const double T_profile_max = 1.0e-1;
  {
    std::ofstream file(profile_file);
    file << n_profile << " 1\n";
    for (size_t i = 0; i < n_profile; ++i) {
      const double frac =
          static_cast<double>(i) / static_cast<double>(n_profile - 1);
      const double rho =
          rho_profile_min * std::pow(rho_profile_max / rho_profile_min, frac);
      const double T =
          T_profile_min * std::pow(T_profile_max / T_profile_min, frac);
      const double ye =
          ye_profile_min + frac * (ye_profile_max - ye_profile_min);
      file << std::scientific << std::setprecision(14) << rho << " " << T << " "
           << ye << "\n";
    }
  }

  // CGS bounds chosen so geometric rho dumps in [1e-10, 2e-10]
  const double baryon_mass_rotns_cgs =
      hydro::units::geometric::default_baryon_mass *
      hydro::units::cgs::mass_unit;
  const double cu_length = std::pow(hydro::units::cgs::length_unit, 3.0);
  const double rho_geo_lower = 1.0e-10;
  const double rho_geo_upper = 2.0e-10;
  const double lower_rho_cgs =
      rho_geo_lower / (cu_length * eos.baryon_mass()) * baryon_mass_rotns_cgs;
  const double upper_rho_cgs =
      rho_geo_upper / (cu_length * eos.baryon_mass()) * baryon_mass_rotns_cgs;

  const size_t num_points = 5;
  ExportEosForRotNS::dump_eos(eos, num_points, output_file, lower_rho_cgs,
                              upper_rho_cgs, profile_file);
  const auto rows = read_eos_output(output_file);
  file_system::rm(output_file, false);
  file_system::rm(profile_file, false);

  CHECK(rows.size() == num_points);

  // Both T and Y_e are log-linear / linear in log10(rho): cubic polynomial
  // interpolation is exact. Pressure = rho * T * exp(Ye) verifies both.
  const double log10_rho_profile_min = std::log10(rho_profile_min);
  const double log10_rho_profile_max = std::log10(rho_profile_max);
  for (const auto& row : rows) {
    const double n_cgs = std::pow(10.0, row[0]);
    const double rho = n_cgs * cu_length * eos.baryon_mass();
    const double frac = (std::log10(rho) - log10_rho_profile_min) /
                        (log10_rho_profile_max - log10_rho_profile_min);
    const double ye_expected =
        ye_profile_min + frac * (ye_profile_max - ye_profile_min);
    const double T_expected =
        T_profile_min * std::pow(T_profile_max / T_profile_min, frac);
    CHECK(row[2] == approx(ye_expected).epsilon(1.0e-10));
    const double p_geo_expected = rho * T_expected * std::exp(ye_expected);
    CHECK(row[3] ==
          approx(std::log10(p_geo_expected * hydro::units::cgs::pressure_unit))
              .epsilon(1.0e-10));
  }
}

void test_errors() {
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      eos{EquationsOfState::PolytropicFluid<true>{100.0, 2.0}};

  // Output file already exists
  const std::string existing_file = "test_existing_output.dat";
  { const std::ofstream f(existing_file); }
  CHECK_THROWS_WITH(ExportEosForRotNS::dump_eos(eos, 5, existing_file, 1.0e13,
                                                1.0e15, std::nullopt),
                    Catch::Matchers::ContainsSubstring("already exists"));
  file_system::rm(existing_file, false);

  // Thermodynamic profile table that does not cover the requested density range
  {
    const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
        eos_eq{EquationsOfState::IdealFluid<true>{2.0}};
    const std::string t_file = "test_error_narrow_profile.dat";
    if (file_system::check_if_file_exists(t_file)) {
      file_system::rm(t_file, false);
    }
    // Table covers only geometric rho ~ 1e-7 to 1e-6; dump requests 1e-9 to
    // 1e-5 (CGS), which will exceed both ends after unit conversion.
    {
      std::ofstream f(t_file);
      f << "3 0\n1e-7 1e-3\n5e-7 2e-3\n1e-6 3e-3\n";
    }
    const std::string out_file = "test_error_narrow_profile_output.dat";
    if (file_system::check_if_file_exists(out_file)) {
      file_system::rm(out_file, false);
    }
    CHECK_THROWS_WITH(
        ExportEosForRotNS::dump_eos(eos_eq, 5, out_file, 1.0e9, 1.0e15, t_file),
        Catch::Matchers::ContainsSubstring(
            "extends outside the thermal profile table"));
    file_system::rm(t_file, false);
    if (file_system::check_if_file_exists(out_file)) {
      file_system::rm(out_file, false);
    }
  }
}

void test_fallbacks() {
  // Barotropic EoS + thermodynamic profile file: verifies no exception is
  // thrown; the profile is ignored and a warning is printed to stdout.
  {
    const EquationsOfState::Barotropic3D<
        EquationsOfState::PolytropicFluid<true>>
        eos{EquationsOfState::PolytropicFluid<true>{100.0, 2.0}};
    const std::string t_file = "test_fallback_profile.dat";
    if (file_system::check_if_file_exists(t_file)) {
      file_system::rm(t_file, false);
    }
    {
      std::ofstream f(t_file);
      f << "3 0\n1e-8 1e-4\n1e-7 2e-4\n1e-6 3e-4\n";
    }
    const std::string out_file = "test_fallback_barotropic_output.dat";
    if (file_system::check_if_file_exists(out_file)) {
      file_system::rm(out_file, false);
    }
    ExportEosForRotNS::dump_eos(eos, 3, out_file, 1.0e13, 1.0e15, t_file);
    CHECK(file_system::check_if_file_exists(out_file));
    file_system::rm(out_file, false);
    file_system::rm(t_file, false);
  }

  // Non-barotropic EoS without thermodynamic profile: verifies no exception is
  // thrown and that temperature defaults to the EoS lower bound. For
  // IdealFluid Gamma=2, p = rho*T, so p_geometric/rho_geometric == T_lower.
  {
    const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
        eos{EquationsOfState::IdealFluid<true>{2.0}};
    const std::string out_file = "test_fallback_no_profile_output.dat";
    if (file_system::check_if_file_exists(out_file)) {
      file_system::rm(out_file, false);
    }
    ExportEosForRotNS::dump_eos(eos, 5, out_file, 1.0e13, 1.0e15, std::nullopt);
    const auto rows = read_eos_output(out_file);
    file_system::rm(out_file, false);

    const double expected_T = std::max(eos.temperature_lower_bound(), 1.e-100);
    for (const auto& row : rows) {
      const double n_cgs = std::pow(10.0, row[0]);
      const double p_cgs = std::pow(10.0, row[3]);
      const double rho = n_cgs * std::pow(hydro::units::cgs::length_unit, 3.0) *
                         eos.baryon_mass();
      const double p_geometric = p_cgs / hydro::units::cgs::pressure_unit;
      CHECK(p_geometric == approx(rho * expected_T).epsilon(1.0e-12));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Executables.DumpRotNSEos", "[Unit][EquationsOfState]") {
  test_interpolate_profile_polynomial_branch();
  test_interpolate_profile_linear_fallback_branch();
  test_interpolate_profile_boundary_stencil();
  test_barotropic_eos();
  test_eos_with_thermal_profile();
  test_ye_profile();
  test_errors();
  test_fallbacks();
}
