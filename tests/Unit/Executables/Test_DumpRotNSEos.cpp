// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "Executables/ExportEquationOfStateForRotNS/DumpRotNSEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/FileSystem.hpp"

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
                              upper_rho_cgs);
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

void test_errors() {
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      eos{EquationsOfState::PolytropicFluid<true>{100.0, 2.0}};

  // Output file already exists
  const std::string existing_file = "test_existing_output.dat";
  { const std::ofstream f(existing_file); }
  CHECK_THROWS_WITH(
      ExportEosForRotNS::dump_eos(eos, 5, existing_file, 1.0e13, 1.0e15),
      Catch::Matchers::ContainsSubstring("already exists"));
  file_system::rm(existing_file, false);
}

void test_fallbacks() {
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
    ExportEosForRotNS::dump_eos(eos, 5, out_file, 1.0e13, 1.0e15);
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
  test_barotropic_eos();
  test_errors();
  test_fallbacks();
}
