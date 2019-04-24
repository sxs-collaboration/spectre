// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <vector>
// IWYU pragma: no_include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

// IWYU pragma: no_forward_declare Tags::dt

namespace {

struct AlfvenWaveProxy : grmhd::Solutions::AlfvenWave {
  using grmhd::Solutions::AlfvenWave::AlfvenWave;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3>& x, double t) const noexcept {
    return variables(x, t, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3>& x, double t) const noexcept {
    return variables(x, t, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto wave = test_creation<grmhd::Solutions::AlfvenWave>(
      "  WaveNumber: 2.2\n"
      "  Pressure: 1.23\n"
      "  RestMassDensity: 0.2\n"
      "  AdiabaticIndex: 1.4\n"
      "  BkgdMagneticField: [0.0, 0.0, 2.0]\n"
      "  WaveMagneticField: [0.75, 0.0, 0.0]");
  CHECK(wave == grmhd::Solutions::AlfvenWave(2.2, 1.23, 0.2, 1.4,
                                             {{0.0, 0.0, 2.0}},
                                             {{0.75, 0.0, 0.0}}));
}

void test_move() noexcept {
  grmhd::Solutions::AlfvenWave wave(3.0, 2.1, 1.3, 1.5, {{0.0, 0.0, 0.24}},
                                    {{0.01, 0.0, 0.0}});
  grmhd::Solutions::AlfvenWave wave_copy(3.0, 2.1, 1.3, 1.5, {{0.0, 0.0, 0.24}},
                                         {{0.01, 0.0, 0.0}});
  test_move_semantics(std::move(wave), wave_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::Solutions::AlfvenWave wave(3.0, 2.1, 1.3, 1.5, {{0.0, 0.0, 0.24}},
                                    {{0.01, 0.0, 0.0}});
  test_serialization(wave);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double wavenumber = 2.1;
  const double pressure = 1.3;
  const double rest_mass_density = 0.4;
  const double adiabatic_index = 4. / 3.;
  const std::array<double, 3> bkgd_magnetic_field = {
      {2.3 * cos(M_PI_4) * cos(0.5 * M_PI_4),
       2.3 * cos(M_PI_4) * sin(0.5 * M_PI_4), 2.3 * sin(M_PI_4)}};
  const std::array<double, 3> wave_magnetic_field = {
      {0.7 * cos(M_PI_4 + M_PI_2) * cos(0.5 * M_PI_4),
       0.7 * cos(M_PI_4 + M_PI_2) * sin(0.5 * M_PI_4),
       0.7 * sin(M_PI_4 + M_PI_2)}};

  pypp::check_with_random_values<
      1, AlfvenWaveProxy::hydro_variables_tags<DataType>>(
      &AlfvenWaveProxy::hydro_variables<DataType>,
      AlfvenWaveProxy(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      bkgd_magnetic_field, wave_magnetic_field),
      "TestFunctions",
      {"alfven_rest_mass_density", "alfven_spatial_velocity",
       "alfven_specific_internal_energy", "alfven_pressure",
       "alfven_lorentz_factor", "alfven_specific_enthalpy"},
      {{{-15., 15.}}},
      std::make_tuple(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      bkgd_magnetic_field, wave_magnetic_field),
      used_for_size);

  pypp::check_with_random_values<
      1, AlfvenWaveProxy::grmhd_variables_tags<DataType>>(
      &AlfvenWaveProxy::grmhd_variables<DataType>,
      AlfvenWaveProxy(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      bkgd_magnetic_field, wave_magnetic_field),
      "TestFunctions",
      {"alfven_rest_mass_density", "alfven_spatial_velocity",
       "alfven_specific_internal_energy", "alfven_pressure",
       "alfven_lorentz_factor", "alfven_specific_enthalpy",
       "alfven_magnetic_field", "alfven_divergence_cleaning_field"},
      {{{-15., 15.}}},
      std::make_tuple(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      bkgd_magnetic_field, wave_magnetic_field),
      used_for_size);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution. Not meant to be extensive.
  grmhd::Solutions::AlfvenWave soln(wavenumber, pressure, rest_mass_density,
                                    adiabatic_index, bkgd_magnetic_field,
                                    wave_magnetic_field);
  const auto coords = make_with_value<tnsr::I<DataType, 3>>(used_for_size, 1.0);
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataType>>(used_for_size, 1.0),
      get<gr::Tags::Lapse<DataType>>(soln.variables(
          coords, 0.0, tmpl::list<gr::Tags::Lapse<DataType>>{})));
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataType>>(used_for_size, 1.0),
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(soln.variables(
          coords, 0.0,
          tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>>{})));
  auto expected_spatial_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(used_for_size,
                                                              0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_spatial_metric.get(i, i) = 1.0;
  }
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(soln.variables(
          coords, 0.0,
          tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);
}

void test_solution() noexcept {
  grmhd::Solutions::AlfvenWave solution(2.2, 1.23, 0.2, 1.4, {{0.0, 0.0, 2.0}},
                                        {{0.75, 0.0, 0.0}});
  const std::array<double, 3> x{{1.0, 2.3, -0.4}};
  const std::array<double, 3> dx{{1.e-4, 1.e-4, 1.e-4}};

  domain::creators::Brick<Frame::Inertial> brick(
      x - dx, x + dx, {{false, false, false}}, {{0, 0, 0}}, {{5, 5, 5}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-10, 1.234,
                        1.e-4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.GrMhd.AlfvenWave",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));

  test_solution();
}
