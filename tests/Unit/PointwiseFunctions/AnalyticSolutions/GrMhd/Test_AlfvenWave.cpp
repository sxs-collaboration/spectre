// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

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
      "  BackgroundMagField: 2.0\n"
      "  PerturbationSize: 0.75");
  CHECK(wave.wavenumber() == 2.2);
  CHECK(wave.pressure() == 1.23);
  CHECK(wave.rest_mass_density() == 0.2);
  CHECK(wave.adiabatic_index() == 1.4);
  CHECK(wave.background_mag_field() == 2.0);
  CHECK(wave.perturbation_size() == 0.75);
}

void test_move() noexcept {
  grmhd::Solutions::AlfvenWave wave(3.0, 2.1, 1.3, 1.5, 0.24, 0.01);
  grmhd::Solutions::AlfvenWave wave_copy(3.0, 2.1, 1.3, 1.5, 0.24, 0.01);
  test_move_semantics(std::move(wave), wave_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::Solutions::AlfvenWave wave(3.0, 2.1, 1.3, 1.5, 0.24, 0.01);
  test_serialization(wave);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double wavenumber = 2.1;
  const double pressure = 1.3;
  const double rest_mass_density = 0.4;
  const double adiabatic_index = 4. / 3.;
  const double background_mag_field = 2.3;
  const double perturbation_size = 0.78;

  pypp::check_with_random_values<
      1, AlfvenWaveProxy::hydro_variables_tags<DataType>>(
      &AlfvenWaveProxy::hydro_variables<DataType>,
      AlfvenWaveProxy(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      background_mag_field, perturbation_size),
      "TestFunctions",
      {"alfven_rest_mass_density", "alfven_spatial_velocity",
       "alfven_specific_internal_energy", "alfven_pressure",
       "alfven_lorentz_factor", "alfven_specific_enthalpy"},
      {{{-15., 15.}}},
      std::make_tuple(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      background_mag_field, perturbation_size),
      used_for_size);

  pypp::check_with_random_values<
      1, AlfvenWaveProxy::grmhd_variables_tags<DataType>>(
      &AlfvenWaveProxy::grmhd_variables<DataType>,
      AlfvenWaveProxy(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      background_mag_field, perturbation_size),
      "TestFunctions",
      {"alfven_rest_mass_density", "alfven_spatial_velocity",
       "alfven_specific_internal_energy", "alfven_pressure",
       "alfven_lorentz_factor", "alfven_specific_enthalpy",
       "alfven_magnetic_field", "alfven_divergence_cleaning_field"},
      {{{-15., 15.}}},
      std::make_tuple(wavenumber, pressure, rest_mass_density, adiabatic_index,
                      background_mag_field, perturbation_size),
      used_for_size);
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
}
