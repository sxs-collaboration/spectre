// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Barotropic3D",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  const EoS::PolytropicFluid<true> underlying_eos{100.0, 2.0};
  EoS::Barotropic3D<EoS::PolytropicFluid<true>> eos{underlying_eos};
  CHECK(eos.rest_mass_density_lower_bound() ==
        underlying_eos.rest_mass_density_lower_bound());
  CHECK(eos.rest_mass_density_upper_bound() ==
        underlying_eos.rest_mass_density_upper_bound());
  CHECK(eos.electron_fraction_lower_bound() == 0.0);
  CHECK(eos.electron_fraction_upper_bound() == 1.0);
  CHECK(eos.temperature_lower_bound() == 0.0);
  CHECK(eos.temperature_upper_bound() == std::numeric_limits<double>::max());
  CHECK(eos.is_barotropic());
  CHECK(not eos.is_equilibrium());
  {
    // DataVector functions
    const Scalar<DataVector> rest_mass_density{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    // temperature and electron fraction should be irrelevant
    const Scalar<DataVector> electron_fraction{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    const Scalar<DataVector> temperature{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    const Scalar<DataVector> specific_internal_energy =
        underlying_eos.specific_internal_energy_from_density(rest_mass_density);
    const Scalar<DataVector> pressure =
        underlying_eos.pressure_from_density(rest_mass_density);
    CHECK(eos.pressure_from_density_and_temperature(
              rest_mass_density, temperature, electron_fraction) == pressure);
    CHECK(eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          underlying_eos.pressure_from_density(rest_mass_density));
    // The energy is independent of energy, so this is a garabage value, but it
    // is taken to be zero
    CHECK(eos.temperature_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0));
    DataVector enthalpy_density =
        get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
        get(pressure);
    CHECK(eos.sound_speed_squared_from_density_and_temperature(
              rest_mass_density, temperature, electron_fraction) ==
          Scalar<DataVector>{
              get(rest_mass_density) *
              get(underlying_eos.chi_from_density(rest_mass_density)) /
              enthalpy_density});
  }

  {
    // double functions
    const Scalar<double> rest_mass_density{{1e-10}};
    // temperature and electron fraction should be irrelevant
    const Scalar<double> electron_fraction{{1e-10}};
    const Scalar<double> temperature{{1e-10}};
    const Scalar<double> specific_internal_energy =
        underlying_eos.specific_internal_energy_from_density(rest_mass_density);
    const Scalar<double> pressure =
        underlying_eos.pressure_from_density(rest_mass_density);
    CHECK(eos.pressure_from_density_and_temperature(
              rest_mass_density, temperature, electron_fraction) == pressure);
    CHECK(eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          underlying_eos.pressure_from_density(rest_mass_density));
    // The energy is independent of energy, so this is a garabage value, but it
    // is taken to be zero
    CHECK(eos.temperature_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          make_with_value<Scalar<double>>(rest_mass_density, 0.0));
    double enthalpy_density =
        get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
        get(pressure);
    CHECK(
        eos.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction) ==
        Scalar<double>{get(rest_mass_density) *
                       get(underlying_eos.chi_from_density(rest_mass_density)) /
                       enthalpy_density});
  }
  {
    register_derived_classes_with_charm<EoS::EquationOfState<true, 3>>();
    register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
    const auto eos_pointer = serialize_and_deserialize(
        TestHelpers::test_creation<
            std::unique_ptr<EoS::EquationOfState<true, 3>>>(
            {"Barotropic3D(PolytropicFluid):\n"
             "  PolytropicFluid:\n"
             "    PolytropicConstant: 100.0\n"
             "    PolytropicExponent: 2.0"}));
    const EoS::Barotropic3D<EoS::PolytropicFluid<true>>& deserialized_eos =
        dynamic_cast<const EoS::Barotropic3D<EoS::PolytropicFluid<true>>&>(
            *eos_pointer);
    TestHelpers::EquationsOfState::test_get_clone(deserialized_eos);

    CHECK(eos == deserialized_eos);
    CHECK(eos != EoS::Barotropic3D<EoS::PolytropicFluid<true>>{
                     EoS::PolytropicFluid<true>(10.0, 2.0)});
    CHECK(eos != EoS::Barotropic3D<EoS::PolytropicFluid<true>>{
                     EoS::PolytropicFluid<true>(100.0, 1.0)});
    CHECK(not eos.is_equal(EoS::Barotropic3D<EoS::Spectral>{
        EoS::Spectral(1e-5, 1e-4, {2.0, 0.0, 0.0}, 1e-2)}));
  }
}
