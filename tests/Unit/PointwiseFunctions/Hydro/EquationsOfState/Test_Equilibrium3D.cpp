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
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"


SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Equilibrium3D",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;

  const double ideal_adiabatic_index = 2.0;
  const double cold_polytropic_index = 2.0;
  const EoS::PolytropicFluid<true> cold_eos{100.0, cold_polytropic_index};
  const EoS::IdealFluid<true> eos_ideal_fluid{ideal_adiabatic_index, 0.0};

  EoS::Equilibrium3D<EoS::IdealFluid<true>> eos_ideal_fluid_3d{eos_ideal_fluid};
  const EoS::HybridEos<EoS::PolytropicFluid<true>> underlying_eos{
      cold_eos, ideal_adiabatic_index};
  EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>> eos{
      underlying_eos};
  CHECK(eos.rest_mass_density_lower_bound() ==
        underlying_eos.rest_mass_density_lower_bound());
  CHECK(eos.rest_mass_density_upper_bound() ==
        underlying_eos.rest_mass_density_upper_bound());
  CHECK(eos.electron_fraction_lower_bound() == 0.0);
  CHECK(eos.electron_fraction_upper_bound() == 1.0);
  CHECK(eos.specific_enthalpy_lower_bound() ==
        underlying_eos.specific_enthalpy_lower_bound());
  CHECK(not eos.is_barotropic());
  CHECK(eos.is_equilibrium());

  {
    const double rest_mass_density = 1e-3;
    const double electron_fraction = 0.1;
    const double underlying_specific_energy_lower_bound =
        underlying_eos.specific_internal_energy_lower_bound(rest_mass_density);
    const double underlying_specific_energy_upper_bound =
        underlying_eos.specific_internal_energy_upper_bound(rest_mass_density);
    CHECK(eos.specific_internal_energy_lower_bound(rest_mass_density,
                                                   electron_fraction) ==
          underlying_specific_energy_lower_bound);
    CHECK(eos.specific_internal_energy_upper_bound(rest_mass_density,
                                                   electron_fraction) ==
          underlying_specific_energy_upper_bound);
    CHECK(eos.temperature_lower_bound() ==
          underlying_eos.temperature_lower_bound());
    CHECK(eos.temperature_upper_bound() ==
          underlying_eos.temperature_upper_bound());
  }
  {
    // DataVector functions
    const Scalar<DataVector> rest_mass_density{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    // electron fraction should be irrelevant
    const Scalar<DataVector> electron_fraction{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    const Scalar<DataVector> temperature{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    const Scalar<DataVector> specific_internal_energy =
        underlying_eos.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature);
    const Scalar<DataVector> pressure =
        underlying_eos.pressure_from_density_and_energy(
            rest_mass_density, specific_internal_energy);
    CHECK(eos.pressure_from_density_and_temperature(
              rest_mass_density, temperature, electron_fraction) == pressure);
    CHECK(eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          underlying_eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy));
    CHECK_ITERABLE_APPROX(
        eos.pressure_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction),
        pressure);
    CHECK_ITERABLE_APPROX(
        eos.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction),
        temperature);
  }
  {
    const Scalar<DataVector> rest_mass_density{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    // electron fraction should be irrelevant
    const Scalar<DataVector> electron_fraction{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    const Scalar<DataVector> temperature{
        DataVector{1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}};
    // For the ideal fluid the sound speed can be computed analytically
    const Scalar<DataVector> pressure =
        eos_ideal_fluid_3d.pressure_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction);
    const Scalar<DataVector> specific_internal_energy =
        eos_ideal_fluid_3d
            .specific_internal_energy_from_density_and_temperature(
                rest_mass_density, temperature, electron_fraction);
    const DataVector enthalpy_density =
        get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
        get(pressure);

    CHECK_ITERABLE_APPROX(
        get(eos_ideal_fluid_3d.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction)),
        (ideal_adiabatic_index)*get(pressure) / enthalpy_density);
  }
  {
    // double functions
    const Scalar<double> rest_mass_density{{1e-3}};
    // temperature and electron fraction should be irrelevant
    const Scalar<double> electron_fraction{{1e-1}};
    const Scalar<double> temperature{{1e-1}};
    const Scalar<double> specific_internal_energy =
        underlying_eos.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature);
    const Scalar<double> pressure =
        underlying_eos.pressure_from_density_and_energy(
            rest_mass_density, specific_internal_energy);
    CHECK(eos.pressure_from_density_and_temperature(
              rest_mass_density, temperature, electron_fraction) == pressure);
    CHECK(eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          underlying_eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy));
    CHECK(eos.temperature_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction) ==
          underlying_eos.temperature_from_density_and_energy(
              rest_mass_density, specific_internal_energy));
  }
  {
    // Using the ideal fluid for sound speed computations
    const Scalar<double> rest_mass_density{{1e-3}};
    // temperature and electron fraction should be irrelevant
    const Scalar<double> electron_fraction{{1e-1}};
    const Scalar<double> temperature{{1e-1}};
    const Scalar<double> pressure =
        eos_ideal_fluid_3d.pressure_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction);
    const Scalar<double> specific_internal_energy =
        eos_ideal_fluid_3d
            .specific_internal_energy_from_density_and_temperature(
                rest_mass_density, temperature, electron_fraction);

    const double enthalpy_density =
        get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
        get(pressure);
    // IdealFluid value is known analytically
    CHECK_ITERABLE_APPROX(
        get(eos_ideal_fluid_3d.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction)),
        (ideal_adiabatic_index)*get(pressure) / enthalpy_density);
    // Check that the sound speed calculation works at zero temperature
    CHECK_ITERABLE_APPROX(
        get(eos.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, Scalar<double>{0.0}, electron_fraction)),
        get(rest_mass_density) /
            (get(cold_eos.pressure_from_density(rest_mass_density)) +
             get(rest_mass_density) *
                 (1.0 + get(cold_eos.specific_internal_energy_from_density(
                            rest_mass_density)))) *
            get(cold_eos.chi_from_density(rest_mass_density)));
  }
  {
    register_derived_classes_with_charm<EoS::EquationOfState<true, 3>>();
    register_derived_classes_with_charm<EoS::EquationOfState<true, 2>>();
    register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
    const auto eos_pointer = serialize_and_deserialize(
        TestHelpers::test_creation<
            std::unique_ptr<EoS::EquationOfState<true, 3>>>(
            {"Equilibrium3D(HybridEos(PolytropicFluid)):\n"
             "  HybridEos:\n"
             "    ThermalAdiabaticIndex: 2.0\n"
             "    PolytropicFluid:\n"
             "      PolytropicConstant: 100.0\n"
             "      PolytropicExponent: 2.0"}));
    const EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>&
        deserialized_eos = dynamic_cast<const EoS::Equilibrium3D<
            EoS::HybridEos<EoS::PolytropicFluid<true>>>&>(*eos_pointer);
    TestHelpers::EquationsOfState::test_get_clone(deserialized_eos);

    CHECK(eos == deserialized_eos);
    CHECK(eos !=
          EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>{
              EoS::HybridEos<EoS::PolytropicFluid<true>>({10.0, 2.0}, 2.0)});
    CHECK(eos !=
          EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>{
              EoS::HybridEos<EoS::PolytropicFluid<true>>({100.0, 1.0}, 2.0)});
    CHECK(eos !=
          EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>{
              EoS::HybridEos<EoS::PolytropicFluid<true>>({100.0, 2.0}, 1.5)});
    EoS::Equilibrium3D<EoS::HybridEos<EoS::Spectral>> eq_spectral{
        EoS::HybridEos<EoS::Spectral>{{1e-5, 1e-4, {2.0, 0.0, 0.0, 0.0}, 1e-2},
                                      2.0}};
    CHECK(eq_spectral.is_equal(eq_spectral));
    CHECK(not(eos.is_equal(eq_spectral)));
  }
}
