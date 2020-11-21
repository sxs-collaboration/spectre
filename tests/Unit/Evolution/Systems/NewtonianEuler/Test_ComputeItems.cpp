// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/ComputeItems.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <typename DataType, size_t Dim, typename EquationOfStateType>
void test_compute_items_in_databox(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& velocity,
    const Scalar<DataType>& specific_internal_energy,
    const EquationOfStateType& equation_of_state) noexcept {
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::SoundSpeedSquaredCompute<DataType>>(
      "SoundSpeedSquared");
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::SoundSpeedCompute<DataType>>("SoundSpeed");
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::InternalEnergyDensityCompute<DataType>>(
      "InternalEnergyDensity");
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::KineticEnergyDensityCompute<DataType, Dim>>(
      "KineticEnergyDensity");
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::MachNumberCompute<DataType, Dim>>("MachNumber");
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::RamPressureCompute<DataType, Dim>>("RamPressure");
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::SpecificKineticEnergyCompute<DataType, Dim>>(
      "SpecificKineticEnergy");

  const auto box = db::create<
      db::AddSimpleTags<NewtonianEuler::Tags::MassDensity<DataType>,
                        NewtonianEuler::Tags::Velocity<DataType, Dim>,
                        NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                        hydro::Tags::EquationOfState<EquationOfStateType>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::SoundSpeedSquaredCompute<DataType>,
          NewtonianEuler::Tags::SoundSpeedCompute<DataType>,
          NewtonianEuler::Tags::InternalEnergyDensityCompute<DataType>,
          NewtonianEuler::Tags::KineticEnergyDensityCompute<DataType, Dim>,
          NewtonianEuler::Tags::MachNumberCompute<DataType, Dim>,
          NewtonianEuler::Tags::RamPressureCompute<DataType, Dim>,
          NewtonianEuler::Tags::SpecificKineticEnergyCompute<DataType, Dim>>>(
      mass_density, velocity, specific_internal_energy, equation_of_state);

  const auto expected_sound_speed_squared = NewtonianEuler::sound_speed_squared(
      mass_density, specific_internal_energy, equation_of_state);
  const auto expected_sound_speed =
      Scalar<DataType>{sqrt(get(expected_sound_speed_squared))};
  const auto expected_internal_energy_density =
      NewtonianEuler::internal_energy_density(mass_density,
                                              specific_internal_energy);
  const auto expected_kinetic_energy_density =
      NewtonianEuler::kinetic_energy_density(mass_density, velocity);
  const auto expected_mach_number =
      NewtonianEuler::mach_number(velocity, expected_sound_speed);
  const auto expected_ram_pressure =
      NewtonianEuler::ram_pressure(mass_density, velocity);
  const auto expected_specific_kinetic_energy =
      NewtonianEuler::specific_kinetic_energy(velocity);
  CHECK(db::get<NewtonianEuler::Tags::SoundSpeedSquared<DataType>>(box) ==
        expected_sound_speed_squared);
  CHECK(db::get<NewtonianEuler::Tags::SoundSpeed<DataType>>(box) ==
        expected_sound_speed);
  CHECK(db::get<NewtonianEuler::Tags::InternalEnergyDensity<DataType>>(box) ==
        expected_internal_energy_density);
  CHECK(db::get<NewtonianEuler::Tags::KineticEnergyDensity<DataType>>(box) ==
        expected_kinetic_energy_density);
  CHECK(db::get<NewtonianEuler::Tags::MachNumber<DataType>>(box) ==
        expected_mach_number);
  CHECK(db::get<NewtonianEuler::Tags::RamPressure<DataType, Dim>>(box) ==
        expected_ram_pressure);
  CHECK(db::get<NewtonianEuler::Tags::SpecificKineticEnergy<DataType>>(box) ==
        expected_specific_kinetic_energy);
}

template <size_t Dim, typename DataType>
void test_compute_items(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  std::uniform_real_distribution<> positive_distribution(0.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const auto nn_positive_distribution = make_not_null(&positive_distribution);

  const auto mass_density = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_positive_distribution, used_for_size);
  const auto velocity = make_with_random_values<tnsr::I<DataType, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const DataType velocity_squared = get(dot_product(velocity, velocity));
  Scalar<DataType> specific_internal_energy{};
  Scalar<DataType> sound_speed_squared{};

  CHECK(Scalar<DataType>{0.5 * velocity_squared} ==
        NewtonianEuler::specific_kinetic_energy(velocity));
  CHECK(Scalar<DataType>{0.5 * get(mass_density) * velocity_squared} ==
        NewtonianEuler::kinetic_energy_density(mass_density, velocity));
  auto expected_ram_pressure = make_with_random_values<tnsr::II<DataType, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      expected_ram_pressure.get(i, j) =
          get(mass_density) * velocity.get(i) * velocity.get(j);
    }
  }
  // Re-computation of dependent components in expected_ram_pressure will
  // introduce some small error, so we loosen tolerance a bit.
  CHECK_ITERABLE_CUSTOM_APPROX(
      expected_ram_pressure,
      NewtonianEuler::ram_pressure(mass_density, velocity),
      Approx::custom().epsilon(1.e-10));

  // check with representative equation of state of one independent variable
  EquationsOfState::PolytropicFluid<false> eos_1d(0.003, 4.0 / 3.0);
  specific_internal_energy =
      eos_1d.specific_internal_energy_from_density(mass_density);
  CHECK(Scalar<DataType>{get(mass_density) * get(specific_internal_energy)} ==
        NewtonianEuler::internal_energy_density(mass_density,
                                                specific_internal_energy));
  get(sound_speed_squared) =
      get(eos_1d.chi_from_density(mass_density)) +
      get(eos_1d.kappa_times_p_over_rho_squared_from_density(mass_density));
  CHECK_ITERABLE_CUSTOM_APPROX(
      Scalar<DataType>{sqrt(velocity_squared / get(sound_speed_squared))},
      NewtonianEuler::mach_number(
          velocity, Scalar<DataType>{sqrt(get(sound_speed_squared))}),
      Approx::custom().epsilon(1.e-8));
  CHECK(sound_speed_squared ==
        NewtonianEuler::sound_speed_squared(mass_density,
                                            specific_internal_energy, eos_1d));

  test_compute_items_in_databox(mass_density, velocity,
                                specific_internal_energy, eos_1d);

  // check with representative equation of state of two independent variables
  EquationsOfState::IdealFluid<false> eos_2d(5.0 / 3.0);
  specific_internal_energy = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_positive_distribution, used_for_size);
  CHECK(Scalar<DataType>{get(mass_density) * get(specific_internal_energy)} ==
        NewtonianEuler::internal_energy_density(mass_density,
                                                specific_internal_energy));
  get(sound_speed_squared) =
      get(eos_2d.chi_from_density_and_energy(mass_density,
                                             specific_internal_energy)) +
      get(eos_2d.kappa_times_p_over_rho_squared_from_density_and_energy(
          mass_density, specific_internal_energy));
  CHECK_ITERABLE_CUSTOM_APPROX(
      Scalar<DataType>{sqrt(velocity_squared / get(sound_speed_squared))},
      NewtonianEuler::mach_number(
          velocity, Scalar<DataType>{sqrt(get(sound_speed_squared))}),
      Approx::custom().epsilon(1.e-8));
  CHECK(sound_speed_squared ==
        NewtonianEuler::sound_speed_squared(mass_density,
                                            specific_internal_energy, eos_2d));
  test_compute_items_in_databox(mass_density, velocity,
                                specific_internal_energy, eos_2d);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.TurbulenceComputeItems",
    "[Unit][Evolution]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_items, (1, 2, 3))
}
