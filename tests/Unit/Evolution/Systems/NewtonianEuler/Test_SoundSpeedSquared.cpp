// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SoundSpeedSquared
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SpecificInternalEnergy
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare hydro::Tags::EquationOfState

namespace {

template <typename DataType, typename EquationOfStateType>
void test_compute_item_in_databox(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationOfStateType& equation_of_state) noexcept {
  CHECK(NewtonianEuler::Tags::SoundSpeedSquaredCompute<DataType>::name() ==
        "SoundSpeedSquared");
  CHECK(NewtonianEuler::Tags::SoundSpeedCompute<DataType>::name() ==
        "SoundSpeed");
  const auto box = db::create<
      db::AddSimpleTags<NewtonianEuler::Tags::MassDensity<DataType>,
                        NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                        hydro::Tags::EquationOfState<EquationOfStateType>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::SoundSpeedSquaredCompute<DataType>,
          NewtonianEuler::Tags::SoundSpeedCompute<DataType>>>(
      mass_density, specific_internal_energy, equation_of_state);

  const auto expected_sound_speed_squared = NewtonianEuler::sound_speed_squared(
      mass_density, specific_internal_energy, equation_of_state);
  CHECK(db::get<NewtonianEuler::Tags::SoundSpeedSquared<DataType>>(box) ==
        expected_sound_speed_squared);
  CHECK(db::get<NewtonianEuler::Tags::SoundSpeed<DataType>>(box) ==
        Scalar<DataType>{sqrt(get(expected_sound_speed_squared))});
}

template <typename DataType>
void test_sound_speed_squared(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto mass_density = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_distribution, used_for_size);
  Scalar<DataType> specific_internal_energy{};

  // check with representative equation of state of one independent variable
  EquationsOfState::PolytropicFluid<false> eos_1d(0.003, 4.0 / 3.0);
  specific_internal_energy =
      eos_1d.specific_internal_energy_from_density(mass_density);
  CHECK(Scalar<DataType>{get(eos_1d.chi_from_density(mass_density)) +
                         get(eos_1d.kappa_times_p_over_rho_squared_from_density(
                             mass_density))} ==
        NewtonianEuler::sound_speed_squared(mass_density,
                                            specific_internal_energy, eos_1d));
  test_compute_item_in_databox(mass_density, specific_internal_energy, eos_1d);

  // check with representative equation of state of two independent variables
  EquationsOfState::IdealFluid<false> eos_2d(5.0 / 3.0);
  specific_internal_energy = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_distribution, used_for_size);
  CHECK(Scalar<DataType>{
            get(eos_2d.chi_from_density_and_energy(mass_density,
                                                   specific_internal_energy)) +
            get(eos_2d.kappa_times_p_over_rho_squared_from_density_and_energy(
                mass_density, specific_internal_energy))} ==
        NewtonianEuler::sound_speed_squared(mass_density,
                                            specific_internal_energy, eos_2d));
  test_compute_item_in_databox(mass_density, specific_internal_energy, eos_2d);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.SoundSpeedSquared",
                  "[Unit][Evolution]") {
  test_sound_speed_squared(std::numeric_limits<double>::signaling_NaN());
  test_sound_speed_squared(DataVector(5));
}
