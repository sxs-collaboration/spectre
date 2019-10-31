// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {

template <typename DataType, typename EquationOfStateType>
void test_compute_item_in_databox(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& specific_enthalpy,
    const EquationOfStateType& equation_of_state) noexcept {
  CHECK(hydro::Tags::SoundSpeedSquaredCompute<DataType>::name() ==
        "SoundSpeedSquared");
  const auto box = db::create<
      db::AddSimpleTags<hydro::Tags::RestMassDensity<DataType>,
                        hydro::Tags::SpecificInternalEnergy<DataType>,
                        hydro::Tags::SpecificEnthalpy<DataType>,
                        hydro::Tags::EquationOfState<EquationOfStateType>>,
      db::AddComputeTags<hydro::Tags::SoundSpeedSquaredCompute<DataType>>>(
      rest_mass_density, specific_internal_energy, specific_enthalpy,
      equation_of_state);

  const auto expected_sound_speed_squared =
      hydro::sound_speed_squared(rest_mass_density, specific_internal_energy,
                                 specific_enthalpy, equation_of_state);
  CHECK(db::get<hydro::Tags::SoundSpeedSquared<DataType>>(box) ==
        expected_sound_speed_squared);
}

template <typename DataType>
void test_sound_speed_squared(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto rest_mass_density = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_distribution, used_for_size);
  Scalar<DataType> specific_internal_energy{};
  Scalar<DataType> specific_enthalpy{};

  // check with representative equation of state of one independent variable
  EquationsOfState::PolytropicFluid<true> eos_1d(0.003, 4.0 / 3.0);
  specific_internal_energy =
      eos_1d.specific_internal_energy_from_density(rest_mass_density);
  specific_enthalpy = eos_1d.specific_enthalpy_from_density(rest_mass_density);
  CHECK(
      Scalar<DataType>{(get(eos_1d.chi_from_density(rest_mass_density)) +
                        get(eos_1d.kappa_times_p_over_rho_squared_from_density(
                            rest_mass_density))) /
                       get(specific_enthalpy)} ==
      hydro::sound_speed_squared(rest_mass_density, specific_internal_energy,
                                 specific_enthalpy, eos_1d));
  test_compute_item_in_databox(rest_mass_density, specific_internal_energy,
                               specific_enthalpy, eos_1d);

  // check with representative equation of state of two independent variables
  EquationsOfState::IdealFluid<true> eos_2d(5.0 / 3.0);
  specific_internal_energy = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_distribution, used_for_size);
  specific_enthalpy = eos_2d.specific_enthalpy_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
  CHECK(Scalar<DataType>{
            (get(eos_2d.chi_from_density_and_energy(rest_mass_density,
                                                    specific_internal_energy)) +
             get(eos_2d.kappa_times_p_over_rho_squared_from_density_and_energy(
                 rest_mass_density, specific_internal_energy))) /
            get(specific_enthalpy)} ==
        hydro::sound_speed_squared(rest_mass_density, specific_internal_energy,
                                   specific_enthalpy, eos_2d));
  test_compute_item_in_databox(rest_mass_density, specific_internal_energy,
                               specific_enthalpy, eos_2d);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.SoundSpeedSquared",
                  "[Unit][Evolution]") {
  test_sound_speed_squared(std::numeric_limits<double>::signaling_NaN());
  test_sound_speed_squared(DataVector(5));
}
