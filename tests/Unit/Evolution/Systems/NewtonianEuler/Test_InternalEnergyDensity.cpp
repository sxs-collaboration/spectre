// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/InternalEnergyDensity.hpp"
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
template <typename DataType>
void test_in_databox(const Scalar<DataType>& mass_density,
                     const Scalar<DataType>& specific_internal_energy) {
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::InternalEnergyDensityCompute<DataType>>(
      "InternalEnergyDensity");
  const auto box = db::create<
      db::AddSimpleTags<NewtonianEuler::Tags::MassDensity<DataType>,
                        NewtonianEuler::Tags::SpecificInternalEnergy<DataType>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::InternalEnergyDensityCompute<DataType>>>(
      mass_density, specific_internal_energy);

  const auto expected_internal_energy_density =
      NewtonianEuler::internal_energy_density(mass_density,
                                              specific_internal_energy);
  CHECK(db::get<NewtonianEuler::Tags::InternalEnergyDensity<DataType>>(box) ==
        expected_internal_energy_density);
}

template <typename DataType>
void test(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  std::uniform_real_distribution<> positive_distribution(0.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_positive_distribution = make_not_null(&positive_distribution);

  const auto mass_density = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_positive_distribution, used_for_size);
  Scalar<DataType> specific_internal_energy{};

  // check with representative equation of state of one independent variable
  EquationsOfState::PolytropicFluid<false> eos_1d(0.003, 4.0 / 3.0);
  specific_internal_energy =
      eos_1d.specific_internal_energy_from_density(mass_density);
  CHECK(Scalar<DataType>{get(mass_density) * get(specific_internal_energy)} ==
        NewtonianEuler::internal_energy_density(mass_density,
                                                specific_internal_energy));

  test_in_databox(mass_density, specific_internal_energy);

  // check with representative equation of state of two independent variables
  EquationsOfState::IdealFluid<false> eos_2d(5.0 / 3.0);
  specific_internal_energy = make_with_random_values<Scalar<DataType>>(
      nn_generator, nn_positive_distribution, used_for_size);
  CHECK(Scalar<DataType>{get(mass_density) * get(specific_internal_energy)} ==
        NewtonianEuler::internal_energy_density(mass_density,
                                                specific_internal_energy));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.InternalEnergyDensity",
                  "[Unit][Evolution]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  test(d);
  test(dv);
}
