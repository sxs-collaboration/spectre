// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/SpecificKineticEnergy.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <typename DataType, size_t Dim>
void test_in_databox(const tnsr::I<DataType, Dim>& velocity) noexcept {
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::SpecificKineticEnergyCompute<DataType, Dim>>(
      "SpecificKineticEnergy");

  const auto box = db::create<
      db::AddSimpleTags<NewtonianEuler::Tags::Velocity<DataType, Dim>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::SpecificKineticEnergyCompute<DataType, Dim>>>(
      velocity);

  const auto expected_specific_kinetic_energy =
      NewtonianEuler::specific_kinetic_energy(velocity);
  CHECK(db::get<NewtonianEuler::Tags::SpecificKineticEnergy<DataType>>(box) ==
        expected_specific_kinetic_energy);
}

template <size_t Dim, typename DataType>
void test(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  std::uniform_real_distribution<> positive_distribution(0.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto velocity = make_with_random_values<tnsr::I<DataType, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const DataType velocity_squared = get(dot_product(velocity, velocity));

  CHECK(Scalar<DataType>{0.5 * velocity_squared} ==
        NewtonianEuler::specific_kinetic_energy(velocity));

  test_in_databox(velocity);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.SpecificKineticEnergy",
                  "[Unit][Evolution]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test, (1, 2, 3))
}
