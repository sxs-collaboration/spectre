// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/KineticEnergyDensity.hpp"
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
template <size_t Dim, typename DataType>
void test_in_databox(const Scalar<DataType>& mass_density,
                     const tnsr::I<DataType, Dim>& velocity) {
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::KineticEnergyDensityCompute<DataType, Dim>>(
      "KineticEnergyDensity");

  const auto box = db::create<
      db::AddSimpleTags<NewtonianEuler::Tags::MassDensity<DataType>,
                        NewtonianEuler::Tags::Velocity<DataType, Dim>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::KineticEnergyDensityCompute<DataType, Dim>>>(
      mass_density, velocity);

  const auto expected_kinetic_energy_density =
      NewtonianEuler::kinetic_energy_density(mass_density, velocity);
  CHECK(db::get<NewtonianEuler::Tags::KineticEnergyDensity<DataType>>(box) ==
        expected_kinetic_energy_density);
}

template <size_t Dim, typename DataType>
void test(const DataType& used_for_size) {
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

  CHECK(Scalar<DataType>{0.5 * get(mass_density) * velocity_squared} ==
        NewtonianEuler::kinetic_energy_density(mass_density, velocity));

  test_in_databox(mass_density, velocity);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.KineticEnergyDensity",
                  "[Unit][Evolution]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test, (1, 2, 3))
}
