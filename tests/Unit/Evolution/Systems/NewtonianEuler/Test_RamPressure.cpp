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
#include "Evolution/Systems/NewtonianEuler/RamPressure.hpp"
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

template <typename DataType, size_t Dim>
void test_in_databox(const Scalar<DataType>& mass_density,
                     const tnsr::I<DataType, Dim>& velocity) noexcept {
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::RamPressureCompute<DataType, Dim>>("RamPressure");

  const auto box = db::create<
      db::AddSimpleTags<NewtonianEuler::Tags::MassDensity<DataType>,
                        NewtonianEuler::Tags::Velocity<DataType, Dim>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::RamPressureCompute<DataType, Dim>>>(
      mass_density, velocity);

  const auto expected_ram_pressure =
      NewtonianEuler::ram_pressure(mass_density, velocity);
  CHECK(db::get<NewtonianEuler::Tags::RamPressure<DataType, Dim>>(box) ==
        expected_ram_pressure);
}

template <size_t Dim, typename DataType>
void test(const DataType& used_for_size) noexcept {
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

  test_in_databox(mass_density, velocity);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.RamPressure",
                  "[Unit][Evolution]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test, (1, 2, 3))
}
