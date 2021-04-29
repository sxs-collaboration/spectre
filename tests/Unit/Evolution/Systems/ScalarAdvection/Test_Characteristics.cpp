// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Characteristics.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"

namespace {
template <size_t Dim>
void test_characteristics(const gsl::not_null<std::mt19937*> gen) noexcept {
  // test for tags
  TestHelpers::db::test_compute_tag<
      ScalarAdvection::Tags::LargestCharacteristicSpeedCompute<Dim>>(
      "LargestCharacteristicSpeed");

  // test for randomly generated velocity field
  const DataVector used_for_size(10);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto velocity_field = make_with_random_values<tnsr::I<DataVector, Dim>>(
      gen, make_not_null(&distribution), used_for_size);
  double largest_characteristic_speed =
      std::numeric_limits<double>::signaling_NaN();
  ScalarAdvection::Tags::LargestCharacteristicSpeedCompute<Dim>::function(
      make_not_null(&largest_characteristic_speed), velocity_field);
  CHECK(largest_characteristic_speed ==
        max(get(magnitude<DataVector>(velocity_field))));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Characteristics",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  test_characteristics<1>(make_not_null(&gen));
  test_characteristics<2>(make_not_null(&gen));
}
