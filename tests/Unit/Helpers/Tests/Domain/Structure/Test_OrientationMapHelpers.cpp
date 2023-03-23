// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/Structure/OrientationMapHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace TestHelpers::domain {
namespace {
template <size_t Dim>
void test_valid_orientation_maps() {
  auto orientation_maps = valid_orientation_maps<Dim>();
  constexpr size_t expected_size =
      (Dim == 1 ? 2 : two_to_the(Dim - 1) * factorial(Dim));
  CHECK(orientation_maps.size() == expected_size);
  for (const auto& orientation_map : orientation_maps) {
    if constexpr (Dim > 1) {
      CHECK(get(determinant(discrete_rotation_jacobian(orientation_map))) ==
            1.0);
    }
    CHECK(alg::count(orientation_maps, orientation_map) == 1);
  }
}
template <size_t Dim>
void test_random_orientation_maps(gsl::not_null<std::mt19937*> generator) {
  const auto valid_maps = valid_orientation_maps<Dim>();
  for (size_t s = 0; s < 6; ++s) {
    const auto random_maps = random_orientation_maps<Dim>(s, generator);
    const size_t expected_size = std::min(s, valid_maps.size());
    CHECK(random_maps.size() == expected_size);
    for (const auto& random_map : random_maps) {
      CHECK(alg::count(valid_maps, random_map) == 1);
      CHECK(alg::count(random_maps, random_map) == 1);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("TestHelpers.Domain.OrientationMapHelpers",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test_valid_orientation_maps<1>();
  test_valid_orientation_maps<2>();
  test_valid_orientation_maps<3>();
  test_random_orientation_maps<1>(make_not_null(&generator));
  test_random_orientation_maps<2>(make_not_null(&generator));
  test_random_orientation_maps<3>(make_not_null(&generator));
}
}  // namespace TestHelpers::domain
