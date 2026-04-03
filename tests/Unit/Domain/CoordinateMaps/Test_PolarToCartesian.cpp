// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain {
namespace {
void test_map_at_point(const CoordinateMaps::PolarToCartesian& map,
                       const std::array<double, 2>& source_point,
                       const std::array<double, 2>& target_point) {
  test_inverse_map(map, source_point);
  if (source_point != std::array{0.0, 0.0}) {  // inv jac singular at origin
    test_coordinate_map_argument_types(map, source_point);
    test_inv_jacobian(map, source_point);
  }
  CAPTURE(source_point);
  CAPTURE(target_point);
  CHECK_ITERABLE_APPROX(map(source_point), target_point);
  CHECK_ITERABLE_APPROX(map.inverse(target_point).value(), source_point);
}

void test_map(const CoordinateMaps::PolarToCartesian& map) {
  CHECK(not map.is_identity());
  CHECK_FALSE(map != map);
  test_serialization(map);
  test_map_at_point(map, {{0.0, 0.0}}, {{0.0, 0.0}});
  test_map_at_point(map, {{1.0, 0.0}}, {{1.0, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_4}}, {{M_SQRT1_2, M_SQRT1_2}});
  test_map_at_point(map, {{1.0, M_PI_2}}, {{0.0, 1.0}});
  test_map_at_point(map, {{1.0, 3.0 * M_PI_4}}, {{-M_SQRT1_2, M_SQRT1_2}});
  test_map_at_point(map, {{1.0, M_PI}}, {{-1.0, 0.0}});
  test_map_at_point(map, {{1.0, 5.0 * M_PI_4}}, {{-M_SQRT1_2, -M_SQRT1_2}});
  test_map_at_point(map, {{1.0, 3.0 * M_PI_2}}, {{0.0, -1.0}});
  test_map_at_point(map, {{1.0, 7.0 * M_PI_4}}, {{M_SQRT1_2, -M_SQRT1_2}});
  test_map_at_point(map, {{1.0, M_PI}}, {{-1.0, std::copysign(0.0, -1.0)}});
}

void test() {
  const CoordinateMaps::PolarToCartesian original_map{};
  test_map(original_map);
  const auto serialized_map = serialize_and_deserialize(original_map);
  test_map(serialized_map);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.PolarToCartesian",
                  "[Domain][Unit]") {
  test();
}
}  // namespace domain
