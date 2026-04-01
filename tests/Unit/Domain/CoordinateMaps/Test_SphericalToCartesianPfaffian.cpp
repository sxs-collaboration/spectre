// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain {
namespace {
void test_map_at_point(const CoordinateMaps::SphericalToCartesianPfaffian& map,
                       const std::array<double, 3>& source_point,
                       const std::array<double, 3>& target_point) {
  test_inverse_map(map, source_point);
  if (source_point !=
      std::array{0.0, 0.5 * M_PI, 0.0}) {  // inv jac singular at origin
    test_coordinate_map_argument_types(map, source_point);
    test_inv_jacobian(map, source_point);
  }
  CHECK_ITERABLE_APPROX(map(source_point), target_point);
  CHECK_ITERABLE_APPROX(map.inverse(target_point).value(), source_point);
}

void test_map(const CoordinateMaps::SphericalToCartesianPfaffian& map) {
  CHECK(not map.is_identity());
  CHECK_FALSE(map != map);
  test_serialization(map);
  const std::array<double, 3> source_origin{{0.0, 0.5 * M_PI, 0.0}};
  const std::array<double, 3> target_origin{{0.0, 0.0, 0.0}};
  test_map_at_point(map, source_origin, target_origin);
  const std::array<double, 3> source_north_pole{{0.5, 0.0, 0.0}};
  const std::array<double, 3> target_north_pole{{0.0, 0.0, 0.5}};
  test_map_at_point(map, source_north_pole, target_north_pole);
  const std::array<double, 3> source_south_pole{{0.25, M_PI, 0.0}};
  const std::array<double, 3> target_south_pole{{0.0, 0.0, -0.25}};
  test_map_at_point(map, source_south_pole, target_south_pole);
  test_map_at_point(map, {{0.0, M_PI_2, 0.0}}, {{0.0, 0.0, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, 0.0}}, {{1.0, 0.0, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, M_PI_4}},
                    {{M_SQRT1_2, M_SQRT1_2, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, M_PI_2}}, {{0.0, 1.0, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, 3.0 * M_PI_4}},
                    {{-M_SQRT1_2, M_SQRT1_2, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, M_PI}}, {{-1.0, 0.0, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, 5.0 * M_PI_4}},
                    {{-M_SQRT1_2, -M_SQRT1_2, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, 3.0 * M_PI_2}}, {{0.0, -1.0, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, 7.0 * M_PI_4}},
                    {{M_SQRT1_2, -M_SQRT1_2, 0.0}});
  test_map_at_point(map, {{1.0, M_PI_2, M_PI}},
                    {{-1.0, std::copysign(0.0, -1.0), 0.0}});
  const Mesh<3> mesh{
      {5, 3, 5},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
  const auto xi = logical_coordinates(mesh);
  const std::array<DataVector, 3> source_coords{xi[0] + 2.0, xi[1], xi[2]};
  test_inv_jacobian(map, source_coords);
}

void test() {
  const CoordinateMaps::SphericalToCartesianPfaffian original_map{};
  test_map(original_map);
  const auto serialized_map = serialize_and_deserialize(original_map);
  test_map(serialized_map);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericalToCartesianPfaffian",
                  "[Domain][Unit]") {
  test();
}
}  // namespace domain
