// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cmath>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
void test_bulged_cube_fail() {
  INFO("Bulged cube fail");
  const CoordinateMaps::BulgedCube map(2.0 * sqrt(3.0), 0.5, false);

  // The corners of the mapped cube have coordinates +/- 2 in each
  // dimension.  If we inverse-map points that are sufficiently
  // outside of the mapped cube, then the inverse map will fail.
  const std::array<double, 3> test_mapped_point1{{3.0, 3.0, 3.0}};
  const std::array<double, 3> test_mapped_point2{{2.0, 2.0, 2.01}};
  const std::array<double, 3> test_mapped_point3{{4.0, 0.0, 0.0}};

  // These points are outside the mapped BulgedCube. So inverse should either
  // return the correct inverse (which happens to be computable for
  // these points) or it should return boost::none.
  const std::array<double, 3> test_mapped_point4{{2.0, 2.0, 2.0}};
  const std::array<double, 3> test_mapped_point5{{3.0, 0.0, 0.0}};

  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point1)));
  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point2)));
  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point3)));
  if (map.inverse(test_mapped_point4)) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point4).get()),
                          test_mapped_point4);
  }
  if (map.inverse(test_mapped_point5)) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point5).get()),
                          test_mapped_point5);
  }
}

void test_bulged_cube(bool with_equiangular_map) {
  INFO("Bulged cube");
  const std::array<double, 3> lower_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> upper_corner{{1.0, 1.0, 1.0}};
  const CoordinateMaps::BulgedCube map(2.0 * sqrt(3.0), 0.5,
                                       with_equiangular_map);

  CHECK(map(lower_corner)[0] == approx(-2.0));
  CHECK(map(lower_corner)[1] == approx(-2.0));
  CHECK(map(lower_corner)[2] == approx(-2.0));
  CHECK(map(upper_corner)[0] == approx(2.0));
  CHECK(map(upper_corner)[1] == approx(2.0));
  CHECK(map(upper_corner)[2] == approx(2.0));

  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};
  const std::array<double, 3> test_point4{{0.0, 0.0, 0.0}};
  const std::array<DataVector, 3> test_points{
      {DataVector{-1.0, 1.0, 0.7, 0.0}, DataVector{0.25, 1.0, -0.2, 0.0},
       DataVector{0.0, -0.5, 0.4, 0.0}}};
  const DataVector& collocation_pts =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(7);
  const std::array<DataVector, 3> test_points2{
      {collocation_pts, collocation_pts, collocation_pts}};

  test_jacobian(map, test_point1);
  test_jacobian(map, test_point2);
  test_jacobian(map, test_point3);
  test_jacobian(map, test_point4);

  test_inv_jacobian(map, test_point1);
  test_inv_jacobian(map, test_point2);
  test_inv_jacobian(map, test_point3);
  test_inv_jacobian(map, test_point4);

  test_coordinate_map_implementation<CoordinateMaps::BulgedCube>(map);

  CHECK(serialize_and_deserialize(map) == map);
  CHECK_FALSE(serialize_and_deserialize(map) != map);

  test_coordinate_map_argument_types(map, test_point1);

  test_inverse_map(map, test_point1);
  test_inverse_map(map, test_point2);
  test_inverse_map(map, test_point3);
  test_inverse_map(map, test_point4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.BulgedCube", "[Domain][Unit]") {
  const CoordinateMaps::BulgedCube map(sqrt(3.0), 0, false);
  const std::array<double, 3> lower_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> upper_corner{{1.0, 1.0, 1.0}};
  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};

  CHECK_ITERABLE_APPROX(map(lower_corner), lower_corner);
  CHECK_ITERABLE_APPROX(map(upper_corner), upper_corner);
  CHECK_ITERABLE_APPROX(map(test_point1), test_point1);
  CHECK_ITERABLE_APPROX(map(test_point2), test_point2);
  CHECK_ITERABLE_APPROX(map(test_point3), test_point3);

  test_jacobian(map, test_point1);
  test_jacobian(map, test_point2);
  test_jacobian(map, test_point3);

  test_inv_jacobian(map, test_point1);
  test_inv_jacobian(map, test_point2);
  test_inv_jacobian(map, test_point3);

  test_coordinate_map_implementation<CoordinateMaps::BulgedCube>(map);

  CHECK(serialize_and_deserialize(map) == map);
  CHECK_FALSE(serialize_and_deserialize(map) != map);

  test_coordinate_map_argument_types(map, test_point1);
  test_bulged_cube(true);   // Equiangular
  test_bulged_cube(false);  // Equidistant
  test_bulged_cube_fail();

  check_if_map_is_identity(CoordinateMaps::BulgedCube{sqrt(3.0), 0, false});
}
}  // namespace domain
