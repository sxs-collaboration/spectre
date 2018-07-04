// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame

namespace {
template <size_t Dim, typename T>
void test_coordinates_compute_item(const domain::Mesh<Dim>& mesh,
                                   T map) noexcept {
  using map_tag = domain::Tags::ElementMap<Dim, Frame::Grid>;
  const auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, map_tag>,
      db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>,
                         domain::Tags::MappedCoordinates<
                             map_tag, domain::Tags::LogicalCoordinates<Dim>>>>(
      mesh,
      domain::ElementMap<Dim, Frame::Grid>(
          domain::ElementId<Dim>(0),
          domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(map)));
  CHECK_ITERABLE_APPROX(
      (db::get<domain::Tags::Coordinates<Dim, Frame::Grid>>(box)),
      (domain::make_coordinate_map<Frame::Logical, Frame::Grid>(map)(
          db::get<domain::Tags::Coordinates<Dim, Frame::Logical>>(box))));

  /// [coordinates_name]
  CHECK(domain::Tags::Coordinates<Dim, Frame::Logical>::name() ==
        "LogicalCoordinates");
  CHECK(domain::Tags::Coordinates<Dim, Frame::Inertial>::name() ==
        "InertialCoordinates");
  /// [coordinates_name]
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinatesTag", "[Unit][Domain]") {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2d = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  test_coordinates_compute_item(
      domain::Mesh<1>{5, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto},
      Affine{-1.0, 1.0, -0.3, 0.7});
  test_coordinates_compute_item(
      domain::Mesh<2>{{{5, 7}},
                      Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto},
      Affine2d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
  test_coordinates_compute_item(
      domain::Mesh<3>{{{5, 6, 9}},
                      Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto},
      Affine3d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}
