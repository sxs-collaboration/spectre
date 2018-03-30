// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"

namespace {
template <size_t Dim, typename T>
void test_coordinates_compute_item(Index<Dim> extents, T map) noexcept {
  using map_tag = Tags::ElementMap<Dim, Frame::Grid>;
  const auto box = db::create<
      db::AddSimpleTags<Tags::Extents<Dim>, map_tag>,
      db::AddComputeTags<
          Tags::LogicalCoordinates<Dim>,
          Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>>>(
      extents, ElementMap<Dim, Frame::Grid>(
                   ElementId<Dim>(0),
                   make_coordinate_map_base<Frame::Logical, Frame::Grid>(map)));
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>>(box)),
      (make_coordinate_map<Frame::Logical, Frame::Grid>(map)(
          db::get<Tags::LogicalCoordinates<Dim>>(box))));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinatesTag", "[Unit][Domain]") {
  using Affine = CoordinateMaps::Affine;
  using Affine2d = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  test_coordinates_compute_item(Index<1>{5}, Affine{-1.0, 1.0, -0.3, 0.7});
  test_coordinates_compute_item(
      Index<2>{5, 7},
      Affine2d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
  test_coordinates_compute_item(
      Index<3>{5, 6, 9},
      Affine3d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}
