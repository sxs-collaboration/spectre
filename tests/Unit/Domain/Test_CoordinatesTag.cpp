// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace Frame {
struct Grid;
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain {
namespace {
template <size_t Dim, typename T>
void test_coordinates_compute_item(const Mesh<Dim>& mesh, T map) {
  using map_tag = Tags::ElementMap<Dim, Frame::Grid>;
  const auto box = db::create<
      db::AddSimpleTags<Tags::Mesh<Dim>, map_tag>,
      db::AddComputeTags<
          Tags::LogicalCoordinates<Dim>,
          Tags::MappedCoordinates<map_tag, Tags::LogicalCoordinates<Dim>>>>(
      mesh,
      ElementMap<Dim, Frame::Grid>(
          ElementId<Dim>(0),
          make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)));
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::Coordinates<Dim, Frame::Grid>>(box)),
      (make_coordinate_map<Frame::ElementLogical, Frame::Grid>(map)(
          db::get<Tags::Coordinates<Dim, Frame::ElementLogical>>(box))));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinatesTag", "[Unit][Domain]") {
  using Affine = CoordinateMaps::Affine;
  using Affine2d = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  test_coordinates_compute_item(
      Mesh<1>{5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      Affine{-1.0, 1.0, -0.3, 0.7});
  test_coordinates_compute_item(
      Mesh<2>{{{5, 7}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      Affine2d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
  test_coordinates_compute_item(
      Mesh<3>{{{5, 6, 9}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      Affine3d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}
}  // namespace domain
