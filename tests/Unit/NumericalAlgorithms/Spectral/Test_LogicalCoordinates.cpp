// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace domain {
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.LogicalCoordinates",
                  "[Domain][Unit]") {
  using Affine2d = CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<
      CoordinateMaps::Affine, CoordinateMaps::Affine, CoordinateMaps::Affine>;

  const Mesh<1> mesh_1d{3, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{
      {{2, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  // [logical_coordinates_example]
  const Mesh<3> mesh_3d{{{5, 3, 2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const CoordinateMaps::Affine x_map{-1.0, 1.0, -3.0, 7.0};
  const CoordinateMaps::Affine y_map{-1.0, 1.0, -13.0, 47.0};
  const CoordinateMaps::Affine z_map{-1.0, 1.0, -32.0, 74.0};

  const auto map_3d = make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
      Affine3d{x_map, y_map, z_map});

  const auto x_3d = map_3d(logical_coordinates(mesh_3d));
  // [logical_coordinates_example]

  const auto map_1d = make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
      CoordinateMaps::Affine{x_map});
  const auto map_2d = make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
      Affine2d{x_map, y_map});
  const auto x_1d = map_1d(logical_coordinates(mesh_1d));
  const auto x_2d = map_2d(logical_coordinates(mesh_2d));

  CHECK(x_1d[0][0] == -3.0);
  CHECK(x_1d[0][1] == 2.0);
  CHECK(x_1d[0][2] == 7.0);

  CHECK(x_2d[0][0] == -3.0);
  CHECK(x_2d[0][1] == 7.0);

  CHECK(x_3d[0][0] == -3.0);
  CHECK(x_3d[0][2] == 2.0);
  CHECK(x_3d[0][4] == 7.0);

  CHECK(x_2d[1][0] == -13.0);
  CHECK(x_2d[1][2] == 17.0);
  CHECK(x_2d[1][4] == 47.0);

  CHECK(x_3d[1][0] == -13.0);
  CHECK(x_3d[1][5] == 17.0);
  CHECK(x_3d[1][10] == 47.0);

  CHECK(x_3d[2][0] == -32.0);
  CHECK(x_3d[2][15] == 74.0);

  TestHelpers::db::test_compute_tag<Tags::LogicalCoordinates<1>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_compute_tag<Tags::LogicalCoordinates<2>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_compute_tag<Tags::LogicalCoordinates<3>>(
      "ElementLogicalCoordinates");
}
}  // namespace domain
