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
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"

namespace {

void test_spherical_logical_coords() {
  for (size_t l = 2; l < 5; ++l) {
    const size_t nth = l + 1;
    const size_t nph = 2 * l + 1;
    const Mesh<2> mesh_s2{
        {nth, nph},
        {Spectral::Basis::SphericalHarmonic,
         Spectral::Basis::SphericalHarmonic},
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
    const auto xi = logical_coordinates(mesh_s2);
    const ylm::Spherepack ylm(l, l);
    const auto xi_expected = ylm.theta_phi_points();
    CHECK(get<0>(xi) == xi_expected[0]);
    CHECK(get<1>(xi) == xi_expected[1]);
  }
}

void test_radial_zernike_logical_coords() {
  for (const auto& basis :
       {Spectral::Basis::ZernikeB1, Spectral::Basis::ZernikeB2,
        Spectral::Basis::ZernikeB3}) {
    for (size_t n = 2; n < 5; ++n) {
      const Mesh<1> mesh{n, basis, Spectral::Quadrature::GaussRadauUpper};
      const auto xi = logical_coordinates(mesh);
      CHECK(get<0>(xi)[n - 1] == approx(1.0));
      if (n >= 4) {
        // logical coordinates in [-1, 1] (with the mapping to [0, 1]
        // internally taken care of)
        CHECK(get<0>(xi)[0] < 0.0);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.LogicalCoordinates",
                  "[Domain][Unit]") {
  test_spherical_logical_coords();
  test_radial_zernike_logical_coords();
  using Affine2d =
      domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;

  const Mesh<1> mesh_1d{3, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{
      {{2, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  // [logical_coordinates_example]
  const Mesh<3> mesh_3d{{{5, 3, 2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const domain::CoordinateMaps::Affine x_map{-1.0, 1.0, -3.0, 7.0};
  const domain::CoordinateMaps::Affine y_map{-1.0, 1.0, -13.0, 47.0};
  const domain::CoordinateMaps::Affine z_map{-1.0, 1.0, -32.0, 74.0};

  const auto map_3d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine3d{x_map, y_map, z_map});

  const auto x_3d = map_3d(logical_coordinates(mesh_3d));
  // [logical_coordinates_example]

  const auto map_1d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          domain::CoordinateMaps::Affine{x_map});
  const auto map_2d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
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

  const Mesh<3> mesh_spherical{
      {3, 1, 1},
      {Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
       Spectral::Basis::Cartoon},
      {Spectral::Quadrature::GaussLobatto,
       Spectral::Quadrature::SphericalSymmetry,
       Spectral::Quadrature::SphericalSymmetry}};
  using Affine = domain::CoordinateMaps::Affine;
  using Identity = domain::CoordinateMaps::Identity<1>;
  const Identity identity_cartoon_map;
  const Affine affine_x_map(-1.0, 1.0, -1.0, 1.0);
  const domain::CoordinateMap<
      Frame::ElementLogical, Frame::Grid,
      domain::CoordinateMaps::ProductOf3Maps<Affine, Identity, Identity>>
      map_spherical{{affine_x_map, identity_cartoon_map, identity_cartoon_map}};
  const auto x_spherical = map_spherical(logical_coordinates(mesh_spherical));
  CHECK(x_spherical[1][0] == 0.0);
  CHECK(x_spherical[1][1] == 0.0);
  CHECK(x_spherical[1][2] == 0.0);
  CHECK(x_spherical[2][0] == 0.0);
  CHECK(x_spherical[2][1] == 0.0);
  CHECK(x_spherical[2][2] == 0.0);

  const Mesh<3> mesh_axial{
      {2, 3, 1},
      {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
       Spectral::Basis::Cartoon},
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
       Spectral::Quadrature::AxialSymmetry}};
  const domain::CoordinateMap<
      Frame::ElementLogical, Frame::Grid,
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity>>
      map_axial{{affine_x_map, affine_x_map, identity_cartoon_map}};
  const auto x_axial = map_axial(logical_coordinates(mesh_axial));
  CHECK(x_axial[2][0] == 0.0);
  CHECK(x_axial[2][1] == 0.0);
  CHECK(x_axial[2][2] == 0.0);
  CHECK(x_axial[2][3] == 0.0);
  CHECK(x_axial[2][4] == 0.0);
  CHECK(x_axial[2][5] == 0.0);

  const Mesh<1> mesh_cartoon_error{2, Spectral::Basis::Cartoon,
                                   Spectral::Quadrature::AxialSymmetry};
  CHECK_THROWS_WITH((logical_coordinates(mesh_cartoon_error)),
                    Catch::Matchers::ContainsSubstring(
                        "Only 1 grid point is allowed in a Cartoon basis."));

  TestHelpers::db::test_compute_tag<domain::Tags::LogicalCoordinates<1>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_compute_tag<domain::Tags::LogicalCoordinates<2>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_compute_tag<domain::Tags::LogicalCoordinates<3>>(
      "ElementLogicalCoordinates");
}
}  // namespace
