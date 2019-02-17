// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.LiftFlux",
                  "[Unit][NumericalAlgorithms]") {
  const Mesh<2> mesh{
      {{3, 5}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map(-1., 1., -5., 7.);
  const Affine eta_map(-1., 1., 2., 5.);
  const auto coordinate_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                 eta_map));
  const double element_length = (eta_map(std::array<double, 1>{{1.}}) -
                                 eta_map(std::array<double, 1>{{-1.}}))[0];

  const double weight = Spectral::quadrature_weights(mesh.slice_through(1))[0];
  const auto boundary_mesh = mesh.slice_through(0);

  const auto magnitude_of_face_normal = magnitude(unnormalized_face_normal(
      boundary_mesh, coordinate_map, Direction<2>::lower_eta()));

  Variables<tmpl::list<Tags::NormalDotNumericalFlux<Var>>> flux(
      boundary_mesh.number_of_grid_points());
  get(get<Tags::NormalDotNumericalFlux<Var>>(flux)) = DataVector({2., 3., 5.});

  const Variables<tmpl::list<Var>> expected =
      -2. / (element_length * weight) * flux;

  CHECK(dg::lift_flux(flux, mesh.extents(1), magnitude_of_face_normal) ==
        expected);
}
