// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/Spectral/Legendre.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Var {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.LiftFlux",
                  "[Unit][NumericalAlgorithms]") {
  const size_t perpendicular_extent = 5;

  const CoordinateMaps::AffineMap xi_map(-1., 1., -5., 7.);
  const CoordinateMaps::AffineMap eta_map(-1., 1., 2., 5.);
  const auto coordinate_map = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::ProductOf2Maps<CoordinateMaps::AffineMap,
                                     CoordinateMaps::AffineMap>(
      xi_map, eta_map));
  const double element_length = (eta_map(std::array<double, 1>{{1.}}) -
                                 eta_map(std::array<double, 1>{{-1.}}))[0];

  const double weight = Basis::lgl::quadrature_weights(perpendicular_extent)[0];

  const Index<1> boundary_extents{{{3}}};
  const DataVector magnitude_of_face_normal =
      magnitude(unnormalized_face_normal(boundary_extents, coordinate_map,
                                         Direction<2>::lower_eta()));

  Variables<tmpl::list<Var>> local_flux(boundary_extents.product());
  get<Var>(local_flux).get() = {1., 2., 3.};
  Variables<tmpl::list<Var>> numerical_flux(boundary_extents.product());
  get<Var>(numerical_flux).get() = {2., 3., 5.};

  const Variables<tmpl::list<Var>> expected =
      -2. / (element_length * weight) * (numerical_flux - local_flux);

  CHECK(dg::lift_flux(local_flux, numerical_flux, perpendicular_extent,
                      magnitude_of_face_normal) ==
        expected);
}
