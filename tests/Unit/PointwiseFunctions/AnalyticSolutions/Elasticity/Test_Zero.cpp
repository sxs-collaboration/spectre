// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/Zero.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
auto make_coord_map() {
  using AffineMap = domain::CoordinateMaps::Affine;
  if constexpr (Dim == 1) {
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap>{{-1., 1., 0., M_PI}};
  } else if constexpr (Dim == 2) {
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap2D>{
        {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
  } else {
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap3D>{
        {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
  }
}

template <size_t Dim>
void test_solution() {
  const Elasticity::Solutions::Zero<Dim> solution{};
  const auto created_solution =
      TestHelpers::test_creation<Elasticity::Solutions::Zero<Dim>>("");
  CHECK(created_solution == solution);
  test_serialization(solution);
  test_copy_semantics(solution);

  using system = Elasticity::FirstOrderSystem<Dim>;
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{1., 1.};
  const Mesh<Dim> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto coord_map = make_coord_map<Dim>();
  FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
      solution, mesh, coord_map, 1.e-14,
      std::make_tuple(constitutive_relation,
                      coord_map(logical_coordinates(mesh))));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Elasticity.Zero",
                  "[PointwiseFunctions][Unit]") {
  test_solution<2>();
  test_solution<3>();
}
