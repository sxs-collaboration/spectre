// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

struct HalfSpaceMirrorProxy : Elasticity::Solutions::HalfSpaceMirror {
  using Elasticity::Solutions::HalfSpaceMirror::HalfSpaceMirror;

  using field_tags = tmpl::list<Elasticity::Tags::Displacement<3>,
                                Elasticity::Tags::Strain<3>>;
  using source_tags =
      tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<3>>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, 3>& x) const noexcept {
    return Elasticity::Solutions::HalfSpaceMirror::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, 3>& x) const noexcept {
    return Elasticity::Solutions::HalfSpaceMirror::variables(x, source_tags{});
  }
};

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Elasticity.HalfSpaceMirror",
    "[PointwiseFunctions][Unit][Elasticity]") {
  const Elasticity::Solutions::HalfSpaceMirror check_solution{
      0.177, 1.0,
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>{
          // Iron: E=72, nu=0.17
          36.36363636363637, 30.76923076923077},
      50, 1.e-13};
  const Elasticity::Solutions::HalfSpaceMirror created_solution =
      TestHelpers::test_creation<Elasticity::Solutions::HalfSpaceMirror>(
          "BeamWidth: 0.177\n"
          "AppliedForce: 1.0\n"
          "Material:\n"
          "  BulkModulus: 36.36363636363637\n"
          "  ShearModulus: 30.76923076923077\n"
          "IntegrationIntervals: 20\n"
          "IntergrationTolerance: 1.e-13\n");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Elasticity"};
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>
      constitutive_relation{36.36363636363637, 30.76923076923077};
  const HalfSpaceMirrorProxy solution{0.177, 1.0, constitutive_relation, 50,
                                      1.e-13};
  pypp::check_with_random_values<1,
                                 tmpl::list<Elasticity::Tags::Displacement<3>,
                                            Elasticity::Tags::Strain<3>>>(
      &HalfSpaceMirrorProxy::field_variables, solution, "HalfSpaceMirror",
      {"displacement", "strain"}, {{{0., 5.}}},
      std::make_tuple(0.177, 1.0, 36.36363636363637, 30.76923076923077),
      DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<3>>>>(
      &HalfSpaceMirrorProxy::source_variables, solution, "HalfSpaceMirror",
      {"source"}, {{{0., 5.}}}, std::make_tuple(), DataVector(5));

  {
    INFO("Test elasticity system with half-space mirror");
    // Verify that the solution numerically solves the system and that the
    // discretization error decreases exponentially with polynomial order
    using system = Elasticity::FirstOrderSystem<dim>;
    const typename system::fluxes fluxes_computer{};
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{{{-1., 1., 0., 0.5}, {-1., 1., 0., 0.5}, {-1., 1., 0., 0.5}}};
    const Mesh<dim> mesh{10, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e4, 1.,
        std::make_tuple(constitutive_relation, inertial_coords));
  };
}
