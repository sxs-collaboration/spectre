// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
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
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct HalfSpaceMirrorProxy : Elasticity::Solutions::HalfSpaceMirror<> {
  using Elasticity::Solutions::HalfSpaceMirror<>::HalfSpaceMirror;

  using field_tags = tmpl::list<Elasticity::Tags::Displacement<3>,
                                Elasticity::Tags::Strain<3>>;
  using source_tags =
      tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<3>>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, 3>& x) const noexcept {
    return Elasticity::Solutions::HalfSpaceMirror<>::variables(x, field_tags{});
  }

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, 3>& x) const noexcept {
    return Elasticity::Solutions::HalfSpaceMirror<>::variables(x,
                                                               source_tags{});
  }
};

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Elasticity.HalfSpaceMirror",
    "[PointwiseFunctions][Unit][Elasticity]") {
  // Fused silica: E = 72, nu = 0.17
  // => K = E / (3 * (1 - 2 nu)), mu = E / (2 * (1 + nu))
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>
      constitutive_relation{36.36363636363637, 30.76923076923077};
  const Elasticity::Solutions::HalfSpaceMirror<> check_solution{
      0.177, constitutive_relation, 350, 1.e-12, 1.e-10};
  const auto created_solution =
      TestHelpers::test_creation<Elasticity::Solutions::HalfSpaceMirror<>>(
          "BeamWidth: 0.177\n"
          "Material:\n"
          "  BulkModulus: 36.36363636363637\n"
          "  ShearModulus: 30.76923076923077\n"
          "IntegrationIntervals: 350\n"
          "AbsoluteTolerance: 1e-12\n"
          "RelativeTolerance: 1e-10\n");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);
  test_copy_semantics(check_solution);

  const HalfSpaceMirrorProxy solution{0.177, constitutive_relation, 350, 1e-11,
                                      1e-11};
  {
    INFO("Random-value tests");
    pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions"};
    // The accuracy of the random-value tests is limited by the numerical
    // integrations in both the HalfSpaceMirror solution and the Python
    // implementation
    const double eps = 1.e-9;
    pypp::check_with_random_values<1>(
        &HalfSpaceMirrorProxy::field_variables, solution,
        "AnalyticSolutions.Elasticity.HalfSpaceMirror",
        {"displacement", "strain"}, {{{0., 3.}}},
        std::make_tuple(0.177, 36.36363636363637, 30.76923076923077),
        DataVector(5), eps);
    pypp::check_with_random_values<1>(
        &HalfSpaceMirrorProxy::source_variables, solution,
        "AnalyticSolutions.Elasticity.HalfSpaceMirror", {"source"},
        {{{0., 3.}}}, std::make_tuple(), DataVector(5), eps);
  }
  {
    INFO(
        "Test that functions behave expectedly at the origin and vanish far "
        "away from the source");
    const tnsr::I<DataVector, 3> x{
        {{{0., 20., 0.}, {0., 0., 0.}, {0., 0., 20.}}}};
    const auto solution_vars = variables_from_tagged_tuple(
        solution.variables(x, tmpl::list<Elasticity::Tags::Displacement<3>,
                                         Elasticity::Tags::Strain<3>>{}));
    Variables<tmpl::list<Elasticity::Tags::Displacement<3>,
                         Elasticity::Tags::Strain<3>>>
        expected_vars{3};
    auto& expected_displacement =
        get<Elasticity::Tags::Displacement<3>>(expected_vars);
    get<0>(expected_displacement) = DataVector{0., 0., 0.};
    get<1>(expected_displacement) = DataVector{0., 0., 0.};
    get<2>(expected_displacement) = DataVector{4.30e-02, 0., 0.};
    auto& expected_strain = get<Elasticity::Tags::Strain<3>>(expected_vars);
    get<0, 0>(expected_strain) = DataVector{-5.45e-02, 0., 0.};
    get<1, 0>(expected_strain) = DataVector{0., 0., 0.};
    get<2, 0>(expected_strain) = DataVector{0., 0., 0.};
    get<1, 1>(expected_strain) = DataVector{-5.45e-02, 0., 0.};
    get<2, 1>(expected_strain) = DataVector{0., 0., 0.};
    get<2, 2>(expected_strain) = DataVector{-10.90e-02, 0., 0.};
    Approx custom_approx = Approx::custom().margin(5e-4);
    CHECK_VARIABLES_CUSTOM_APPROX(solution_vars, expected_vars, custom_approx);
  };

  {
    INFO("Test elasticity system with half-space mirror");
    // Verify that the solution numerically solves the system
    using system = Elasticity::FirstOrderSystem<3>;
    const typename system::fluxes fluxes_computer{};
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{{{-1., 1., 0., 0.5}, {-1., 1., 0., 0.5}, {-1., 1., 0., 0.5}}};
    const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, fluxes_computer, mesh, coord_map, 0.05,
        std::make_tuple(constitutive_relation, inertial_coords));
  };
}

// [[OutputRegex, The numerical integral failed]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticSolutions.Elasticity.HalfSpaceMirror.ConvergenceError",
    "[PointwiseFunctions][Unit][Elasticity]") {
  ERROR_TEST();
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>
      constitutive_relation{36.36363636363637, 30.76923076923077};
  const HalfSpaceMirrorProxy solution{0.177, constitutive_relation, 5, 1e-15,
                                      1e-15};
  const tnsr::I<DataVector, 3> x{{{{0.3}, {1.3}, {2.3}}}};
  const auto solution_vars = variables_from_tagged_tuple(
      solution.variables(x, tmpl::list<Elasticity::Tags::Displacement<3>,
                                       Elasticity::Tags::Strain<3>>{}));
}
