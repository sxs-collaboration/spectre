// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
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
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

struct BentBeamProxy : Elasticity::Solutions::BentBeam {
  using Elasticity::Solutions::BentBeam::BentBeam;

  using field_tags =
      tmpl::list<Elasticity::Tags::Displacement<2>, Elasticity::Tags::Strain<2>,
                 Elasticity::Tags::Stress<2>>;
  using source_tags =
      tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<2>>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, 2>& x) const noexcept {
    return Elasticity::Solutions::BentBeam::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, 2>& x) const noexcept {
    return Elasticity::Solutions::BentBeam::variables(x, source_tags{});
  }

  double potential_energy() const noexcept {
    return Elasticity::Solutions::BentBeam::potential_energy();
  }
};

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Elasticity.BentBeam",
    "[PointwiseFunctions][Unit][Elasticity]") {
  const Elasticity::Solutions::BentBeam check_solution{
      5., 1., 0.5,
      // Iron: E=100, nu=0.29
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>{
          79.36507936507935, 38.75968992248062}};
  const auto created_solution =
      TestHelpers::test_creation<Elasticity::Solutions::BentBeam>(
          "Length: 5.\n"
          "Height: 1.\n"
          "BendingMoment: 0.5\n"
          "Material:\n"
          "  BulkModulus: 79.36507936507935\n"
          "  ShearModulus: 38.75968992248062\n");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);

  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions"};
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>
      constitutive_relation{79.36507936507935, 38.75968992248062};
  const BentBeamProxy solution{5., 1., 0.5, constitutive_relation};

  tnsr::I<DataVector, 2> x{{{{1., 2.}, {2., 1.}}}};
  const auto solution_vars = variables_from_tagged_tuple(solution.variables(
      x,
      tmpl::list<Elasticity::Tags::Displacement<2>, Elasticity::Tags::Strain<2>,
                 Elasticity::Tags::Stress<2>>{}));
  Variables<
      tmpl::list<Elasticity::Tags::Displacement<2>, Elasticity::Tags::Strain<2>,
                 Elasticity::Tags::Stress<2>>>
      expected_vars{2};
  auto& expected_displacement =
      get<Elasticity::Tags::Displacement<2>>(expected_vars);
  get<0>(expected_displacement) = DataVector{-0.12, -0.12};
  get<1>(expected_displacement) = DataVector{-0.1227, -0.0588};
  auto& expected_strain = get<Elasticity::Tags::Strain<2>>(expected_vars);
  get<0, 0>(expected_strain) = DataVector{-0.12, -0.06};
  get<1, 0>(expected_strain) = DataVector{0., 0.};
  get<1, 1>(expected_strain) = DataVector{0.0348, 0.0174};
  auto& expected_stress = get<Elasticity::Tags::Stress<2>>(expected_vars);
  get<0, 0>(expected_stress) = DataVector{12., 6.};
  get<1, 0>(expected_stress) = DataVector{0., 0.};
  get<1, 1>(expected_stress) = DataVector{0., 0.};
  CHECK_VARIABLES_APPROX(solution_vars, expected_vars);
  CHECK_ITERABLE_APPROX(solution.potential_energy(), 0.075);

  pypp::check_with_random_values<
      1, tmpl::list<Elasticity::Tags::Displacement<2>,
                    Elasticity::Tags::Strain<2>, Elasticity::Tags::Stress<2>>>(
      &BentBeamProxy::field_variables, solution,
      "AnalyticSolutions.Elasticity.BentBeam",
      {"displacement", "strain", "stress"}, {{{-5., 5.}}},
      std::make_tuple(5., 1., 0.5, 79.36507936507935, 38.75968992248062),
      DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<2>>>>(
      &BentBeamProxy::source_variables, solution,
      "AnalyticSolutions.Elasticity.BentBeam", {"source"}, {{{-5., 5.}}},
      std::make_tuple(), DataVector(5));

  {
    INFO("Test elasticity system with bent beam");
    // Verify that the solution numerically solves the system
    using system = Elasticity::FirstOrderSystem<2>;
    const typename system::fluxes fluxes_computer{};
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>
        coord_map{{{-1., 1., -2.5, 2.5}, {-1., 1., -0.5, 0.5}}};
    // Since the solution is a polynomial of degree 2 it should numerically
    // solve the system equations to machine precision on 3 grid points per
    // dimension.
    const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, fluxes_computer, mesh, coord_map,
        std::numeric_limits<double>::epsilon() * 100.,
        std::make_tuple(constitutive_relation, inertial_coords));
  };
}
