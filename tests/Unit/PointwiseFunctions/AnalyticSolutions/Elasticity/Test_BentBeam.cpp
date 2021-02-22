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
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

struct BentBeamProxy : Elasticity::Solutions::BentBeam<> {
  using Elasticity::Solutions::BentBeam<>::BentBeam;

  using field_tags =
      tmpl::list<Elasticity::Tags::Displacement<2>, Elasticity::Tags::Strain<2>,
                 Elasticity::Tags::MinusStress<2>,
                 Elasticity::Tags::PotentialEnergyDensity<2>>;
  using source_tags =
      tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<2>>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, 2>& x) const noexcept {
    return Elasticity::Solutions::BentBeam<>::variables(x, field_tags{});
  }

  // check_with_random_values() does not allow for arguments to be static
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, 2>& x) const noexcept {
    return Elasticity::Solutions::BentBeam<>::variables(x, source_tags{});
  }
};

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Elasticity.BentBeam",
    "[PointwiseFunctions][Unit][Elasticity]") {
  const Elasticity::Solutions::BentBeam<> check_solution{
      5., 1., 0.5,
      // Iron: E=100, nu=0.29
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>{
          79.36507936507935, 38.75968992248062}};
  const auto created_solution =
      TestHelpers::test_creation<Elasticity::Solutions::BentBeam<>>(
          "Length: 5.\n"
          "Height: 1.\n"
          "BendingMoment: 0.5\n"
          "Material:\n"
          "  BulkModulus: 79.36507936507935\n"
          "  ShearModulus: 38.75968992248062\n");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);
  test_copy_semantics(check_solution);

  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions"};
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>
      constitutive_relation{79.36507936507935, 38.75968992248062};
  const BentBeamProxy solution{5., 1., 0.5, constitutive_relation};

  tnsr::I<DataVector, 2> x{{{{1., 2.}, {2., 1.}}}};
  const auto solution_vars =
      variables_from_tagged_tuple(solution.field_variables(x));
  Variables<typename BentBeamProxy::field_tags> expected_vars{2};
  auto& expected_displacement =
      get<Elasticity::Tags::Displacement<2>>(expected_vars);
  get<0>(expected_displacement) = DataVector{-0.12, -0.12};
  get<1>(expected_displacement) = DataVector{-0.1227, -0.0588};
  auto& expected_strain = get<Elasticity::Tags::Strain<2>>(expected_vars);
  get<0, 0>(expected_strain) = DataVector{-0.12, -0.06};
  get<1, 0>(expected_strain) = DataVector{0., 0.};
  get<1, 1>(expected_strain) = DataVector{0.0348, 0.0174};
  auto& expected_stress = get<Elasticity::Tags::MinusStress<2>>(expected_vars);
  get<0, 0>(expected_stress) = DataVector{-12., -6.};
  get<1, 0>(expected_stress) = DataVector{0., 0.};
  get<1, 1>(expected_stress) = DataVector{0., 0.};
  auto& expected_potential_energy_density =
      get<Elasticity::Tags::PotentialEnergyDensity<2>>(expected_vars);
  get(expected_potential_energy_density) = DataVector{0.72, 0.18};
  CHECK_VARIABLES_APPROX(solution_vars, expected_vars);
  CHECK(solution.potential_energy() == approx(0.075));

  pypp::check_with_random_values<1>(
      &BentBeamProxy::field_variables, solution,
      "AnalyticSolutions.Elasticity.BentBeam",
      {"displacement", "strain", "minus_stress", "potential_energy_density"},
      {{{-5., 5.}}},
      std::make_tuple(5., 1., 0.5, 79.36507936507935, 38.75968992248062),
      DataVector(5));
  pypp::check_with_random_values<1>(&BentBeamProxy::source_variables, solution,
                                    "AnalyticSolutions.Elasticity.BentBeam",
                                    {"source"}, {{{-5., 5.}}},
                                    std::make_tuple(), DataVector(5));

  using AffineMap = domain::CoordinateMaps::Affine;
  using AffineMap2D =
      domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
  // Since potential_energy() in BentBeam returns the energy stored in the
  // entire beam, the coord_map has to match its dimensions.
  const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>
      coord_map{{{-1., 1., -5. / 2., 5. / 2.}, {-1., 1., -1. / 2., 1. / 2.}}};
  // Since the solution is a polynomial of degree 2, it should numerically
  // solve the system equations to machine precision on 3 grid points per
  // dimension.
  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = coord_map(logical_coords);
  {
    INFO("Test elasticity system with bent beam");
    using system = Elasticity::FirstOrderSystem<2>;
    // Verify that the solution numerically solves the system
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, mesh, coord_map,
        std::numeric_limits<double>::epsilon() * 100.,
        std::make_tuple(constitutive_relation, inertial_coords));
  };

  {
    INFO("Test elastic potential energy of bent beam");
    // Verify that the energy density integrated over the whole beam is correct
    const auto jacobian = coord_map.jacobian(logical_coords);
    const auto strain = get<Elasticity::Tags::Strain<2>>(solution.variables(
        inertial_coords, tmpl::list<Elasticity::Tags::Strain<2>>{}));
    const auto pointwise_energy = Elasticity::potential_energy_density(
        strain, inertial_coords, constitutive_relation);
    const auto det_jacobian = get(determinant(jacobian));
    double potential_energy =
        definite_integral(det_jacobian * get(pointwise_energy), mesh);
    CHECK(potential_energy == approx(solution.potential_energy()));
  };
}
