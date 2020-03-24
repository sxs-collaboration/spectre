// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

struct BentBeamProxy : Elasticity::Solutions::BentBeam {
  using Elasticity::Solutions::BentBeam::BentBeam;

  using field_tags = tmpl::list<Elasticity::Tags::Displacement<2>,
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
  const Elasticity::Solutions::BentBeam created_solution =
      TestHelpers::test_creation<Elasticity::Solutions::BentBeam>(
          "Length: 5.\n"
          "Height: 1.\n"
          "BendingMoment: 0.5\n"
          "Material:\n"
          "  BulkModulus: 79.36507936507935\n"
          "  ShearModulus: 38.75968992248062\n");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Elasticity"};
  const BentBeamProxy solution{
      5., 1., 0.5,
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>{
          79.36507936507935, 38.75968992248062}};

  tnsr::I<DataVector, 2> x{{{{1., 2.}, {2., 1.}}}};
  const auto solution_vars = variables_from_tagged_tuple(
      solution.variables(x, tmpl::list<Elasticity::Tags::Displacement<2>,
                                       Elasticity::Tags::Stress<2>>{}));
  Variables<tmpl::list<Elasticity::Tags::Displacement<2>,
                       Elasticity::Tags::Stress<2>>>
      expected_vars{2};
  auto& expected_displacement =
      get<Elasticity::Tags::Displacement<2>>(expected_vars);
  get<0>(expected_displacement) = DataVector{-0.12, -0.12};
  get<1>(expected_displacement) = DataVector{-0.1227, -0.0588};
  auto& expected_stress = get<Elasticity::Tags::Stress<2>>(expected_vars);
  get<0, 0>(expected_stress) = DataVector{12., 6.};
  get<1, 0>(expected_stress) = DataVector{0., 0.};
  get<1, 1>(expected_stress) = DataVector{0., 0.};
  CHECK_VARIABLES_APPROX(solution_vars, expected_vars);

  pypp::check_with_random_values<1,
                                 tmpl::list<Elasticity::Tags::Displacement<2>,
                                            Elasticity::Tags::Stress<2>>>(
      &BentBeamProxy::field_variables, solution, "BentBeam",
      {"displacement", "stress"}, {{{-5., 5.}}},
      std::make_tuple(5., 1., 0.5, 79.36507936507935, 38.75968992248062),
      DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::FixedSource<Elasticity::Tags::Displacement<2>>>>(
      &BentBeamProxy::source_variables, solution, "BentBeam", {"source"},
      {{{-5., 5.}}}, std::make_tuple(), DataVector(5));
}
