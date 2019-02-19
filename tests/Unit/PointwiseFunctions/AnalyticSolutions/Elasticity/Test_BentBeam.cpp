// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

struct BentBeamProxy : Elasticity::Solutions::BentBeam {
  using Elasticity::Solutions::BentBeam::BentBeam;

  using field_tags = tmpl::list<Elasticity::Tags::Displacement<2>,
                                Elasticity::Tags::Stress<2>>;
  using source_tags =
      tmpl::list<Tags::Source<Elasticity::Tags::Displacement<2>>>;

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
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>{79.3651,
                                                                 38.7597}};
  const Elasticity::Solutions::BentBeam created_solution =
      test_creation<Elasticity::Solutions::BentBeam>(
          "  Length: 5.\n"
          "  Height: 1.\n"
          "  BendingMoment: 0.5\n"
          "  Material:\n"
          "    BulkModulus: 79.3651\n"
          "    ShearModulus: 38.7597\n");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Elasticity"};
  const BentBeamProxy solution{
      5., 1., 0.5,
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>{79.3651,
                                                                 38.7597}};
  pypp::check_with_random_values<1,
                                 tmpl::list<Elasticity::Tags::Displacement<2>,
                                            Elasticity::Tags::Stress<2>>>(
      &BentBeamProxy::field_variables, solution, "BentBeam",
      {"displacement", "stress"}, {{{-5., 5.}}},
      std::make_tuple(5., 1., 0.5, 79.3651, 38.7597), DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::Source<Elasticity::Tags::Displacement<2>>>>(
      &BentBeamProxy::source_variables, solution, "BentBeam", {"source"},
      {{{-5., 5.}}}, std::make_tuple(), DataVector(5));
}
