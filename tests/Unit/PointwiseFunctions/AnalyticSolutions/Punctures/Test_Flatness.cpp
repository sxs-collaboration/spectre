// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Punctures/Flatness.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"

namespace Punctures::Solutions {

SPECTRE_TEST_CASE("Unit.AnalyticSolutions.Punctures.Flatness",
                  "[PointwiseFunctions][Unit]") {
  const auto created = TestHelpers::test_factory_creation<
      elliptic::analytic_data::AnalyticSolution, Flatness>("Flatness");
  REQUIRE(dynamic_cast<const Flatness*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Flatness&>(*created);
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  using test_tags = tmpl::list<
      Tags::Field, ::Tags::deriv<Tags::Field, tmpl::size_t<3>, Frame::Inertial>,
      Punctures::Tags::Alpha,
      Punctures::Tags::TracelessConformalExtrinsicCurvature,
      Punctures::Tags::Beta, ::Tags::FixedSource<Punctures::Tags::Field>>;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  DataVector used_for_size{3};
  const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto vars =
      variables_from_tagged_tuple(solution.variables(x, test_tags{}));
  CHECK_VARIABLES_APPROX(vars, (Variables<test_tags>{3, 0.}));
}

}  // namespace Punctures::Solutions
