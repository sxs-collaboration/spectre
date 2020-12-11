// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Tag : db::SimpleTag {
  using type = int;
};
struct TestOptionsGroup {
  static std::string name() noexcept { return "TestNonlinearSolver"; }
};
}  // namespace

namespace NonlinearSolver {

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.NonlinearSolver.Tags",
                  "[Unit][ParallelAlgorithms]") {
  TestHelpers::db::test_prefix_tag<Tags::Correction<Tag>>("Correction(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::OperatorAppliedTo<Tag>>(
      "NonlinearOperatorAppliedTo(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::Residual<Tag>>(
      "NonlinearResidual(Tag)");
  TestHelpers::db::test_simple_tag<Tags::StepLength<TestOptionsGroup>>(
      "StepLength(TestNonlinearSolver)");
  TestHelpers::db::test_simple_tag<Tags::SufficientDecrease<TestOptionsGroup>>(
      "SufficientDecrease(TestNonlinearSolver)");
  TestHelpers::db::test_simple_tag<Tags::DampingFactor<TestOptionsGroup>>(
      "DampingFactor(TestNonlinearSolver)");
  TestHelpers::db::test_simple_tag<
      Tags::MaxGlobalizationSteps<TestOptionsGroup>>(
      "MaxGlobalizationSteps(TestNonlinearSolver)");
  TestHelpers::db::test_prefix_tag<Tags::Globalization<Tag>>(
      "Globalization(Tag)");
  {
    INFO("ResidualCompute");
    TestHelpers::db::test_compute_tag<
        Tags::ResidualCompute<Tag, ::Tags::Source<Tag>>>(
        "NonlinearResidual(Tag)");
    const auto box = db::create<
        db::AddSimpleTags<::Tags::Source<Tag>, Tags::OperatorAppliedTo<Tag>>,
        db::AddComputeTags<Tags::ResidualCompute<Tag, ::Tags::Source<Tag>>>>(3,
                                                                             2);
    CHECK(db::get<Tags::Residual<Tag>>(box) == 1);
  }
}

}  // namespace NonlinearSolver
