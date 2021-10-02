// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Tag : db::SimpleTag {
  using type = int;
};
struct TestOptionsGroup {
  static std::string name() { return "TestLinearSolver"; }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.LinearSolver.Tags",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Operand<Tag>>(
      "LinearOperand(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::OperatorAppliedTo<Tag>>(
      "LinearOperatorAppliedTo(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Residual<Tag>>(
      "LinearResidual(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::MagnitudeSquare<Tag>>(
      "LinearMagnitudeSquare(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Magnitude<Tag>>(
      "LinearMagnitude(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Orthogonalization<Tag>>(
      "LinearOrthogonalization(Tag)");
  TestHelpers::db::test_prefix_tag<
      LinearSolver::Tags::OrthogonalizationHistory<Tag>>(
      "LinearOrthogonalizationHistory(Tag)");
  TestHelpers::db::test_prefix_tag<
      LinearSolver::Tags::KrylovSubspaceBasis<Tag>>("KrylovSubspaceBasis(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Preconditioned<Tag>>(
      "Preconditioned(Tag)");

  {
    INFO("ResidualCompute");
    TestHelpers::db::test_compute_tag<
        LinearSolver::Tags::ResidualCompute<Tag, ::Tags::Source<Tag>>>(
        "LinearResidual(Tag)");
    const auto box = db::create<
        db::AddSimpleTags<::Tags::Source<Tag>,
                          LinearSolver::Tags::OperatorAppliedTo<Tag>>,
        db::AddComputeTags<
            LinearSolver::Tags::ResidualCompute<Tag, ::Tags::Source<Tag>>>>(3,
                                                                            2);
    CHECK(db::get<LinearSolver::Tags::Residual<Tag>>(box) == 1);
  }
}
