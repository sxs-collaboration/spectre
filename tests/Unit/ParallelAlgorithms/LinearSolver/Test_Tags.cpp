// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Tag : db::SimpleTag {
  using type = int;
};
struct TestOptionsGroup {
  static std::string name() noexcept { return "TestLinearSolver"; }
};
using magnitude_square_tag = LinearSolver::Tags::MagnitudeSquare<Tag>;
using magnitude_tag = LinearSolver::Tags::Magnitude<Tag>;
using residual_magnitude_tag =
    LinearSolver::Tags::Magnitude<LinearSolver::Tags::Residual<Tag>>;
using initial_residual_magnitude_tag =
    LinearSolver::Tags::Initial<residual_magnitude_tag>;
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.LinearSolver.Tags",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Operand<Tag>>(
      "LinearOperand(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::OperatorAppliedTo<Tag>>(
      "LinearOperatorAppliedTo(Tag)");
  TestHelpers::db::test_simple_tag<
      LinearSolver::Tags::IterationId<TestOptionsGroup>>(
      "IterationId(TestLinearSolver)");
  TestHelpers::db::test_simple_tag<
      LinearSolver::Tags::HasConverged<TestOptionsGroup>>(
      "HasConverged(TestLinearSolver)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Residual<Tag>>(
      "LinearResidual(Tag)");
  TestHelpers::db::test_prefix_tag<LinearSolver::Tags::Initial<Tag>>(
      "Initial(Tag)");
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
  TestHelpers::db::test_simple_tag<
      LinearSolver::Tags::ConvergenceCriteria<TestOptionsGroup>>(
      "ConvergenceCriteria(TestLinearSolver)");
  TestHelpers::db::test_simple_tag<
      LinearSolver::Tags::Iterations<TestOptionsGroup>>(
      "Iterations(TestLinearSolver)");
  TestHelpers::db::test_simple_tag<
      LinearSolver::Tags::Verbosity<TestOptionsGroup>>(
      "Verbosity(TestLinearSolver)");

  {
    INFO("HasConvergedCompute");
    TestHelpers::db::test_compute_tag<
        LinearSolver::Tags::HasConvergedCompute<Tag, TestOptionsGroup>>(
        "HasConverged(TestLinearSolver)");
    const auto box = db::create<
        db::AddSimpleTags<
            LinearSolver::Tags::ConvergenceCriteria<TestOptionsGroup>,
            LinearSolver::Tags::IterationId<TestOptionsGroup>,
            residual_magnitude_tag, initial_residual_magnitude_tag>,
        db::AddComputeTags<
            LinearSolver::Tags::HasConvergedCompute<Tag, TestOptionsGroup>>>(
        Convergence::Criteria{2, 0., 0.5}, 2_st, 1., 1.);
    CHECK(db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box)
              .reason() == Convergence::Reason::MaxIterations);
  }

  {
    INFO("HasConvergedCompute - Zero initial residual");
    // A vanishing initial residual should work because the absolute residual
    // condition takes precedence so that no FPE occurs
    const auto box = db::create<
        db::AddSimpleTags<
            LinearSolver::Tags::ConvergenceCriteria<TestOptionsGroup>,
            LinearSolver::Tags::IterationId<TestOptionsGroup>,
            residual_magnitude_tag, initial_residual_magnitude_tag>,
        db::AddComputeTags<
            LinearSolver::Tags::HasConvergedCompute<Tag, TestOptionsGroup>>>(
        Convergence::Criteria{2, 0., 0.5}, 1_st, 0., 0.);
    CHECK(db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box)
              .reason() == Convergence::Reason::AbsoluteResidual);
  }

  {
    INFO("HasConvergedByIterationsCompute - not converged");
    TestHelpers::db::test_compute_tag<
        LinearSolver::Tags::HasConvergedByIterationsCompute<TestOptionsGroup>>(
        "HasConverged(TestLinearSolver)");
    const auto box = db::create<
        db::AddSimpleTags<LinearSolver::Tags::Iterations<TestOptionsGroup>,
                          LinearSolver::Tags::IterationId<TestOptionsGroup>>,
        db::AddComputeTags<LinearSolver::Tags::HasConvergedByIterationsCompute<
            TestOptionsGroup>>>(2_st, 1_st);
    CHECK_FALSE(
        db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box));
  }

  {
    INFO("HasConvergedByIterationsCompute - has converged");
    const auto box = db::create<
        db::AddSimpleTags<LinearSolver::Tags::Iterations<TestOptionsGroup>,
                          LinearSolver::Tags::IterationId<TestOptionsGroup>>,
        db::AddComputeTags<LinearSolver::Tags::HasConvergedByIterationsCompute<
            TestOptionsGroup>>>(2_st, 2_st);
    CHECK(db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged<TestOptionsGroup>>(box)
              .reason() == Convergence::Reason::MaxIterations);
  }

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

  {
    INFO("MagnitudeCompute");
    TestHelpers::db::test_compute_tag<
        LinearSolver::Tags::MagnitudeCompute<magnitude_square_tag>>(
        "LinearMagnitude(Tag)");
    const auto box =
        db::create<db::AddSimpleTags<magnitude_square_tag>,
                   db::AddComputeTags<LinearSolver::Tags::MagnitudeCompute<
                       magnitude_square_tag>>>(4.);
    CHECK(db::get<magnitude_tag>(box) == approx(2.));
  }

  TestHelpers::db::test_compute_tag<
      Tags::NextCompute<LinearSolver::Tags::IterationId<TestOptionsGroup>>>(
      "Next(IterationId(TestLinearSolver))");
}
