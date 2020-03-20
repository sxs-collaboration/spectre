// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
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
  TestHelpers::db::test_simple_tag<LinearSolver::Tags::IterationId>(
      "LinearIterationId");
  TestHelpers::db::test_simple_tag<LinearSolver::Tags::HasConverged>(
      "LinearSolverHasConverged");
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
  TestHelpers::db::test_simple_tag<LinearSolver::Tags::ConvergenceCriteria>(
      "ConvergenceCriteria");
  TestHelpers::db::test_simple_tag<LinearSolver::Tags::Verbosity>("Verbosity");

  {
    INFO("HasConvergedCompute");
    TestHelpers::db::test_compute_tag<
        LinearSolver::Tags::HasConvergedCompute<Tag>>(
        "LinearSolverHasConverged");
    const auto box = db::create<
        db::AddSimpleTags<LinearSolver::Tags::ConvergenceCriteria,
                          LinearSolver::Tags::IterationId,
                          residual_magnitude_tag,
                          initial_residual_magnitude_tag>,
        db::AddComputeTags<LinearSolver::Tags::HasConvergedCompute<Tag>>>(
        Convergence::Criteria{2, 0., 0.5}, 2_st, 1., 1.);
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box).reason() ==
          Convergence::Reason::MaxIterations);
  }

  {
    INFO("HasConvergedCompute - Zero initial residual");
    // A vanishing initial residual should work because the absolute residual
    // condition takes precedence so that no FPE occurs
    const auto box = db::create<
        db::AddSimpleTags<LinearSolver::Tags::ConvergenceCriteria,
                          LinearSolver::Tags::IterationId,
                          residual_magnitude_tag,
                          initial_residual_magnitude_tag>,
        db::AddComputeTags<LinearSolver::Tags::HasConvergedCompute<Tag>>>(
        Convergence::Criteria{2, 0., 0.5}, 1_st, 0., 0.);
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box).reason() ==
          Convergence::Reason::AbsoluteResidual);
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
      Tags::NextCompute<LinearSolver::Tags::IterationId>>(
      "Next(LinearIterationId)");
}
