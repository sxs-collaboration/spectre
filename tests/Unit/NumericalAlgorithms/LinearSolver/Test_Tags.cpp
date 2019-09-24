// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Tag : db::SimpleTag {
  static std::string name() noexcept { return "Tag"; }
  using type = int;
};
using magnitude_square_tag = LinearSolver::Tags::MagnitudeSquare<Tag>;
using magnitude_tag = LinearSolver::Tags::Magnitude<Tag>;
using residual_magnitude_tag =
    LinearSolver::Tags::Magnitude<LinearSolver::Tags::Residual<Tag>>;
using initial_residual_magnitude_tag =
    LinearSolver::Tags::Initial<residual_magnitude_tag>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Tags",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  CHECK(LinearSolver::Tags::Operand<Tag>::name() == "LinearOperand(Tag)");
  CHECK(LinearSolver::Tags::OperatorAppliedTo<Tag>::name() ==
        "LinearOperatorAppliedTo(Tag)");
  CHECK(LinearSolver::Tags::HasConverged::name() == "LinearSolverHasConverged");
  CHECK(LinearSolver::Tags::Residual<Tag>::name() == "LinearResidual(Tag)");
  CHECK(LinearSolver::Tags::Initial<Tag>::name() == "Initial(Tag)");
  CHECK(LinearSolver::Tags::MagnitudeSquare<Tag>::name() ==
        "LinearMagnitudeSquare(Tag)");
  CHECK(LinearSolver::Tags::Magnitude<Tag>::name() == "LinearMagnitude(Tag)");
  CHECK(LinearSolver::Tags::Orthogonalization<Tag>::name() ==
        "LinearOrthogonalization(Tag)");
  CHECK(LinearSolver::Tags::OrthogonalizationHistory<Tag>::name() ==
        "LinearOrthogonalizationHistory(Tag)");
  CHECK(LinearSolver::Tags::KrylovSubspaceBasis<Tag>::name() ==
        "KrylovSubspaceBasis(Tag)");
  CHECK(LinearSolver::Tags::ConvergenceCriteria::name() ==
        "ConvergenceCriteria");

  {
    INFO("HasConvergedCompute");
    CHECK(LinearSolver::Tags::HasConvergedCompute<Tag>::name() ==
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
    CHECK(LinearSolver::Tags::MagnitudeCompute<magnitude_square_tag>::name() ==
          "LinearMagnitude(Tag)");
    const auto box =
        db::create<db::AddSimpleTags<magnitude_square_tag>,
                   db::AddComputeTags<LinearSolver::Tags::MagnitudeCompute<
                       magnitude_square_tag>>>(4.);
    CHECK(db::get<magnitude_tag>(box) == approx(2.));
  }
}
