// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/none.hpp>
#include <boost/none_t.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/LinearSolver/Convergence.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Convergence",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  const LinearSolver::ConvergenceCriteria criteria{2, 0., 0.5};

  {
    INFO("ConvergenceCriteria");
    CHECK(criteria == LinearSolver::ConvergenceCriteria{2, 0., 0.5});
    CHECK(criteria != LinearSolver::ConvergenceCriteria{3, 0., 0.5});
    CHECK(criteria != LinearSolver::ConvergenceCriteria{2, 1., 0.5});
    CHECK(criteria != LinearSolver::ConvergenceCriteria{2, 0., 0.6});
    test_serialization(criteria);
    test_copy_semantics(criteria);
    const auto created_criteria =
        test_creation<LinearSolver::ConvergenceCriteria>(
            "  MaxIterations: 2\n"
            "  AbsoluteResidual: 0.\n"
            "  RelativeResidual: 0.5\n");
    CHECK(created_criteria == criteria);
  }

  {
    INFO("Convergence logic");
    CHECK(LinearSolver::convergence_criteria_match(criteria, 1, 1., 1.) ==
          boost::none);
    CHECK(LinearSolver::convergence_criteria_match(criteria, 2, 1., 1.) ==
          LinearSolver::ConvergenceReason::MaxIterations);
    CHECK(LinearSolver::convergence_criteria_match(criteria, 1, 0., 1.) ==
          LinearSolver::ConvergenceReason::AbsoluteResidual);
    CHECK(LinearSolver::convergence_criteria_match(criteria, 1, 1., 2.) ==
          LinearSolver::ConvergenceReason::RelativeResidual);
  }

  {
    INFO("HasNotConverged");
    const LinearSolver::HasConverged has_not_converged_by_default{};
    CHECK_FALSE(has_not_converged_by_default);
    test_serialization(has_not_converged_by_default);
    test_copy_semantics(has_not_converged_by_default);
    const LinearSolver::HasConverged has_not_converged{criteria, 1, 1., 1.};
    CHECK_FALSE(has_not_converged);
    CHECK(get_output(has_not_converged) ==
          "The linear solver has not yet converged.\n");
    test_serialization(has_not_converged);
    test_copy_semantics(has_not_converged);
  }

  {
    INFO("HasConverged - MaxIterations")
    const LinearSolver::HasConverged has_converged{criteria, 2, 1., 1.};
    CHECK(has_converged);
    CHECK(has_converged.reason() ==
          LinearSolver::ConvergenceReason::MaxIterations);
    CHECK(get_output(has_converged) ==
          "The linear solver has reached its maximum number of iterations "
          "(2).\n");
    test_serialization(has_converged);
    test_copy_semantics(has_converged);
  }

  {
    INFO("HasConverged - AbsoluteResidual")
    const LinearSolver::HasConverged has_converged{criteria, 1, 0., 1.};
    CHECK(has_converged);
    CHECK(has_converged.reason() ==
          LinearSolver::ConvergenceReason::AbsoluteResidual);
    CHECK(get_output(has_converged) ==
          "The linear solver has converged in 1 iterations: AbsoluteResidual - "
          "The residual magnitude has decreased to 0 or below (0).\n");
    test_serialization(has_converged);
    test_copy_semantics(has_converged);
  }

  {
    INFO("HasConverged - RelativeResidual")
    const LinearSolver::HasConverged has_converged{criteria, 1, 1., 2.};
    CHECK(has_converged);
    CHECK(has_converged.reason() ==
          LinearSolver::ConvergenceReason::RelativeResidual);
    CHECK(get_output(has_converged) ==
          "The linear solver has converged in 1 iterations: RelativeResidual - "
          "The residual magnitude has decreased to a fraction of 0.5 of its "
          "initial value or below (0.5).\n");
    test_serialization(has_converged);
    test_copy_semantics(has_converged);
  }
}

// [[OutputRegex, Tried to retrieve the convergence reason, but has not yet
// converged.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearSolver.Convergence.HasConvergedReasonAssert",
    "[Unit][NumericalAlgorithms][LinearSolver]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const LinearSolver::HasConverged has_not_converged{{2, 0., 0.5}, 1, 1., 1.};
  has_not_converged.reason();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
