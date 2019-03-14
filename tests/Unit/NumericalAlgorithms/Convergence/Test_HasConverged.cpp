// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/none.hpp>
#include <boost/none_t.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.Convergence.HasConverged",
                  "[Unit][NumericalAlgorithms]") {
  const Convergence::Criteria criteria{2, 0., 0.5};

  {
    INFO("Convergence logic");
    CHECK(Convergence::criteria_match(criteria, 1, 1., 1.) == boost::none);
    CHECK(Convergence::criteria_match(criteria, 2, 1., 1.) ==
          Convergence::Reason::MaxIterations);
    CHECK(Convergence::criteria_match(criteria, 1, 0., 1.) ==
          Convergence::Reason::AbsoluteResidual);
    CHECK(Convergence::criteria_match(criteria, 1, 1., 2.) ==
          Convergence::Reason::RelativeResidual);
  }

  {
    INFO("HasNotConverged");
    const Convergence::HasConverged has_not_converged_by_default{};
    CHECK_FALSE(has_not_converged_by_default);
    test_serialization(has_not_converged_by_default);
    test_copy_semantics(has_not_converged_by_default);
    const Convergence::HasConverged has_not_converged{criteria, 1, 1., 1.};
    CHECK_FALSE(has_not_converged);
    CHECK(get_output(has_not_converged) == "Not yet converged.\n");
    test_serialization(has_not_converged);
    test_copy_semantics(has_not_converged);
  }

  {
    INFO("HasConverged - MaxIterations")
    const Convergence::HasConverged has_converged{criteria, 2, 1., 1.};
    CHECK(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::MaxIterations);
    CHECK(get_output(has_converged) ==
          "Reached the maximum number of iterations (2).\n");
    test_serialization(has_converged);
    test_copy_semantics(has_converged);
  }

  {
    INFO("HasConverged - AbsoluteResidual")
    const Convergence::HasConverged has_converged{criteria, 1, 0., 1.};
    CHECK(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(get_output(has_converged) ==
          "AbsoluteResidual - The residual magnitude has decreased to 0 or "
          "below (0).\n");
    test_serialization(has_converged);
    test_copy_semantics(has_converged);
  }

  {
    INFO("HasConverged - RelativeResidual")
    const Convergence::HasConverged has_converged{criteria, 1, 1., 2.};
    CHECK(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::RelativeResidual);
    CHECK(get_output(has_converged) ==
          "RelativeResidual - The residual magnitude has decreased to a "
          "fraction of 0.5 of its initial value or below (0.5).\n");
    test_serialization(has_converged);
    test_copy_semantics(has_converged);
  }
}

// [[OutputRegex, Tried to retrieve the convergence reason, but has not yet
// converged.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.Convergence.HasConvergedReasonAssert",
    "[Unit][NumericalAlgorithms]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Convergence::HasConverged has_not_converged{
      Convergence::Criteria{2, 0., 0.5}, 1, 1., 1.};
  has_not_converged.reason();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
