// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.Convergence.Criteria",
                  "[Unit][NumericalAlgorithms]") {
  const Convergence::Criteria criteria{2, 0.1, 0.5};
  CHECK(criteria == Convergence::Criteria{2, 0.1, 0.5});
  CHECK(criteria != Convergence::Criteria{3, 0.1, 0.5});
  CHECK(criteria != Convergence::Criteria{2, 1., 0.5});
  CHECK(criteria != Convergence::Criteria{2, 0.1, 0.6});
  test_serialization(criteria);
  test_copy_semantics(criteria);
  const auto created_criteria = test_creation<Convergence::Criteria>(
      "  MaxIterations: 2\n"
      "  AbsoluteResidual: 0.1\n"
      "  RelativeResidual: 0.5\n");
  CHECK(created_criteria == criteria);
}
