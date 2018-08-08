// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <string>

#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.IterationId",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  using Hash = std::hash<LinearSolver::IterationId>;

  const LinearSolver::IterationId id{2};

  CHECK(id.step_number == 2);
  CHECK(id == LinearSolver::IterationId{2});
  CHECK(id != LinearSolver::IterationId{3});
  CHECK(Hash{}(id) == Hash{}(LinearSolver::IterationId{2}));
  CHECK(Hash{}(id) != Hash{}(LinearSolver::IterationId{3}));
  check_cmp(id, LinearSolver::IterationId{3});
  CHECK(get_output(id) == "2");
  test_serialization(id);
  test_copy_semantics(id);

  SECTION("Tag") {
    CHECK(LinearSolver::Tags::IterationId::name() == "IterationId");
  }
}
