// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Tags.hpp"

namespace {
struct TestSolver {};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelLinearSolver.Richardson.Tags",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  TestHelpers::db::test_simple_tag<
      LinearSolver::Richardson::Tags::RelaxationParameter<TestSolver>>(
      "RelaxationParameter(TestSolver)");
}
