// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"

namespace {
struct Tag : db::SimpleTag {
  static std::string name() noexcept { return "Tag"; }
  using type = int;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Tags",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  CHECK(LinearSolver::Tags::Operand<Tag>::name() == "LinearOperand(Tag)");
  CHECK(LinearSolver::Tags::OperatorAppliedTo<Tag>::name() ==
        "LinearOperatorAppliedTo(Tag)");
  CHECK(LinearSolver::Tags::Residual<Tag>::name() == "LinearResidual(Tag)");
  CHECK(LinearSolver::Tags::MagnitudeSquare<Tag>::name() ==
        "LinearMagnitudeSquare(Tag)");
  CHECK(LinearSolver::Tags::Magnitude<Tag>::name() == "LinearMagnitude(Tag)");
  CHECK(LinearSolver::Tags::Orthogonalization<Tag>::name() ==
        "LinearOrthogonalization(Tag)");
  CHECK(LinearSolver::Tags::OrthogonalizationHistory<Tag>::name() ==
        "LinearOrthogonalizationHistory(Tag)");
  CHECK(LinearSolver::Tags::KrylovSubspaceBasis<Tag>::name() ==
        "KrylovSubspaceBasis(Tag)");
}
