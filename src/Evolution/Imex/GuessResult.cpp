// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Imex/GuessResult.hpp"

#include "Utilities/ErrorHandling/Error.hpp"

namespace imex {
[[noreturn]] void NoJacobianBecauseSolutionIsAnalytic::apply() {
  ERROR(
      "NoJacobianBecauseSolutionIsAnalytic was used but initial guess did "
      "not return GuessResult::ExactSolution");
}
}  // namespace imex
