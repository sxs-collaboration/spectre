// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Convergence/Criteria.hpp"

#include <pup.h>  // IWYU pragma: keep

namespace Convergence {

Criteria::Criteria(const size_t max_iterations_in,
                   const double absolute_residual_in,
                   const double relative_residual_in) noexcept
    : max_iterations(max_iterations_in),
      absolute_residual(absolute_residual_in),
      relative_residual(relative_residual_in) {}

void Criteria::pup(PUP::er& p) noexcept {
  p | max_iterations;
  p | absolute_residual;
  p | relative_residual;
}

bool operator==(const Criteria& lhs, const Criteria& rhs) noexcept {
  return lhs.max_iterations == rhs.max_iterations and
         lhs.absolute_residual == rhs.absolute_residual and
         lhs.relative_residual == rhs.relative_residual;
}

bool operator!=(const Criteria& lhs, const Criteria& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Convergence
