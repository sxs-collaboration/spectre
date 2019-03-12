// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"

#include <boost/optional.hpp>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"

namespace Convergence {

boost::optional<Reason> criteria_match(
    const Criteria& criteria, const size_t iteration_id,
    const double residual_magnitude,
    const double initial_residual_magnitude) noexcept {
  if (residual_magnitude <= criteria.absolute_residual) {
    return Reason::AbsoluteResidual;
  }
  if (residual_magnitude / initial_residual_magnitude <=
      criteria.relative_residual) {
    return Reason::RelativeResidual;
  }
  if (iteration_id >= criteria.max_iterations) {
    return Reason::MaxIterations;
  }
  return boost::none;
}

HasConverged::HasConverged(const Criteria& criteria, const size_t iteration_id,
                           const double residual_magnitude,
                           const double initial_residual_magnitude) noexcept
    : reason_(criteria_match(criteria, iteration_id, residual_magnitude,
                             initial_residual_magnitude)),
      criteria_(criteria),
      iteration_id_(iteration_id),
      residual_magnitude_(residual_magnitude),
      initial_residual_magnitude_(initial_residual_magnitude) {}

Reason HasConverged::reason() const noexcept {
  ASSERT(reason_,
         "Tried to retrieve the convergence reason, but has not yet converged. "
         "Check if the instance of HasConverged evaluates to `true` first.");
  return *reason_;
}

std::ostream& operator<<(std::ostream& os,
                         const HasConverged& has_converged) noexcept {
  if (has_converged) {
    switch (has_converged.reason()) {
      case Reason::MaxIterations:
        return os << "Reached the maximum number of iterations ("
                  << has_converged.criteria_.max_iterations << ").\n";
      case Reason::AbsoluteResidual:
        return os
               << "AbsoluteResidual - The residual magnitude has decreased to "
               << has_converged.criteria_.absolute_residual << " or below ("
               << has_converged.residual_magnitude_ << ").\n";
      case Reason::RelativeResidual:
        return os << "RelativeResidual - The residual magnitude has decreased "
                     "to a fraction of "
                  << has_converged.criteria_.relative_residual
                  << " of its initial value or below ("
                  << has_converged.residual_magnitude_ /
                         has_converged.initial_residual_magnitude_
                  << ").\n";
      default:
        ERROR("Unknown convergence reason");
    };
  } else {
    return os << "Not yet converged.\n";
  }
}

void HasConverged::pup(PUP::er& p) noexcept {
  p | criteria_;
  p | iteration_id_;
  p | residual_magnitude_;
  p | initial_residual_magnitude_;
  if (p.isUnpacking()) {
    reason_ = criteria_match(criteria_, iteration_id_, residual_magnitude_,
                             initial_residual_magnitude_);
  }
}

bool operator==(const HasConverged& lhs, const HasConverged& rhs) noexcept {
  return lhs.criteria_ == rhs.criteria_ and
         lhs.iteration_id_ == rhs.iteration_id_ and
         lhs.residual_magnitude_ == rhs.residual_magnitude_ and
         lhs.initial_residual_magnitude_ == rhs.initial_residual_magnitude_;
}

bool operator!=(const HasConverged& lhs, const HasConverged& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Convergence
