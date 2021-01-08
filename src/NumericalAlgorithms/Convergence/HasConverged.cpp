// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"

#include <optional>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Convergence {

std::optional<Reason> criteria_match(
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
  return std::nullopt;
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

HasConverged::HasConverged(const size_t num_iterations,
                           const size_t iteration_id) noexcept
    : reason_(iteration_id >= num_iterations
                  ? std::optional(Reason::NumIterations)
                  : std::nullopt),
      // Store the target num iterations in the convergence criteria's
      // max_iterations
      criteria_(num_iterations, 0, 0),
      iteration_id_(iteration_id),
      residual_magnitude_(std::numeric_limits<double>::max()),
      initial_residual_magnitude_(std::numeric_limits<double>::max()) {}

Reason HasConverged::reason() const noexcept {
  ASSERT(reason_,
         "Tried to retrieve the convergence reason, but has not yet converged. "
         "Check if the instance of HasConverged evaluates to `true` first.");
  return *reason_;
}

size_t HasConverged::num_iterations() const noexcept { return iteration_id_; }

double HasConverged::residual_magnitude() const noexcept {
  return residual_magnitude_;
}

double HasConverged::initial_residual_magnitude() const noexcept {
  return initial_residual_magnitude_;
}

std::ostream& operator<<(std::ostream& os,
                         const HasConverged& has_converged) noexcept {
  if (has_converged) {
    switch (has_converged.reason()) {
      case Reason::NumIterations:
        return os << "Reached the target number of iterations ("
                  // The target num iterations is internally stored in the
                  // max_iterations field
                  << has_converged.criteria_.max_iterations << ").";
      case Reason::MaxIterations:
        return os << "Reached the maximum number of iterations ("
                  << has_converged.criteria_.max_iterations << ").";
      case Reason::AbsoluteResidual:
        return os
               << "AbsoluteResidual - The residual magnitude has decreased to "
               << has_converged.criteria_.absolute_residual << " or below ("
               << has_converged.residual_magnitude_ << ").";
      case Reason::RelativeResidual:
        return os << "RelativeResidual - The residual magnitude has decreased "
                     "to a fraction of "
                  << has_converged.criteria_.relative_residual
                  << " of its initial value or below ("
                  << has_converged.residual_magnitude_ /
                         has_converged.initial_residual_magnitude_
                  << ").";
      default:
        ERROR("Unknown convergence reason");
    };
  } else {
    return os << "Not yet converged.";
  }
}

void HasConverged::pup(PUP::er& p) noexcept {
  p | reason_;
  p | criteria_;
  p | iteration_id_;
  p | residual_magnitude_;
  p | initial_residual_magnitude_;
}

bool operator==(const HasConverged& lhs, const HasConverged& rhs) noexcept {
  return lhs.reason_ == rhs.reason_ and lhs.criteria_ == rhs.criteria_ and
         lhs.iteration_id_ == rhs.iteration_id_ and
         lhs.residual_magnitude_ == rhs.residual_magnitude_ and
         lhs.initial_residual_magnitude_ == rhs.initial_residual_magnitude_;
}

bool operator!=(const HasConverged& lhs, const HasConverged& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Convergence
