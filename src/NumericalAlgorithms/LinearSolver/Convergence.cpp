// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearSolver/Convergence.hpp"

#include <boost/optional.hpp>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"

namespace LinearSolver {

ConvergenceCriteria::ConvergenceCriteria(
    const size_t max_iterations_in, const double absolute_residual_in,
    const double relative_residual_in) noexcept
    : max_iterations(max_iterations_in),
      absolute_residual(absolute_residual_in),
      relative_residual(relative_residual_in) {}

void ConvergenceCriteria::pup(PUP::er& p) noexcept {
  p | max_iterations;
  p | absolute_residual;
  p | relative_residual;
}

bool operator==(const ConvergenceCriteria& lhs,
                const ConvergenceCriteria& rhs) noexcept {
  return lhs.max_iterations == rhs.max_iterations and
         lhs.absolute_residual == rhs.absolute_residual and
         lhs.relative_residual == rhs.relative_residual;
}

bool operator!=(const ConvergenceCriteria& lhs,
                const ConvergenceCriteria& rhs) noexcept {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os,
                         const ConvergenceReason& convergence_reason) noexcept {
  switch (convergence_reason) {
    case ConvergenceReason::MaxIterations:
      return os << "MaxIterations";
    case ConvergenceReason::AbsoluteResidual:
      return os << "AbsoluteResidual";
    case ConvergenceReason::RelativeResidual:
      return os << "RelativeResidual";
    default:
      ERROR("Unknown convergence reason");
  }
}

boost::optional<ConvergenceReason> convergence_criteria_match(
    const ConvergenceCriteria& convergence_criteria, const size_t iteration_id,
    const double residual_magnitude,
    const double initial_residual_magnitude) noexcept {
  if (residual_magnitude <= convergence_criteria.absolute_residual) {
    return ConvergenceReason::AbsoluteResidual;
  }
  if (residual_magnitude / initial_residual_magnitude <=
      convergence_criteria.relative_residual) {
    return ConvergenceReason::RelativeResidual;
  }
  if (iteration_id >= convergence_criteria.max_iterations) {
    return ConvergenceReason::MaxIterations;
  }
  return boost::none;
}

HasConverged::HasConverged(const ConvergenceCriteria& convergence_criteria,
                           const size_t iteration_id,
                           const double residual_magnitude,
                           const double initial_residual_magnitude) noexcept
    : reason_(convergence_criteria_match(convergence_criteria, iteration_id,
                                         residual_magnitude,
                                         initial_residual_magnitude)),
      convergence_criteria_(convergence_criteria),
      iteration_id_(iteration_id),
      residual_magnitude_(residual_magnitude),
      initial_residual_magnitude_(initial_residual_magnitude) {}

ConvergenceReason HasConverged::reason() const noexcept {
  ASSERT(reason_,
         "Tried to retrieve the convergence reason, but has not yet converged. "
         "Check if the instance of HasConverged evaluates to `true` first.");
  return *reason_;
}

std::ostream& operator<<(std::ostream& os,
                         const HasConverged& has_converged) noexcept {
  if (has_converged) {
    switch (has_converged.reason()) {
      case ConvergenceReason::MaxIterations:
        return os << "The linear solver has reached its maximum number of "
                     "iterations ("
                  << has_converged.convergence_criteria_.max_iterations
                  << ").\n";
      case ConvergenceReason::AbsoluteResidual:
        return os << "The linear solver has converged in "
                  << has_converged.iteration_id_
                  << " iterations: AbsoluteResidual - The residual magnitude "
                     "has decreased to "
                  << has_converged.convergence_criteria_.absolute_residual
                  << " or below (" << has_converged.residual_magnitude_
                  << ").\n";
      case ConvergenceReason::RelativeResidual:
        return os << "The linear solver has converged in "
                  << has_converged.iteration_id_
                  << " iterations: RelativeResidual - The residual magnitude "
                     "has decreased to a fraction of "
                  << has_converged.convergence_criteria_.relative_residual
                  << " of its initial value or below ("
                  << has_converged.residual_magnitude_ /
                         has_converged.initial_residual_magnitude_
                  << ").\n";
      default:
        ERROR("Unknown convergence reason");
    };
  } else {
    return os << "The linear solver has not yet converged.\n";
  }
}

void HasConverged::pup(PUP::er& p) noexcept {
  p | convergence_criteria_;
  p | iteration_id_;
  p | residual_magnitude_;
  p | initial_residual_magnitude_;
  if (p.isUnpacking()) {
    reason_ = convergence_criteria_match(convergence_criteria_, iteration_id_,
                                         residual_magnitude_,
                                         initial_residual_magnitude_);
  }
}

bool operator==(const HasConverged& lhs, const HasConverged& rhs) noexcept {
  return lhs.convergence_criteria_ == rhs.convergence_criteria_ and
         lhs.iteration_id_ == rhs.iteration_id_ and
         lhs.residual_magnitude_ == rhs.residual_magnitude_ and
         lhs.initial_residual_magnitude_ == rhs.initial_residual_magnitude_;
}

bool operator!=(const HasConverged& lhs, const HasConverged& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace LinearSolver
