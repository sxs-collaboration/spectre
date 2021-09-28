// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>

#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Convergence {

/*!
 * \brief Determine whether the `criteria` are met.
 *
 * \note This function assumes the `iteration_id` is that of the latest
 * completed step and that it is zero-indexed, where zero indicates the initial
 * state of the algorithm. Therefore, the `MaxIteration` criterion will match if
 * the `iteration_id` is equal or higher. For example, a `MaxIteration` of 0
 * means the algorithm should run no iterations, so it matches if the
 * `iteration_id` is 0 or higher since that's the initial state before any
 * steps have been performed. A `MaxIteration` of 1 matches if the
 * `iteration_id` is 1 or higher since one iteration is complete. At this point,
 * also the `residual_magnitude` reflects the state of the algorithm after
 * completion of the first iteration. The `initial_residual_magnitude` always
 * refers to the state before the first iteration has begun, i.e. where the
 * `iteration_id` is zero.
 *
 * \returns A `Convergence::Reason` if the criteria are met, or
 * `std::nullopt` otherwise. The possible convergence reasons are:
 *
 * - `Convergence::Reason::AbsoluteResidual` if the `residual_magnitude`
 *   meets the convergence criteria's `absolute_residual`.
 * - `Convergence::Reason::RelativeResidual` if `residual_magnitude /
 *   initial_residual_magnitude` meets the convergence criteria's
 *   `relative_residual`.
 * - `Convergence::Reason::MaxIterations` if the `iteration_id` is the
 *   convergence_criteria's `max_iterations` or higher. This is often
 *   interpreted as an error because the algorithm did not converge in the
 *   alloted number of iterations.
 */
std::optional<Reason> criteria_match(const Criteria& criteria,
                                     size_t iteration_id,
                                     double residual_magnitude,
                                     double initial_residual_magnitude);

/*!
 * \brief Signals convergence of the algorithm.
 *
 * \details Evaluates to `true` if the algorithm has converged and no
 * further iterations should be performed. In this case, the `reason()` member
 * function provides more information. If `false`, calling `reason()` is an
 * error.
 *
 * The stream operator provides a human-readable description of the convergence
 * status.
 *
 * This type default-constructs to a state that signals the algorithm has
 * not yet converged.
 */
struct HasConverged {
 public:
  HasConverged() = default;
  /*!
   * \brief Determine whether the \p criteria are met by means of
   * `Convergence::criteria_match`.
   */
  HasConverged(const Criteria& criteria, size_t iteration_id,
               double residual_magnitude, double initial_residual_magnitude);

  /// Construct at a state where `iteration_id` iterations of a total of
  /// `num_iterations` have completed. Use when the algorithm is intended to run
  /// for a fixed number of iterations. The convergence `reason()` will be
  /// `Convergence::Reason::NumIterations`.
  HasConverged(size_t num_iterations, size_t iteration_id);

  explicit operator bool() const { return static_cast<bool>(reason_); }

  /*!
   * \brief The reason the algorithm has converged.
   *
   * \warning Calling this function is an error if the algorithm has not yet
   * converged.
   */
  Reason reason() const;

  /// The number of iterations the algorithm has completed
  size_t num_iterations() const;

  /// The residual magnitude after the last iteration. NaN if no iteration has
  /// completed yet.
  double residual_magnitude() const;

  /// The residual magnitude before the first iteration. NaN if this information
  /// is not available yet.
  double initial_residual_magnitude() const;

  void pup(PUP::er& p);  // NOLINT

  friend bool operator==(const HasConverged& lhs, const HasConverged& rhs);
  friend bool operator!=(const HasConverged& lhs, const HasConverged& rhs);

  friend std::ostream& operator<<(std::ostream& os,
                                  const HasConverged& has_converged);

 private:
  std::optional<Reason> reason_{};
  Criteria criteria_{};
  size_t iteration_id_{};
  double residual_magnitude_{};
  double initial_residual_magnitude_{};
};

}  // namespace Convergence
