// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cstddef>
#include <iosfwd>

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
 * \returns a `Convergence::Reason` if the criteria are met, or
 * `boost::none` otherwise.
 */
boost::optional<Reason> criteria_match(
    const Criteria& criteria, size_t iteration_id, double residual_magnitude,
    double initial_residual_magnitude) noexcept;

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
               double residual_magnitude,
               double initial_residual_magnitude) noexcept;

  explicit operator bool() const noexcept { return static_cast<bool>(reason_); }

  /*!
   * \brief The reason the algorithm has converged.
   *
   * \warning Calling this function is an error if the algorithm has not yet
   * converged.
   */
  Reason reason() const noexcept;

  /// The number of iterations the algorithm has completed
  size_t num_iterations() const noexcept;

  /// The residual magnitude after the last iteration. NaN if no iteration has
  /// completed yet.
  double residual_magnitude() const noexcept;

  /// The residual magnitude before the first iteration. NaN if this information
  /// is not available yet.
  double initial_residual_magnitude() const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

  friend bool operator==(const HasConverged& lhs,
                         const HasConverged& rhs) noexcept;
  friend bool operator!=(const HasConverged& lhs,
                         const HasConverged& rhs) noexcept;

  friend std::ostream& operator<<(std::ostream& os,
                                  const HasConverged& has_converged) noexcept;

 private:
  // This default initialization is equivalent to boost::none, but works around
  // a `-Wmaybe-uninitialized` warning on GCC 7 in Release mode
  boost::optional<Reason> reason_ =
      boost::make_optional(false, Reason::MaxIterations);
  Criteria criteria_{};
  size_t iteration_id_{};
  double residual_magnitude_{};
  double initial_residual_magnitude_{};
};

}  // namespace Convergence
