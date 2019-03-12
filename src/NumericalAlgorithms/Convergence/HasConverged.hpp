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
 * \brief Determine whether the \p criteria are met.
 *
 * \note This function assumes the \p iteration_id is that of the next, but
 * not yet performed step. For instance, a `MaxIteration` criterion of 1 will
 * match if the \p iteration_id is 1 or higher, since the first iteration
 * (with id 0) has been completed. At this point, also the \p residual_magnitude
 * reflects the state of the algorithm after completion of the first iteration.
 * The `initial_residual_magnitude` always refers to the state before the first
 * iteration has begun.
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

  void pup(PUP::er& p) noexcept;  // NOLINT

  friend bool operator==(const HasConverged& lhs,
                         const HasConverged& rhs) noexcept;
  friend bool operator!=(const HasConverged& lhs,
                         const HasConverged& rhs) noexcept;

  friend std::ostream& operator<<(std::ostream& os,
                                  const HasConverged& has_converged) noexcept;

 private:
  boost::optional<Reason> reason_{boost::none};
  Criteria criteria_{};
  size_t iteration_id_{};
  double residual_magnitude_{};
  double initial_residual_magnitude_{};
};

}  // namespace Convergence
