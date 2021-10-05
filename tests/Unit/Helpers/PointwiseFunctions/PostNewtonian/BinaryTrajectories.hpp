// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <utility>

/*!
 * \brief Class to compute post-Newtonian trajectories
 *
 * \details Computes the leading post-Newtonian trajectories \f$x_1^i(t)\f$
 * and \f$x_2^i(t)\f$ for an equal-mass binary with a total mass of 1.
 * Currently, this class implements the leading-order terms of the integral of
 * Eq. (226) and the square root of Eq. (228) of \cite Blanchet:2013haa :
 * \f{align}{
 *   r(t) &= \left(r_0^4 - \frac{64}{5}t\right)^{1/4}, \\
 *   \Omega(t) &= r^{-3/2}(t).
 * \f}
 * In terms of these functions, the positions of objects 1 and 2 are
 * \f{align}{
 *   x_1(t) &= \frac{r(t)}{2}\cos\left[\Omega(t) t\right], \\
 *   y_1(t) &= \frac{r(t)}{2}\sin\left[\Omega(t) t\right], \\
 *   x_2(t) &= -\frac{r(t)}{2}\cos\left[\Omega(t) t\right], \\
 *   y_2(t) &= -\frac{r(t)}{2}\sin\left[\Omega(t) t\right], \\
 *   z_1(t) &= z_2(t) = 0.
 * \f} These trajectories are useful for, e.g., testing a horizon-tracking
 * control system.
 *
 * \note The trajectories could be generalized to higher post-Newtonian order if
 * needed.
 */
class BinaryTrajectories {
 public:
  BinaryTrajectories(double initial_separation);
  BinaryTrajectories() = default;
  BinaryTrajectories(BinaryTrajectories&&) = default;
  BinaryTrajectories& operator=(BinaryTrajectories&&) = default;
  BinaryTrajectories(const BinaryTrajectories&) = default;
  BinaryTrajectories& operator=(const BinaryTrajectories&) = default;
  ~BinaryTrajectories() = default;

  double separation(double time) const;
  double orbital_frequency(double time) const;
  std::pair<std::array<double, 3>, std::array<double, 3>> positions(
      double time) const;

 private:
  double initial_separation_fourth_power_;
};
