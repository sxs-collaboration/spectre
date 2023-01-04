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
 *   x_1(t) &= -\frac{r(t)}{2}\cos\left[\Omega(t) t\right] + v_x t, \\
 *   y_1(t) &= -\frac{r(t)}{2}\sin\left[\Omega(t) t\right], + v_y t\\
 *   x_2(t) &= \frac{r(t)}{2}\cos\left[\Omega(t) t\right], + v_x t\\
 *   y_2(t) &= \frac{r(t)}{2}\sin\left[\Omega(t) t\right], + v_y t\\
 *   z_1(t) &= z_2(t) = v_z t.
 * \f} These trajectories are useful for, e.g., testing a horizon-tracking
 * control system.
 *
 * \parblock
 * \note The trajectories could be generalized to higher post-Newtonian order if
 * needed.
 * \endparblock
 * \parblock
 * \note If the `newtonian` argument is true, then this will just give Kepler's
 * third law
 * \endparblock
 */
class BinaryTrajectories {
 public:
  BinaryTrajectories(double initial_separation,
                     const std::array<double, 3>& velocity =
                         std::array<double, 3>{{0.0, 0.0, 0.0}},
                     bool newtonian = false);
  BinaryTrajectories() = default;
  BinaryTrajectories(BinaryTrajectories&&) = default;
  BinaryTrajectories& operator=(BinaryTrajectories&&) = default;
  BinaryTrajectories(const BinaryTrajectories&) = default;
  BinaryTrajectories& operator=(const BinaryTrajectories&) = default;
  ~BinaryTrajectories() = default;

  /// Gives separation as function of time
  double separation(double time) const;
  /// Gives orbital frequency \f$f\f$ as a function of time calculated from
  /// Kepler's third law \f$f^2\propto\frac{1}{a^3}\f$ where \f$a\f$ is
  /// calculated from `separation`.
  double orbital_frequency(double time) const;
  /// Gives the angular velocity of the objects as a function of time.
  /// Calculated by \f$\omega(t)=\frac{d\theta(t)}{dt}\f$ where
  /// \f$\theta(t)=f(t)t\f$.
  ///
  /// \note For the newtonian case, `orbital_frequency` and `angular_velocity`
  /// give the same result because the orbital frequency is independent of time.
  double angular_velocity(const double time) const;
  /// Gives the positions of the two objects as a function of time.
  std::pair<std::array<double, 3>, std::array<double, 3>> positions(
      double time) const;
  /// Same as `positions`, except the separation remains constant (equal to the
  /// initial separation).
  ///
  /// \note This is useful for testing the rotation control system by itself,
  /// because we want the frequency to vary, but the separation to remain the
  /// same. This way, we don't have to worry about expansion effects.
  std::pair<std::array<double, 3>, std::array<double, 3>>
  positions_no_expansion(double time) const;

 private:
  std::pair<std::array<double, 3>, std::array<double, 3>> position_impl(
      double time, double separation) const;

  double initial_separation_fourth_power_;
  std::array<double, 3> velocity_;
  bool newtonian_;
};
