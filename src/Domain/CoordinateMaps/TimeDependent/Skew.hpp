// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps::TimeDependent {
/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief %Time dependent coordinate map that keeps the $y$ and $z$ coordinates
 * unchanged, but will distort (or skew) the $x$ coordinate based on functions
 * of time and radial distance to the origin.
 *
 * \details This coordinate map is only available in 3 dimensions and is
 * intended to be used in the BinaryCompactObject domain.
 *
 * ### Mapped Coordinates
 * The Skew coordinate map is given by the mapping
 *
 * \begin{align}
 *   \label{eq:map}
 *   \bar{x} &= x - W(\vec{x})\left(\tan(F_y(t))(y-y_C) +
 *       \tan(F_z(t))(z-z_C)\right) \\
 *   \bar{y} &= y \\
 *   \bar{z} &= z
 * \end{align}
 *
 * where $\vec{x}_C = (x_C, y_C, z_C)$ is the \p center of the skew map (which
 * is different than the origin of the coordinate system), and $F_y(t)$ and
 * $F_z(t)$ are the angles within the $(x,y)$ plane between the undistorted
 * $x$-axis and the skewed $\bar{y}$-axis at the origin, represented by
 * `domain::FunctionsOfTime::FunctionOfTime`s. The actual function of time
 * should have two components; the first corresponds to $y$ and the second
 * corresponds to $z$.
 *
 * $W(\vec{x})$ is a spatial function that should be 1 at $\vec{x}_C$ (i.e.
 * maximally skewed between the two objects) and fall off to 0 at the \p
 * outer_radius, $R$. Typically the \p outer_radius should be set to the
 * envelope radius for the `domain::creators::BinaryCompactObject`. The reason
 * that $W(\vec{x})$ *should* be 1 at $\vec{x}_C$, and doesn't *need* to be 1 at
 * $\vec{x}_C$ is because having $W(\vec{x})$ centered at the origin is much
 * better for the smoothness of higher derivatives of $W(\vec{x})$ than if it
 * were centered at $\vec{x}_C$. The important part is that the map is left
 * invariant along the $x=x_C$ line and that $W(\vec{x}_C)\approx 1$ for the
 * skew control system, both of which are satisfied if $W(\vec{x})$ is centered
 * at the origin and $|\vec{x}_C|\ll R$.
 *
 * With that in mind, $W$ is chosen to be
 *
 * \begin{equation}
 *   \label{eq:W}
 *   W(\vec{x}) = \frac{1}{2}\left(1 + \cos(\pi\lambda(\vec{x}))\right).
 * \end{equation}
 *
 * When the $\cos$ term is $-1$, then $W = 0$, and when the $\cos$ term is 1,
 * then $W = 1$. Therefore, the function $\lambda(\vec{x})$ must go from 0 at
 * the origin, to 1 at $R$. We choose $\lambda(\vec{x})$ to quadratically go
 * from 0 at the origin to 1 at $R$ with the form
 *
 * \begin{equation}
 *   \label{eq:lambda}
 *   \lambda(\vec{x}) = \frac{|\vec{x}|^2}{R^2}.
 * \end{equation}
 *
 * A quadratic form was chosen for $\lambda$ rather than higher powers of 2 to
 * avoid needing too much resolution to resolve a steep falloff in the function.
 *
 * \note If the quantity $S = \tan(F_y(t))(y-y_C) + \tan(F_z(t))(z-z_C)$ becomes
 * too large, then the map becomes singular because multiple $x$ will be mapped
 * to the same $\bar{x}$. This is due to the fact we are adding a linear term to
 * a cosine term with different weights. If $S$ is too large, the cosine term
 * will overpower the linear and the map will become singular.
 *
 * ### Inverse
 * To find the inverse, we need to solve a 1D root find for the $x$ component of
 * the coordinate. The inverse of the $\bar{y}$ and $\bar{z}$ coordinates are
 * trivial because there was no mapping. The equation we need to find the root
 * of is
 *
 * \begin{equation}
 *   0 = x - W(\vec{x})\left(\tan(F_y(t))(y-y_C) + \tan(F_z(t))(z-z_C)\right) -
 *   \bar{x}
 * \end{equation}
 *
 * We can bound the root by noticing that in $\ref{eq:map}$, if we substitute
 * the extremal values of $W = 0$ and $W = 1$, we get bounds on $\bar{x}$ which
 * we can turn into bounds on $x$:
 *
 * \begin{align}
 *   x &<=& \bar{x} &<=& x - (\tan(F_y)(y-y_C) + \tan(F_z)(z-z_C)) \\
 *   0 &<=& \bar{x} - x &<=& -(\tan(F_y)(y-y_C) + \tan(F_z)(z-z_C)) \\
 *   -\bar{x} &<=& - x &<=& -\bar{x} - (\tan(F_y)(y-y_C) + \tan(F_z)(z-z_C)) \\
 *   \bar{x} &>=& x &>=& \bar{x} + (\tan(F_y)(y-y_C) + \tan(F_z)(z-z_C))
 * \end{align}
 *
 * where on each line we just made simple arithmetic operations. We pad each
 * bound by $10^{-14}$ just to avoid roundoff issues. If either of the bounds is
 * within roundoff of zero, the map is the identity at that point and we forgo
 * the root find. The root that is found is the original $x$ coordinate.
 *
 * ### Frame Velocity
 * Taking the time derivative of $\ref{eq:map}$, the frame velocity is
 *
 * \begin{align}
 *   \label{eq:frame_vel}
 *   \dot{\bar{x}} &= -W(\vec{x})\left(\dot{F}_y(t)(1 + \tan^2(F_y(t)))(y-y_C) +
 *     \dot{F}_z(t)(1 + \tan^2(F_z(t))(z-z_C))\right) \\
 *   \dot{\bar{y}} &= 0 \\
 *   \dot{\bar{z}} &= 0
 * \end{align}
 *
 * ### Jacobian and Inverse Jacobian
 * Considering the first terms in each equation of $\ref{eq:map}$, part of the
 * jacobian will be the identity matrix. The rest will come from only the $x$
 * equation. Therefore we can express the jacobian as
 *
 * \begin{equation}
 *   \frac{\partial\bar{x}^i}{\partial x^j} = \delta^i_j + {W^i}_j
 * \end{equation}
 *
 * where all components of ${W^i}_j$ are zero except the following
 *
 * \begin{align}
 *   {W^0}_0 &= \frac{\partial(\bar{x}-x)}{\partial x} &= -\frac{\partial
 *     W(\vec{x})}{\partial x}&\left(\tan(F_y(t))(y-y_C) +
 *     \tan(F_z(t))(z-z_C)\right), \\
 *   {W^0}_1 &= \frac{\partial(\bar{x}-x)}{\partial y} &= -\frac{\partial
 *     W(\vec{x})}{\partial y}&\left(\tan(F_y(t))(y-y_C) +
 *     \tan(F_z(t))(z-z_C)\right) -
 *     W\tan(F_y(t)), \\
 *   {W^0}_2 &= \frac{\partial(\bar{x}-x)}{\partial z} &= -\frac{\partial
 *     W(\vec{x})}{\partial z}&\left(\tan(F_y(t))(y-y_C) +
 *     \tan(F_z(t))(z-z_C)\right) - W\tan(F_z(t)).
 * \end{align}
 *
 * The gradient of $W(\vec{x})$ (Eq. $\ref{eq:W}$) is given by
 *
 * \begin{equation}
 *   \frac{\partial W(\vec{x})}{\partial x^i} =
 *   -\frac{\pi}{2}\frac{\partial\lambda(\vec{x})}{\partial
 *   x^i}\sin(\pi\lambda(\vec{x})).
 * \end{equation}
 *
 * The gradient of $\lambda(\vec{x})$ (Eq. $\ref{eq:lambda}$) is given by
 *
 * \begin{equation}
 *   \frac{\partial \lambda(\vec{x})}{\partial x^i} = \frac{2x^i}{R^2}
 * \end{equation}
 *
 * The inverse jacobian is computed by numerically inverting the jacobian.
 */
class Skew {
 public:
  static constexpr size_t dim = 3;

  Skew(std::string function_of_time_name, const std::array<double, 3>& center,
       double outer_radius);
  Skew() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords, double time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> frame_velocity(
      const std::array<T, 3>& source_coords, double time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords, double time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords, double time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static bool is_identity() { return false; }

  const std::unordered_set<std::string>& function_of_time_names() const {
    return f_of_t_names_;
  }

 private:
  template <typename T>
  tt::remove_cvref_wrap_t<T> get_width(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> get_width_deriv(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> map_and_velocity_helper(
      const std::array<T, 3>& source_coords, double time,
      const domain::FunctionsOfTimeMap& functions_of_time,
      bool return_velocity) const;

  template <typename T>
  void check_for_singular_map(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Skew& lhs, const Skew& rhs);
  std::string f_of_t_name_;
  std::array<double, 3> center_{};
  double outer_radius_{};
  double one_over_outer_radius_squared_{};
  std::unordered_set<std::string> f_of_t_names_;
};

bool operator!=(const Skew& lhs, const Skew& rhs);

}  // namespace domain::CoordinateMaps::TimeDependent
