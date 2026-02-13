// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class FlattOffsetWedge.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Map from a cube to a volume that connects three planes and
 * a portion of one spherical surface.
 *
 * \image html FlatOffsetWedge.svg "Two slices through a FlatOffsetWedge."
 *
 * \details A cube is mapped to the volume shown in the figure.
 *
 * A $y=0$ slice through the volume is shown in the left
 * panel of the figure, and a $x=$const slice through the volume is shown
 * in the right panel of the figure.
 *
 * The upper-$z$ face of the volume is a portion of a spherical surface
 * of radius $R$ centered about point $C$ in the figure, which is along
 * the $x$-axis a distance $D$ from the origin (point $O$ in the figure).
 * The lower-$z$ face of the volume is a two-dimensional rectangle of
 * constant $z$, located a distance $L$ above the origin.
 * This rectangle extends from
 * $0 \leq x \leq D$ in the $x$ direction, and from
 * $-L \leq y \leq+L$ in the $y$ direction.
 *
 * The lower-$x$ face of the volume is a portion of the plane at constant
 * $x=0$, and the upper-$x$ face of the volume is a portion of the plane
 * at constant $x=D$. See the $x$ extents of the left panel of the
 * figure.
 *
 * Every constant-$x$ cross section of the volume looks like the right panel
 * of the figure: it is a two-dimensional wedge with 45 degree
 * opening angle and a base of
 * width $2L$.  The outer radius of this wedge varies with $x$ (this radius
 * is $R$ at $x=D$ and $\sqrt{R^2-D^2}$ at $x=0$).
 *
 * ### Restrictions
 *
 * We require $D^2 + L^2 < R^2$ or else the lower-$x$ face of the mapped
 * volume lies outside of the sphere of radius $R$.  This implies $L<R$
 * and $D<R$.
 * However, the condition that the upper-$z$ and lower-$z$ surface don't touch
 * each other at the $\xi=-1$, $\eta=\pm 1$ corners is more stringent:
 * $D^2 + 2L^2 < R^2$.  This implies that $L<R/\sqrt{2}$.
 *
 * ## Equations for the map
 * Given our cube coordinates $\xi,\eta,\zeta$, each taking on values from
 * $-1$ to $+1$, we can derive the formulas for the map.
 *
 * Define
 * \begin{equation}
 *    q \equiv \frac{D}{2R}.
 * \end{equation}
 *
 * Notice that $q<1/2$, because of the restriction $D<R$.
 *
 * First, note that $x$ is a function of $\xi$ only, for all points in
 * the cube:
 * \begin{align}
 *  x(\xi,\eta,\zeta) &= q R (\xi+1).
 * \end{align}
 *
 * ### Surface map for bottom and top surfaces
 * The $y$ and $z$ coordinates for the mapped $\zeta=-1$ surface are
 * \begin{align}
 *   \begin{bmatrix}
 *      y(\xi,\eta,-1)\\
 *      z(\xi,\eta,-1)
 *   \end{bmatrix} = L
 *   \begin{bmatrix}
 *      \eta\\
 *      1
 *   \end{bmatrix}.
 * \end{align}
 *
 * The $y$ and $z$ coordinates of the top surface,
 * $\zeta=+1$, of the volume are given by
 * \begin{align}
 *   \begin{bmatrix}
 *      y(\xi,\eta,+1)\\
 *      z(\xi,\eta,+1)
 *   \end{bmatrix} =
 *     \frac{R}{\sqrt{1+\eta^2}}
 *              \sqrt{1-q^2(\xi-1)^2}
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}.
 * \end{align}
 *
 * ### Full volume map
 * Adding the $\zeta$ dependence by linear interpolation gives us
 * the full volume map for the $y$ and $z$ coordinates:
 *
 * \begin{align}
 *   \begin{bmatrix}
 *      y(\xi,\eta,\zeta)\\
 *      z(\xi,\eta,\zeta)
 *   \end{bmatrix} =
 *      \left[L \frac{1-\zeta}{2}
 *        + \frac{1+\zeta}{2}
 *          \frac{R}{\sqrt{1+\eta^2}}
 *           \sqrt{1-q^2(\xi-1)^2}\right]
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}.
 * \end{align}
 *
 * ## Inverse
 *
 * The map can be inverted analytically:
 *
 * \begin{align}
 *    \xi   &= \frac{x}{qR}-1\\
 *    \eta  &= \frac{y}{z}\\
 *    \zeta &= \frac{2z - L - P}{P-L},
 *              \label{eq:zetaFromxyz}
 * \end{align}
 * where
 * \begin{align}
 *   P &\equiv R\frac{\sqrt{1-(x-D)^2/R^2}}{\sqrt{1+y^2/z^2}}\\
 *     &= R\frac{\sqrt{1-q^2(\xi-1)^2}}{\sqrt{1+\eta^2}}.
 *              \label{eq:Pdefinition}
 * \end{align}
 *
 * It is easy to determine whether a point $(x,y,z)$ lies within the volume:
 * First compute $\xi$ and $\eta$ and check that they are both in $[-1,1]$.
 * If so, then the arguments of the square roots in
 * Eq. ($\ref{eq:Pdefinition}$) are guaranteed positive, and the denominator
 * of Eq. ($\ref{eq:zetaFromxyz}$) is guaranteed positive
 * by our condition $R^2>2L^2+D^2$,
 * so it is straightforward to
 * compute $\zeta$ and then check if it is in $[-1,1]$.
 *
 * ## Jacobian
 *
 * Straightforward differentiation gives
 *
 * \begin{align}
 *    \frac{\partial x}{\partial \xi}   &= qR\\
 *    \frac{\partial x}{\partial \eta}   &= 0\\
 *    \frac{\partial x}{\partial \zeta}   &= 0,
 * \end{align}
 *
 * \begin{align}
 *   \partial_\zeta \begin{bmatrix}
 *      y\\
 *      z
 *   \end{bmatrix} &= \frac{1}{2}
 *    \left[\frac{R}{\sqrt{1+\eta^2}}\sqrt{1-q^2(\xi-1)^2}
 *    -L\right]
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}, \\
 *   \partial_\xi \begin{bmatrix}
 *      y\\
 *      z
 *   \end{bmatrix} &=
 *    \left[\frac{Rq^2 (1+\zeta)(1-\xi)}{2\sqrt{1+\eta^2}}
 *    \left(1-q^2(\xi-1)^2\right)^{-1/2}\right]
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}, \\
 *   \partial_\eta \begin{bmatrix}
 *      y\\
 *      z
 *   \end{bmatrix} &=
 *       -\left[\frac{R\eta(1+\zeta)}{2(1+\eta^2)^{3/2}}
 *        \sqrt{1-q^2(\xi-1)^2}\right]
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}+
 *   \begin{bmatrix}
 *      z \\
 *      0
 *   \end{bmatrix}.
 * \end{align}
 *
 * ## Inverse Jacobian
 *
 * Let
 * \begin{align}
 *   \Xi &\equiv P\frac{1+\zeta}{P-L}.
 * \end{align}
 * Then
 * \begin{align}
 *    \frac{\partial \zeta}{\partial x}  &=
 *        \Xi q\frac{\xi-1}{R\sqrt{1-q^2(\xi-1)^2}},\\
 *    \frac{\partial \zeta}{\partial y}  &=
 *        \Xi \frac{\eta}{z\sqrt{1+\eta^2}},\\
 *    \frac{\partial \zeta}{\partial z}  &=
 *        -\Xi \eta\frac{\eta}{z\sqrt{1+\eta^2}}
 *     +\frac{2}{P-L},\\
 *    \frac{\partial \xi}{\partial x}   &= \frac{1}{qR}\\
 *    \frac{\partial \xi}{\partial y}   &= 0\\
 *    \frac{\partial \xi}{\partial z}   &= 0\\
 *    \frac{\partial \eta}{\partial x}  &= 0\\
 *    \frac{\partial \eta}{\partial y}  &= \frac{1}{z}\\
 *    \frac{\partial \eta}{\partial z}  &= -\frac{\eta}{z}.
 * \end{align}
 *
 */
class FlatOffsetWedge {
 public:
  static constexpr size_t dim = 3;
  /*!
   * \brief Constructs a FlatOffsetWedge.
   *
   * \param lower_face_y_half_width The half-width $L$ of the lower face in
   * the $y$ direction.
   * \param lower_face_x_width The width $D$ of the lower face in
   * the $x$ direction.
   * \param outer_radius The outer radius $R$.
   */
  FlatOffsetWedge(double lower_face_y_half_width, double lower_face_x_width,
                  double outer_radius);

  FlatOffsetWedge() = default;
  ~FlatOffsetWedge() = default;
  FlatOffsetWedge(FlatOffsetWedge&&) = default;
  FlatOffsetWedge(const FlatOffsetWedge&) = default;
  FlatOffsetWedge& operator=(const FlatOffsetWedge&) = default;
  FlatOffsetWedge& operator=(FlatOffsetWedge&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  static bool is_identity() { return false; }

  static constexpr bool supports_hessian{false};

 private:
  friend bool operator==(const FlatOffsetWedge& lhs,
                         const FlatOffsetWedge& rhs);
  double lower_face_y_half_width_{std::numeric_limits<double>::signaling_NaN()};
  double lower_face_x_width_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const FlatOffsetWedge& lhs, const FlatOffsetWedge& rhs);
}  // namespace domain::CoordinateMaps
