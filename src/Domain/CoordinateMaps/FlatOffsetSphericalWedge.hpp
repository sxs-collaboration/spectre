// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class FlatOffsetSphericalWedge.

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
 * \brief Map from a cube to a volume that connects four planes and
 * portions of two spherical surfaces.
 *
 * \image html FlatOffsetSphericalWedge.svg "Slices of FlatOffsetSphericalWedge"
 *
 * \details A cube is mapped to the volume shown in the figure.
 *
 * A $y=0$ slice through the volume is shown in the left
 * panel of the figure, and a $x=$const slice through the volume is shown
 * in the right panel of the figure.
 *
 * The lower-$z$ face of the volume is a portion of a spherical surface
 * of radius $R_1$ centered about point $C$ in the figure, which is along
 * the $x$-axis a distance $D$ from the origin (point $O$ in the figure).
 * The upper-$z$ face of the volume is a portion of a spherical surface
 * of radius $R_2$ centered about the origin.
 *
 * The lower-$x$ face of the volume is a portion of the plane at constant
 * $x=0$, and the upper-$x$ face of the volume is a portion of the plane
 * at constant $x=D$. See the $x$ extents of the left panel of the
 * figure.
 *
 * Every constant-$x$ cross section of the volume looks like the right
 * panel of the figure: it is a two-dimensional wedge with 45 degree
 * opening angle.  Both the inner and outer radii of this wedge vary
 * with $x$: The inner radius of the wedge is $R_1$ at $x=D$ and
 * $\sqrt{R_1^2-D^2}$ at $x=0$, and the outer radius of the wedge is
 * $R_2$ at $x=0$ and $\sqrt{R_2^2-D^2}$ at $x=D$.
 *
 * ### Relation to FlatOffsetWedge
 *
 * The lower-$z$ face of this FlatOffsetSphericalWedge is the same as
 * the upper-$z$ face of the similar map FlatOffsetWedge; Blocks using
 * these two maps are meant to abut at this surface.  The parameter
 * $D$ means the same thing for the two maps, and the parameter $R$ in
 * FlatOffsetWedge is the same as what we call $R_1$ here.
 * For abutting blocks, the corresponding parameters of the two maps should
 * be equal, and both maps should have the same origin.
 *
 * The formulas below will be similar to those of the FlatOffsetWedge map,
 * but slightly more complicated.
 *
 * ### Restrictions
 *
 * We require $D<R_1$ or else the lower-$x$ face of the mapped
 * volume lies outside of the sphere of radius $R_1$.
 * We also require that $R_2^2 > R_1^2+D^2$ or else the outer sphere
 * and inner sphere will intersect at the upper-$x$ face and the map
 * will be singular.
 *
 * ## Equations for the map
 * Given our cube coordinates $\xi,\eta,\zeta$, each taking on values from
 * $-1$ to $+1$, we can derive the formulas for the map.
 *
 * Define
 * \begin{align}
 *    q &\equiv \frac{D}{2R_1},\\
 *    v &\equiv \frac{R_1}{R_2}.
 * \end{align}
 *
 * Notice that $q<1/2$, because of the restriction $D<R_1$.  Also we
 * must have $v < \left(1+4q^2\right)^{-1/2}$ because of the
 * restriction $R_2^2 > R_1^2+D^2$.
 *
 * Note that $x$ is a function of $\xi$ only, for all points in the volume:
 * \begin{align}
 *  x(\xi,\eta,\zeta) &= q R (\xi+1).
 * \end{align}
 *
 * ### Surface map for bottom and top surfaces
 * The $y$ and $z$ coordinates of the bottom surface,
 * $\zeta=-1$, of the volume are given by
 * \begin{align}
 *   \begin{bmatrix}
 *      y(\xi,\eta,+1)\\
 *      z(\xi,\eta,+1)
 *   \end{bmatrix} =
 *     \frac{R_1}{\sqrt{1+\eta^2}}
 *               \sqrt{1-q^2(\xi-1)^2}
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}.
 * \end{align}
 * This is the same formula as for the upper face of the similar
 * FlatOffsetWedge map.
 *
 * The $y$ and $z$ coordinates for the mapped $\zeta=+1$ surface are
 * \begin{align}
 *   \begin{bmatrix}
 *      y(\xi,\eta,+1)\\
 *      z(\xi,\eta,+1)
 *   \end{bmatrix} =
 *     \frac{R_2}{\sqrt{1+\eta^2}}
 *              \sqrt{1-v^2q^2(\xi+1)^2}
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}.
 * \end{align}
 * Because of the above restrictions on $v$ and $q$, and because $\xi$ and
 * $\eta$ range between $-1$ and $+1$, the arguments of the square
 * roots above are always positive.
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
 *      \left[\frac{1+\zeta}{2}
 *        \frac{R_2}{\sqrt{1+\eta^2}}
 *                   \sqrt{1-v^2q^2(\xi+1)^2}
 *        + \frac{1-\zeta}{2}
 *          \frac{R_1}{\sqrt{1+\eta^2}}
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
 *    \zeta &= \frac{2z - W - P}{W-P},
 *              \label{eq:zetaFromxyz}
 * \end{align}
 * where
 * \begin{align}
 *   P &\equiv R_1\frac{\sqrt{1-(x-D)^2/R_1^2}}{\sqrt{1+y^2/z^2}}\\
 *     &= R_1\frac{\sqrt{1-q^2(\xi-1)^2}}{\sqrt{1+\eta^2}},
 *              \label{eq:Pdefinition} \\
 *   W &\equiv R_2\frac{\sqrt{1-x^2/R_2^2}}{\sqrt{1+y^2/z^2}}\\
 *     &= R_2\frac{\sqrt{1-v^2q^2(\xi+1)^2}}{\sqrt{1+\eta^2}}.
 *              \label{eq:Wdefinition}
 * \end{align}
 *
 * It is easy to determine whether a point $(x,y,z)$ lies within the volume:
 * First compute $\xi$ and $\eta$ and check that they are both in $[-1,1]$.
 * If so, then the arguments of the square roots in
 * Eq. ($\ref{eq:Pdefinition}$) and Eq. ($\ref{eq:Wdefinition}$) are
 * guaranteed positive, and the denominator
 * of Eq. ($\ref{eq:zetaFromxyz}$) is guaranteed positive
 * by our condition $v < 1/\sqrt{1+4q^2}$.
 * Then it is straightforward to
 * compute $\zeta$ and then check if it is in $[-1,1]$.
 *
 * ## Jacobian
 *
 * Straightforward differentiation gives
 *
 * \begin{align}
 *    \frac{\partial x}{\partial \xi}   &= qR_1,\\
 *    \frac{\partial x}{\partial \eta}   &= 0,\\
 *    \frac{\partial x}{\partial \zeta}   &= 0,\\
 *   \partial_\zeta \begin{bmatrix}
 *      y\\
 *      z
 *   \end{bmatrix} &= \frac{W-P}{2}
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}, \\
 *   \partial_\xi \begin{bmatrix}
 *      y\\
 *      z
 *   \end{bmatrix} &=
 *    \frac{q^2}{2}\left[
 *    P\frac{(1-\zeta)(1-\xi)}{1-q^2(1-\xi)^2}
 *    -W\frac{v^2(1+\zeta)(1+\xi)}{1-v^2q^2(1+\xi)^2}
 *    \right]
 *   \begin{bmatrix}
 *      \eta \\
 *      1
 *   \end{bmatrix}, \\
 *   \partial_\eta \begin{bmatrix}
 *      y\\
 *      z
 *   \end{bmatrix} &=
 *       -\frac{\eta}{2(1+\eta^2)}\left[
 *        W(1+\zeta)
 *       +P(1-\zeta)\right]
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
 * Although it is straightforward to construct the inverse jacobian
 * analytically, for now we compute the inverse jacobian by numerically taking
 * the matrix inverse of the jacobian.
 *
 */
class FlatOffsetSphericalWedge {
 public:
  static constexpr size_t dim = 3;
  /*!
   * \brief Constructs a FlatOffsetSphericalWedge.
   *
   * \param lower_face_x_width The width $D$ of the lower face in
   * the $x$ direction.
   * \param inner_radius The inner radius $R_1$.
   * \param outer_radius The outer radius $R_2$.
   */
  FlatOffsetSphericalWedge(double lower_face_x_width, double inner_radius,
                           double outer_radius);

  FlatOffsetSphericalWedge() = default;
  ~FlatOffsetSphericalWedge() = default;
  FlatOffsetSphericalWedge(FlatOffsetSphericalWedge&&) = default;
  FlatOffsetSphericalWedge(const FlatOffsetSphericalWedge&) = default;
  FlatOffsetSphericalWedge& operator=(const FlatOffsetSphericalWedge&) =
      default;
  FlatOffsetSphericalWedge& operator=(FlatOffsetSphericalWedge&&) = default;

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

 private:
  friend bool operator==(const FlatOffsetSphericalWedge& lhs,
                         const FlatOffsetSphericalWedge& rhs);
  double lower_face_x_width_{std::numeric_limits<double>::signaling_NaN()};
  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const FlatOffsetSphericalWedge& lhs,
                const FlatOffsetSphericalWedge& rhs);
}  // namespace domain::CoordinateMaps
