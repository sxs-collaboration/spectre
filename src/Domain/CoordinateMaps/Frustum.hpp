// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief A reorientable map from the cube to a frustum.
 * \image html frustum_table1.png "A frustum with rectangular bases."
 *
 * ### Description and Specifiable Parameters
 * A map from the logical cube to the volume determined by interpolating between
 * two parallel rectangles each perpendicular to the \f$z\f$-axis. A Frustum map
 * \f$\vec{x}(\xi,\eta,\zeta)\f$ is determined by specifying two
 * heights \f$z_1 = \f$ `lower_bound` and \f$z_2 = \f$ `upper_bound` for the
 * positions of the rectangular bases of the frustum along the \f$z-\f$axis,
 * and the sizes of the rectangles are determined by the eight values passed to
 * `face_vertices`:
 *
 * \f{align*}{
 * &\textrm{lower x upper base} : x^{+\zeta}_{-\xi,-\eta} = x(-1,-1,1)\\
 * &\textrm{lower x lower base} : x^{-\zeta}_{-\xi,-\eta} = x(-1,-1,-1)\\
 * &\textrm{upper x upper base} : x^{+\zeta}_{+\xi,+\eta} = x(1,1,1)\\
 * &\textrm{upper x lower base} : x^{-\zeta}_{+\xi,+\eta} = x(1,1,-1)\\
 * &\textrm{lower y upper base} : y^{+\zeta}_{-\xi,-\eta} = y(-1,-1,1)\\
 * &\textrm{lower y lower base} : y^{-\zeta}_{-\xi,-\eta} = y(-1,-1,-1)\\
 * &\textrm{upper y upper base} : y^{+\zeta}_{+\xi,+\eta} = y(1,1,1)\\
 * &\textrm{upper y lower base} : y^{-\zeta}_{+\xi,+\eta} = y(1,1,-1)\f}
 *
 * \note As an example, consider a frustum along the z-axis, with the lower base
 * starting at (x,y) = (-2.0,3.0) and extending to
 * (2.0,5.0), and with the upper base extending from (0.0,1.0) to (1.0,3.0).
 * The corresponding value for `face_vertices` is `{{{{-2.0,3.0}}, {{2.0,5.0}},
 * {{0.0,1.0}}, {{1.0,3.0}}}}`.
 *
 * In the case where the two rectangles are geometrically similar, the volume
 * formed is a geometric frustum. However, this coordinate map generalizes to
 * rectangles which need not be similar.
 * The user may reorient the frustum by passing an OrientationMap to the
 * constructor. If `with_equiangular_map` is true, then this coordinate map
 * applies a tangent function mapping to the logical \f$\xi\f$ and \f$\eta\f$
 * coordinates. We then refer to the generalized
 * logical coordinates as \f$\Xi\f$ and \f$\mathrm{H}\f$. If
 * `projective_scale_factor` is set to a quantity other than unity, then this
 * coordinate map applies a rational function mapping to the logical \f$\zeta\f$
 * coordinate. This generalized logical coordinate is referred to as
 * \f$\mathrm{Z}\f$. If `auto_projective_scale_factor` is `true`, the
 * user-specified `projective_scale_factor` is ignored and an appropriate value
 * for `projective_scale_factor` is computed based on the values passed to
 * `face_vertices`. See the
 * [page on redistributing gridpoints](@ref redistributing_gridpoints)
 * to see more detailed information on equiangular variables and projective
 * scaling.
 *
 * ### Coordinate Map and Jacobian
 * In terms of the `face_vertices` variables, we define the
 * following auxiliary variables:
 *
 * \f{align*}{\Sigma^x &= \frac{1}{4}(x^{+\zeta}_{-\xi,-\eta} +
 * x^{+\zeta}_{+\xi,+\eta} + x^{-\zeta}_{-\xi,-\eta} + x^{-\zeta}_{+\xi,+\eta})
 * \\
 * \Delta^x_{\zeta} &= \frac{1}{4}(x^{+\zeta}_{-\xi,-\eta} +
 * x^{+\zeta}_{+\xi,+\eta} - x^{-\zeta}_{-\xi,-\eta} - x^{-\zeta}_{+\xi,+\eta})
 * \\
 * \Delta^x_{\xi} &= \frac{1}{4}(x^{+\zeta}_{+\xi,+\eta} -
 * x^{+\zeta}_{-\xi,-\eta} + x^{-\zeta}_{+\xi,+\eta} - x^{-\zeta}_{-\xi,-\eta})
 * \\
 * \Delta^x_{\xi\zeta} &= \frac{1}{4}(x^{+\zeta}_{+\xi,+\eta} -
 * x^{+\zeta}_{-\xi,-\eta} - x^{-\zeta}_{+\xi,+\eta} + x^{-\zeta}_{-\xi,-\eta})
 * \\
 * \Sigma^y &= \frac{1}{4}(y^{+\zeta}_{-\xi,-\eta} +
 * y^{+\zeta}_{+\xi,+\eta} + y^{-\zeta}_{-\xi,-\eta} + y^{-\zeta}_{+\xi,+\eta})
 * \\
 * \Delta^y_{\zeta} &= \frac{1}{4}(y^{+\zeta}_{-\xi,-\eta} +
 * y^{+\zeta}_{+\xi,+\eta} - y^{-\zeta}_{-\xi,-\eta} - y^{-\zeta}_{+\xi,+\eta})
 * \\
 * \Delta^y_{\eta} &= \frac{1}{4}(y^{+\zeta}_{+\xi,+\eta} -
 * y^{+\zeta}_{-\xi,-\eta} + y^{-\zeta}_{+\xi,+\eta} - y^{-\zeta}_{-\xi,-\eta})
 * \\
 * \Delta^y_{\eta\zeta} &= \frac{1}{4}(y^{+\zeta}_{+\xi,+\eta} -
 * y^{+\zeta}_{-\xi,-\eta} - y^{-\zeta}_{+\xi,+\eta} + y^{-\zeta}_{-\xi,-\eta})
 * \\
 * \Sigma^z &= \frac{z_1 + z_2}{2}\\
 * \Delta^z &= \frac{z_2 - z_1}{2}
 * \f}
 *
 * The full map is then given by:
 * \f[\vec{x}(\xi,\eta,\zeta) = \begin{bmatrix}
 * \Sigma^x + \Delta^x_{\xi}\Xi + (\Delta^x_{\zeta} +
 * \Delta^x_{\xi\zeta}\Xi)\mathrm{Z}\\
 * \Sigma^y + \Delta^y_{\eta}\mathrm{H} + (\Delta^y_{\zeta} +
 * \Delta^y_{\eta\zeta}\mathrm{H})\mathrm{Z}\\
 * \Sigma^z + \Delta^z\mathrm{Z}\\
 * \end{bmatrix}\f]
 *
 * With Jacobian: \f[J =
 * \begin{bmatrix}
 * (\Delta^x_{\xi} + \Delta^x_{\xi\zeta}\mathrm{Z})\Xi' &
 * 0 & (\Delta^x_{\zeta}+ \Delta^x_{\xi\zeta}\Xi)\mathrm{Z}' \\
 * 0 & (\Delta^y_{\eta} + \Delta^y_{\eta\zeta}\mathrm{Z})\mathrm{H}' &
 * (\Delta^y_{\zeta} + \Delta^y_{\eta\zeta}\mathrm{H})\mathrm{Z}'\\
 * 0 & 0 & \Delta^z\mathrm{Z}' \\
 * \end{bmatrix}
 * \f]
 *
 * ### Suggested values for the projective scale factor
 * \note This section assumes familiarity with projective scaling as discussed
 * on the [page on redistributing gridpoints](@ref redistributing_gridpoints).
 *
 * When constructing a Frustum map, it is not immediately obvious what value of
 * \f$w_{\delta}\f$ to use in the projective map; here we present a choice of
 * \f$w_{\delta}\f$ that will produce minimal grid distortion. We cover two
 * cases: the first is the special case in which the two rectangular Frustum
 * bases are related by a simple scale factor; the second is the general case.
 *
 * \image html ProjectionOfCube.png "Trapezoids from squares. (Davide Cervone)"
 *
 * As seen in Cervone's [Cubes and Hypercubes Rotating]
 * (http://www.math.union.edu/~dpvc/math/4D/rotation/welcome.html), there is a
 * special case in which the inverse projection of a trapezoid is not another
 * trapezoid, but a rectangle where the bases are congruent.
 * Most often one will want to use the special value of \f$w_{\delta}\f$ where
 * this occurs. This value of \f$w_{\delta}\f$ corresponds to the projective
 * transformation mapping a rectangular prism in an ambient 4D space with
 * congruent faces at \f$w=1\f$ and \f$w=w_{\delta}\f$ to the frustum in the
 * plane \f$w=1\f$:
 *
 * \f[w_{\delta} = \frac{L_1}{L_2}\f]
 *
 * where \f$\frac{L_1}{L_2}\f$ is the ratio between any pair of
 * corresponding side lengths of the \f$z_1\f$ and \f$z_2\f$-bases of the
 * frustum, respectively.
 *
 * For the general case one will want to use the value:
 * \f[w_{\delta} =
 * \frac{\sqrt{(x^{-\zeta}_{+\xi,+\eta} - x^{-\zeta}_{-\xi,+\eta})
 *             (y^{-\zeta}_{+\xi,+\eta} - y^{-\zeta}_{+\xi,-\eta})}}
 * {\sqrt{(x^{+\zeta}_{+\xi,+\eta} - x^{+\zeta}_{-\xi,+\eta})
 *        (y^{+\zeta}_{+\xi,+\eta} - y^{+\zeta}_{+\xi,-\eta})}}\f]
 *
 * This is the value for \f$w_{\delta}\f$ used by this CoordinateMap when
 * `auto_projective_scale_factor` is `true`.
 *
 * ### Bulged Frustum Coordinate Map and Jacobian
 * Each of the frustum faces in the frustum map given above are flat, but the
 * upper +z face of the frustum can be bulged out by setting a non-zero value
 * for the `sphericity`, where a value of `0.0` corresponds to the usual flat-
 * face, and a value of `1.0` corresponds to a value of fully spherical. Using
 * OrientationMaps allows the user to create a set of frustums that fully cover
 * a spherical surface. The radius of the sphere is determined by the corner of
 * the frustum that is furthest from the origin.
 *
 * The full map is given by:
 * \f[\vec{x}(\xi,\eta,\zeta) = \begin{bmatrix}
 * \Sigma^x + \Delta^x_{\xi}\Xi + (\Delta^x_{\zeta} +
 * \Delta^x_{\xi\zeta}\Xi)\mathrm{Z}\\
 * \Sigma^y + \Delta^y_{\eta}\mathrm{H} + (\Delta^y_{\zeta} +
 * \Delta^y_{\eta\zeta}\mathrm{H})\mathrm{Z}\\
 * \Sigma^z + \Delta^z\mathrm{Z}\\
 * \end{bmatrix}
 * + s \frac{1 +
 * \mathrm{Z}}{2}\left(\frac{R}{|\vec{\sigma}_{\mathrm{+z}}|}-1\right)
 * \vec{\sigma}_{\mathrm{+z}}
 * \f]
 *
 * where \f$R\f$ is the radius, \f$s\f$ is the `sphericity`, and
 * \f$\vec{\sigma}_{\mathrm{+z}}\f$ is given by:
 * \f[
 * \vec{\sigma}_{\mathrm{+z}} =
 * \begin{bmatrix}
 * \Sigma^x + \Delta^x_{\xi}\Xi + \Delta^x_{\zeta} +
 * \Delta^x_{\xi\zeta}\Xi\\
 * \Sigma^y + \Delta^y_{\eta}\mathrm{H} + \Delta^y_{\zeta} +
 * \Delta^y_{\eta\zeta}\mathrm{H}\\
 * \Sigma^z + \Delta^z\\
 * \end{bmatrix}.
 * \f]
 *
 * The Jacobian is:
 * \f[J =
 * \begin{bmatrix}
 * (\Delta^x_{\xi} + \Delta^x_{\xi\zeta}\mathrm{Z})\Xi' &
 * 0 & (\Delta^x_{\zeta}+ \Delta^x_{\xi\zeta}\Xi)\mathrm{Z}' \\
 * 0 & (\Delta^y_{\eta} + \Delta^y_{\eta\zeta}\mathrm{Z})\mathrm{H}' &
 * (\Delta^y_{\zeta} + \Delta^y_{\eta\zeta}\mathrm{H})\mathrm{Z}'\\
 * 0 & 0 & \Delta^z\mathrm{Z}' \\
 * \end{bmatrix}
 * +
 * \frac{s}{2} \left\{ \left(\frac{R}{|\vec{\sigma}_{\mathrm{+z}}|}-1\right)
 * \vec{\sigma}_{\mathrm{+z}}\hat{\zeta}^{\intercal}\mathrm{Z}'+
 * (1+\mathrm{Z})\left(\frac{R}{|\vec{\sigma}_{\mathrm{+z}}|}\left(
 * \mathbb{1}-\frac{\vec{\sigma}_{\mathrm{+z}}
 * \vec{\sigma}_{\mathrm{+z}}^{\intercal}}
 * {|\vec{\sigma}_{\mathrm{+z}}|^2}\right)-\mathbb{1}\right)
 * J_{\sigma}\right\},
 * \f]
 *
 * where \f$\hat{\zeta}\f$ is the row vector \f$[0, 0, 1]\f$, and
 * \f$J_{\sigma}\f$ is the Jacobian of the upper +z surface, given by:
 * \f[
 * J_{\sigma} =
 * \begin{bmatrix}
 * (\Delta^x_{\xi} + \Delta^x_{\xi\zeta})\Xi' &
 * 0 & 0 \\
 * 0 & (\Delta^y_{\eta} + \Delta^y_{\eta\zeta})\mathrm{H}' &
 * 0 \\
 * 0 & 0 & 0 \\
 * \end{bmatrix}.
 * \f]
 */
class Frustum {
 public:
  static constexpr size_t dim = 3;
  Frustum(const std::array<std::array<double, 2>, 4>& face_vertices,
          double lower_bound, double upper_bound,
          OrientationMap<3> orientation_of_frustum,
          bool with_equiangular_map = false,
          double projective_scale_factor = 1.0,
          bool auto_projective_scale_factor = false,
          double sphericity = 0.0);
  Frustum() = default;
  ~Frustum() = default;
  Frustum(Frustum&&) = default;
  Frustum(const Frustum&) = default;
  Frustum& operator=(const Frustum&) = default;
  Frustum& operator=(Frustum&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  /// Returns std::nullopt if \f$z\f$ is at or beyond the \f$z\f$-coordinate of
  /// the apex of the pyramid, tetrahedron, or triangular prism that is
  /// formed by extending the `Frustum` (for a
  /// \f$z\f$-oriented `Frustum`).
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

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  bool is_identity() const { return is_identity_; }

 private:
  friend bool operator==(const Frustum& lhs, const Frustum& rhs);

  OrientationMap<3> orientation_of_frustum_{};
  bool with_equiangular_map_{false};
  bool is_identity_{false};
  bool with_projective_map_{false};
  double sigma_x_{std::numeric_limits<double>::signaling_NaN()};
  double delta_x_zeta_{std::numeric_limits<double>::signaling_NaN()};
  double delta_x_xi_{std::numeric_limits<double>::signaling_NaN()};
  double delta_x_xi_zeta_{std::numeric_limits<double>::signaling_NaN()};
  double sigma_y_{std::numeric_limits<double>::signaling_NaN()};
  double delta_y_zeta_{std::numeric_limits<double>::signaling_NaN()};
  double delta_y_eta_{std::numeric_limits<double>::signaling_NaN()};
  double delta_y_eta_zeta_{std::numeric_limits<double>::signaling_NaN()};
  double sigma_z_{std::numeric_limits<double>::signaling_NaN()};
  double delta_z_zeta_{std::numeric_limits<double>::signaling_NaN()};
  double w_plus_{std::numeric_limits<double>::signaling_NaN()};
  double w_minus_{std::numeric_limits<double>::signaling_NaN()};
  double sphericity_{std::numeric_limits<double>::signaling_NaN()};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const Frustum& lhs, const Frustum& rhs);
}  // namespace CoordinateMaps
}  // namespace domain
