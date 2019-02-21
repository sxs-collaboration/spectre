// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Three dimensional map from the cube to a wedge.
 * \image html Shell.png "A shell can be constructed out of six wedges."
 *
 * \details The mapping that goes from a reference cube to a three-dimensional
 *  wedge centered on a coordinate axis covering a volume between an inner
 *  surface and outer surface. Each surface can be given a curvature
 *  between flat (a sphericity of 0) or spherical (a sphericity of 1).
 *
 *  The first two logical coordinates correspond to the two angular coordinates,
 *  and the third to the radial coordinate.
 *
 *  The Wedge3D map is constructed by linearly interpolating between a bulged
 *  face of radius `radius_of_inner_surface` to a bulged face of
 *  radius `radius_of_outer_surface`, where the radius of each bulged face
 *  is defined to be the radius of the sphere circumscribing the bulge.
 *
 *  We make a choice here as to whether we wish to use the logical coordinates
 *  parameterizing these surface as they are, in which case we have the
 *  equidistant choice of coordinates, or whether to apply a tangent map to them
 *  which leads us to the equiangular choice of coordinates. In terms of the
 *  logical coordinates, the equiangular coordinates are:
 *
 *  \f[\textrm{equiangular xi} : \Xi(\xi) = \textrm{tan}(\xi\pi/4)\f]
 *
 *  \f[\textrm{equiangular eta}  : \mathrm{H}(\eta) = \textrm{tan}(\eta\pi/4)\f]
 *
 *  With derivatives:
 *
 *  \f[\Xi'(\xi) = \frac{\pi}{4}(1+\Xi^2)\f]
 *
 *  \f[\mathrm{H}'(\eta) = \frac{\pi}{4}(1+\mathrm{H}^2)\f]
 *
 *  The equidistant coordinates are:
 *
 *  \f[ \textrm{equidistant xi}  : \Xi = \xi\f]
 *
 *  \f[ \textrm{equidistant eta}  : \mathrm{H} = \eta\f]
 *
 *  with derivatives:
 *
 *  <center>\f$\Xi'(\xi) = 1\f$, and \f$\mathrm{H}'(\eta) = 1\f$</center>
 *
 *  We also define the variable \f$\rho\f$, given by:
 *
 *  \f[\textrm{rho} : \rho = \sqrt{1+\Xi^2+\mathrm{H}^2}\f]
 *
 *  ### The Spherical Face Map
 *  The surface map for the spherical face of radius \f$R\f$ lying in the
 * \f$+z\f$
 *  direction in either choice of coordinates is then given by:
 *
 *  \f[\vec{\sigma}_{spherical}: \vec{\xi} \rightarrow \vec{x}(\vec{\xi})\f]
 *  Where
 *  \f[
 *  \vec{x}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  z(\xi,\eta)\\
 *  \end{bmatrix}  = \frac{R}{\rho}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  ### The Bulged Face Map
 *  The bulged surface is itself constructed by linearly interpolating between
 *  a cubical face and a spherical face. The surface map for the cubical face
 *  of side length \f$2L\f$ lying in the \f$+z\f$ direction is given by:
 *
 *  \f[\vec{\sigma}_{cubical}: \vec{\xi} \rightarrow \vec{x}(\vec{\xi})\f]
 *  Where
 *  \f[
 *  \vec{x}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  L\\
 *  \end{bmatrix}  = L
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  To construct the bulged map we interpolate between this cubical face map
 *  and a spherical face map of radius \f$R\f$, with the
 *  interpolation parameter being \f$s\f$. The surface map for the bulged face
 *  lying in the \f$+z\f$ direction is then given by:
 *
 *  \f[\vec{\sigma}_{bulged}(\xi,\eta) = {(1-s)L + \frac{sR}{\rho}}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  We constrain L by demanding that the spherical face circumscribe the cube.
 *  With this condition, we have \f$L = R/\sqrt3\f$.
 *  \note This differs from the choice in SpEC where it is demanded that the
 *  surfaces touch at the center, which leads to \f$L = R\f$.
 *
 *  ### The Full Volume Map
 *  The final map for the wedge which lies along the \f$+z\f$ is obtained by
 *  interpolating between the two surfaces with the
 *  interpolation parameter being the logical coordinate \f$\zeta\f$. This
 *  results in:
 *
 *  \f[\vec{x}(\xi,\eta,\zeta) =
 *  \frac{1}{2}\left\{(1-\zeta)\Big[(1-s_{inner})\frac{R_{inner}}{\sqrt 3}
 *   + s_{inner}\frac{R_{inner}}{\rho}\Big] +
 *  (1+\zeta)\Big[(1-s_{outer})\frac{R_{outer}}{\sqrt 3} +s_{outer}
 *  \frac{R_{outer}}{\rho}\Big] \right\}\begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  We will define the variables \f$F(\zeta)\f$ and \f$S(\zeta)\f$, the frustum
 * and sphere factors: \f[F(\zeta) = F_0 + F_1\zeta\f] \f[S(\zeta) = S_0 +
 * S_1\zeta\f]
 *  Where \f{align*}F_0 &= \frac{1}{2} \big\{ (1-s_{outer})R_{outer} +
 * (1-s_{inner})R_{inner}\big\}\\
 *  F_1 &= \partial_{\zeta} F = \frac{1}{2} \big\{ (1-s_{outer})R_{outer} -
 * (1-s_{inner})R_{inner}\big\}\\
 *  S_0 &= \frac{1}{2} \big\{ s_{outer}R_{outer} + s_{inner}R_{inner}\big\}\\
 *  S_1 &= \partial_{\zeta} S = \frac{1}{2} \big\{ s_{outer}R_{outer} -
 * s_{inner}R_{inner}\big\}\f}
 *
 *  The map can then be rewritten as:
 * \f[\vec{x}(\xi,\eta,\zeta) = \left\{\frac{F(\zeta)}{\sqrt 3} +
 * \frac{S(\zeta)}{\rho}\right\}\begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  We provide some common derivatives:
 *  \f[\partial_{\xi}z = \frac{-S(\zeta)\Xi\Xi'}{\rho^3}\f]
 *  \f[\partial_{\eta}z = \frac{-S(\zeta)\mathrm{H}\mathrm{H}'}{\rho^3}\f]
 * \f[\partial_{\zeta}z = \frac{F'}{\sqrt 3} + \frac{S'}{\rho}\f]
 *  The Jacobian then is: \f[J =
 *  \begin{bmatrix}
 *  \Xi'z + \Xi\partial_{\xi}z & \Xi\partial_{\eta}z & \Xi\partial_{\zeta}z \\
 *  \mathrm{H}\partial_{\xi}z & \mathrm{H}'z +
 *  \mathrm{H}\partial_{\eta}z & \mathrm{H}\partial_{\zeta}z\\
 *   \partial_{\xi}z&\partial_{\eta}z &\partial_{\zeta}z \\
 *  \end{bmatrix}
 *  \f]
 *
 *  A common factor that shows up in the inverse jacobian is:
 *  \f[ T:= \frac{S(\zeta)}{(\partial_{\zeta}z)\rho^3}\f]
 *
 *  The inverse Jacobian then is: \f[J^{-1} =
 *  \frac{1}{z}\begin{bmatrix}
 *  \Xi'^{-1} & 0 & -\Xi\Xi'^{-1}\\
 *  0 & \mathrm{H}'^{-1} & -\mathrm{H}\mathrm{H}'^{-1}\\
 *  T\Xi &
 *  T\mathrm{H} &
 *  T + F(\partial_{\zeta}z)^{-1}/\sqrt 3\\
 *  \end{bmatrix}
 *  \f]
 *
 *  ### Changing the radial distribution of the gridpoints
 *  By default, Wedge3D linearly distributes its gridpoints in the radial
 *  direction. An exponential distribution of gridpoints can be obtained by
 *  linearly interpolating in the logarithm of the radius, in order to obtain
 *  a relatively higher resolution at smaller radii. Since this is a radial
 *  rescaling of Wedge3D, this option is only supported for fully spherical
 *  wedges with `sphericity_inner` = `sphericity_outer` = 1.
 *
 *  The linear interpolation done is:
 *  \f[
 *  \log r = \frac{1-\zeta}{2}\log R_{inner} +
 *  \frac{1+\zeta}{2}\log R_{outer}
 *  \f]
 *
 *  The map then is:
 *  \f[\vec{x}(\xi,\eta,\zeta) =
 *  \frac{\sqrt{R_{inner}^{1-\zeta}R_{outer}^{1+\zeta}}}{\rho}\begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  The jacobian simplifies similarly.
 *
 */
class Wedge3D {
 public:
  static constexpr size_t dim = 3;
  enum class WedgeHalves {
    /// Use the entire wedge
    Both,
    /// Use only the upper logical half
    UpperOnly,
    /// Use only the lower logical half
    LowerOnly
  };

  /*!
   * Constructs a 3D wedge.
   * \param radius_inner Distance from the origin to one of the
   * corners which lie on the inner surface.
   * \param radius_outer Distance from the origin to one of the
   * corners which lie on the outer surface.
   * \param orientation_of_wedge The orientation of the desired wedge relative
   * to the orientation of the default wedge which is a wedge that has its
   * curved surfaces pierced by the upper-z axis. The logical xi and eta
   * coordinates point in the cartesian x and y directions, respectively.
   * \param sphericity_inner Value between 0 and 1 which determines
   * whether the inner surface is flat (value of 0), spherical (value of 1) or
   * somewhere in between
   * \param sphericity_outer Value between 0 and 1 which determines
   * whether the outer surface is flat (value of 0), spherical (value of 1) or
   * somewhere in between
   * \param with_equiangular_map Determines whether to apply a tangent function
   * mapping to the logical coordinates (for `true`) or not (for `false`).
   * \param halves_to_use Determines whether to construct a full wedge or only
   * half a wedge. If constructing only half a wedge, the resulting shape has a
   * face normal to the x direction (assuming default OrientationMap). If
   * constructing half a wedge, an intermediate affine map is applied to the
   * logical xi coordinate such that the interval [-1,1] is mapped to the
   * corresponding logical half of the wedge. For example, if `UpperOnly` is
   * specified, [-1,1] is mapped to [0,1], and if `LowerOnly` is specified,
   * [-1,1] is mapped to [-1,0]. The case of `Both` means a full wedge, with no
   * intermediate map applied. In all cases, the logical points returned by the
   * inverse map will lie in the range [-1,1] in each dimension. Half wedges are
   * currently only useful in constructing domains for binary systems.
   * \param with_logarithmic_map Determines whether to apply an exponential
   * function mapping to the "sphere factor", the effect of which is to
   * distribute the radial gridpoints logarithmically in physical space.
   */
  Wedge3D(double radius_inner, double radius_outer,
          OrientationMap<3> orientation_of_wedge, double sphericity_inner,
          double sphericity_outer, bool with_equiangular_map,
          WedgeHalves halves_to_use = WedgeHalves::Both,
          bool with_logarithmic_map = false) noexcept;

  Wedge3D() = default;
  ~Wedge3D() = default;
  Wedge3D(Wedge3D&&) = default;
  Wedge3D(const Wedge3D&) = default;
  Wedge3D& operator=(const Wedge3D&) = default;
  Wedge3D& operator=(Wedge3D&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  /// For a \f$+z\f$-oriented `Wedge3D`, returns invalid if \f$z<=0\f$
  /// or if \f$(x,y,z)\f$ is on or outside the cone defined
  /// by \f$(x^2/z^2 + y^2/z^2+1)^{1/2} = -S/F\f$, where
  /// \f$S = \frac{1}{2}(s_1 r_1 - s_0 r_0)\f$ and
  /// \f$F = \frac{1}{2\sqrt{3}}((1-s_1) r_1 - (1-s_0) r_0)\f$.
  /// Here \f$s_0,s_1\f$ and \f$r_0,r_1\f$ are the specified sphericities
  /// and radii of the inner and outer \f$z\f$ surfaces.  The map is singular on
  /// the cone and on the xy plane.
  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  // factors out calculation of z needed for mapping and jacobian
  template <typename T>
  tt::remove_cvref_wrap_t<T> default_physical_z(const T& zeta,
                                                const T& one_over_rho) const
      noexcept;
  friend bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept;

  double radius_inner_{std::numeric_limits<double>::signaling_NaN()};
  double radius_outer_{std::numeric_limits<double>::signaling_NaN()};
  OrientationMap<3> orientation_of_wedge_{};
  double sphericity_inner_{std::numeric_limits<double>::signaling_NaN()};
  double sphericity_outer_{std::numeric_limits<double>::signaling_NaN()};
  bool with_equiangular_map_ = false;
  WedgeHalves halves_to_use_ = WedgeHalves::Both;
  bool with_logarithmic_map_ = false;
  double scaled_frustum_zero_{std::numeric_limits<double>::signaling_NaN()};
  double sphere_zero_{std::numeric_limits<double>::signaling_NaN()};
  double scaled_frustum_rate_{std::numeric_limits<double>::signaling_NaN()};
  double sphere_rate_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept;
}  // namespace CoordinateMaps
}  // namespace domain
