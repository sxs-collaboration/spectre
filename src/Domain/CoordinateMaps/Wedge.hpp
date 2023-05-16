// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

namespace detail {
// This mapping can be deleted once the 2D and 3D wedges are oriented the same
// (see issue https://github.com/sxs-collaboration/spectre/issues/2988)
template <size_t Dim>
struct WedgeCoordOrientation;
template <>
struct WedgeCoordOrientation<2> {
  static constexpr size_t radial_coord = 0;
  static constexpr size_t polar_coord = 1;
  static constexpr size_t azimuth_coord = 2;  // unused
};
template <>
struct WedgeCoordOrientation<3> {
  static constexpr size_t radial_coord = 2;
  static constexpr size_t polar_coord = 0;
  static constexpr size_t azimuth_coord = 1;
};
}  // namespace detail

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Map from a square or cube to a wedge.
 * \image html Shell.png "A shell can be constructed out of six wedges."
 *
 * \details The mapping that goes from a reference cube (in 3D) or square (in
 *  2D) to a wedge centered on a coordinate axis covering a volume between an
 *  inner surface and outer surface. Each surface can be given a curvature
 *  between flat (a sphericity of 0) or spherical (a sphericity of 1).
 *
 *  In 2D, the first logical coordinate corresponds to the radial coordinate,
 *  and the second logical coordinates correspond to the angular coordinate. In
 *  3D, the first two logical coordinates correspond to the two angular
 *  coordinates, and the third to the radial coordinate. This difference
 *  originates from separate implementations for the 2D and 3D map that were
 *  merged. The 3D implementation can be changed to use the first logical
 *  coordinate as radial direction, but this requires propagating the change
 *  through the rest of the domain code (see issue
 *  https://github.com/sxs-collaboration/spectre/issues/2988).
 *
 *  The following documentation is for the 3D map. The 2D map is obtained by
 *  setting either of the two angular coordinates to zero (and using \f$\xi\f$
 *  as radial coordinate).
 *
 *  The Wedge map is constructed by linearly interpolating between a bulged
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
 *  By default, Wedge linearly distributes its gridpoints in the radial
 *  direction. An exponential distribution of gridpoints can be obtained by
 *  linearly interpolating in the logarithm of the radius, in order to obtain
 *  a relatively higher resolution at smaller radii. Since this is a radial
 *  rescaling of Wedge, this option is only supported for fully spherical
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
 *  Alternatively, an inverse radial distribution can be chosen where the linear
 *  interpolation is:
 *
 *  \f[
 *  \frac{1}{r} = \frac{R_\mathrm{inner} + R_\mathrm{outer}}{2 R_\mathrm{inner}
 *  R_\mathrm{outer}} + \frac{R_\mathrm{inner} - R_\mathrm{outer}}{2
 *  R_\mathrm{inner} R_\mathrm{outer}} \zeta
 *  \f]
 */
template <size_t Dim>
class Wedge {
 public:
  static constexpr size_t dim = Dim;
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
   * \param radial_distribution Determines how to distribute grid points along
   * the radial direction. For wedges that are not exactly spherical, only
   * `Distribution::Linear` is currently supported.
   * \param opening_angles Determines the angular size of the wedge. The
   * default value is pi/2, which corresponds to a wedge size of pi/2. For this
   * setting, four Wedges can be put together to cover 2pi in angle along a
   * great circle. This option is meant to be used with the equiangular map
   * option turned on.
   * \param with_adapted_equiangular_map Determines whether to adapt the
   * point distribution in the wedge to match its physical angular size. When
   * `true`, angular distances are proportional to logical distances. Note
   * that it is not possible to use adapted maps in every Wedge of a Sphere
   * unless each Wedge has the same size along both angular directions.
   */
  Wedge(double radius_inner, double radius_outer, double sphericity_inner,
        double sphericity_outer, OrientationMap<Dim> orientation_of_wedge,
        bool with_equiangular_map,
        WedgeHalves halves_to_use = WedgeHalves::Both,
        Distribution radial_distribution = Distribution::Linear,
        const std::array<double, Dim - 1>& opening_angles =
            make_array<Dim - 1>(M_PI_2),
        bool with_adapted_equiangular_map = true);

  Wedge() = default;
  ~Wedge() = default;
  Wedge(Wedge&&) = default;
  Wedge(const Wedge&) = default;
  Wedge& operator=(const Wedge&) = default;
  Wedge& operator=(Wedge&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords) const;

  /// For a \f$+z\f$-oriented `Wedge`, returns invalid if \f$z<=0\f$
  /// or if \f$(x,y,z)\f$ is on or outside the cone defined
  /// by \f$(x^2/z^2 + y^2/z^2+1)^{1/2} = -S/F\f$, where
  /// \f$S = \frac{1}{2}(s_1 r_1 - s_0 r_0)\f$ and
  /// \f$F = \frac{1}{2\sqrt{3}}((1-s_1) r_1 - (1-s_0) r_0)\f$.
  /// Here \f$s_0,s_1\f$ and \f$r_0,r_1\f$ are the specified sphericities
  /// and radii of the inner and outer \f$z\f$ surfaces.  The map is singular on
  /// the cone and on the xy plane.
  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, Dim>> inverse(
      const std::array<double, Dim>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static constexpr bool is_identity() { return false; }

 private:
  // maps between 2D and 3D choices for coordinate axis orientations
  static constexpr size_t radial_coord =
      detail::WedgeCoordOrientation<Dim>::radial_coord;
  static constexpr size_t polar_coord =
      detail::WedgeCoordOrientation<Dim>::polar_coord;
  static constexpr size_t azimuth_coord =
      detail::WedgeCoordOrientation<Dim>::azimuth_coord;

  // factors out calculation of z needed for mapping and jacobian
  template <typename T>
  tt::remove_cvref_wrap_t<T> default_physical_z(const T& zeta,
                                                const T& one_over_rho) const;

  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Wedge<LocalDim>& lhs,
                         const Wedge<LocalDim>& rhs);

  double radius_inner_{std::numeric_limits<double>::signaling_NaN()};
  double radius_outer_{std::numeric_limits<double>::signaling_NaN()};
  double sphericity_inner_{std::numeric_limits<double>::signaling_NaN()};
  double sphericity_outer_{std::numeric_limits<double>::signaling_NaN()};
  OrientationMap<Dim> orientation_of_wedge_{};
  bool with_equiangular_map_ = false;
  WedgeHalves halves_to_use_ = WedgeHalves::Both;
  Distribution radial_distribution_ = Distribution::Linear;
  double scaled_frustum_zero_{std::numeric_limits<double>::signaling_NaN()};
  double sphere_zero_{std::numeric_limits<double>::signaling_NaN()};
  double scaled_frustum_rate_{std::numeric_limits<double>::signaling_NaN()};
  double sphere_rate_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, Dim - 1> opening_angles_{
      make_array<Dim - 1>(std::numeric_limits<double>::signaling_NaN())};
  std::array<double, Dim - 1> opening_angles_distribution_{
      make_array<Dim - 1>(std::numeric_limits<double>::signaling_NaN())};
};

template <size_t Dim>
bool operator!=(const Wedge<Dim>& lhs, const Wedge<Dim>& rhs);
}  // namespace domain::CoordinateMaps
