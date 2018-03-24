// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Three dimensional map from the cube to a wedge.
 * \image html Shell.png "A shell can be constructed out of six wedges."
 *
 * \details The mapping that goes from a reference cube to a three-dimensional
 *  wedge centered on a coordinate axis covering a volume between an inner
 *  surface and outer surface. One of the surfaces must be spherical, but the
 *  curvature of the other surface can be anything between flat (a sphericity of
 *  0) and spherical (a sphericity of 1).
 *
 *  The first two logical coordinates correspond to the two angular coordinates,
 *  and the third to the radial coordinate.
 *
 *  The Wedge3D map is constructed by linearly interpolating between a bulged
 *  face of radius `radius_of_other_surface` to a spherical face of
 *  radius `radius_of_spherical_surface`, where the radius of the bulged face
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
 *  \f[\sigma_{spherical}: \vec{\xi} \rightarrow \vec{x}(\vec{\xi})\f]
 *  Where
 *  \f[
 *  \vec{x}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  z(\xi,\eta)\\
 *  \end{bmatrix}  = R
 *  \begin{bmatrix}
 *  \Xi/\rho\\
 *  \mathrm{H}/\rho\\
 *  1/\rho\\
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
 *  and a spherical face map of radius `radius_of_other_surface`, with the
 *  interpolation parameter being \f$s\f$, the `sphericity_of_other_surface`.
 *  The surface map for the bulged face lying in the \f$+z\f$ direction is then
 *  given by:
 *
 *  \f[\vec{\sigma}_{other}(\xi,\eta) = {(1-s)L + \frac{sR_{other}}{\rho}}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  We constrain L by demanding that the spherical face circumscribe the cube.
 *  With this condition, we have \f$L = R_{other}/\sqrt3\f$.
 *
 *  ### The Full Volume Map
 *  The final map for the wedge which lies along the \f$+z\f$ is obtained by
 *  interpolating between the spherical and bulged surfaces with the
 *  interpolation parameter being the logical coordinate \f$\zeta\f$. This
 *  results in:
 *
 *  \f[\vec{x}(\xi,\eta,\zeta) = \frac{1}{2}\big\{(1-\zeta)(1-s)L
 *   + (1-\zeta)s\frac{R_{other}}{\rho} +
 *  (1+\zeta)\frac{R_{spherical}}{\rho}\big\}\begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}\f]
 *
 *  We will define the variables \f$b_{f1}\f$ and \f$b_{f2}\f$, the first and
 *  second *blending factors*:
 *
 *  \f[b_{f1} = \frac{1}{2}\big\{(1-\zeta)(1-s)L\big\}\f]
 *
 *  \f[b_{f2} = \frac{1}{2}\big\{(1-\zeta)sR_{other} + (1+\zeta)R_{spherical}
 *  \big\}\f]
 *
 *  We also define their zeta-derivatives, the *blending rates*:
 *
 *  \f[b_{r1} = \frac{\mathrm{d}b_{f1}}{\mathrm{d}\zeta} =
 *  \frac{1}{2}(s-1)L\f]
 *
 *  \f[b_{r2} = \frac{\mathrm{d}b_{f2}}{\mathrm{d}\zeta} =
 *  \frac{1}{2}\big\{-sR_{other} + R_{spherical}\big\}\f]
 *
 *  The Jacobian then is: \f[J =
 *  \begin{bmatrix}
 *  b_{f1}\Xi' & 0 &b_{r1}\Xi\\
 *  0 & b_{f1}\mathrm{H}' &b_{r1}\mathrm{H}\\
 *  0 & 0 &b_{r1}\\
 *  \end{bmatrix} + \begin{bmatrix}
 *  b_{f2}\Xi'(1+\mathrm{H}^2)/{\rho^3} & -b_{f2}\mathrm{H}'\Xi\mathrm{H}/\rho^3
 * &
 *  b_{r2}\Xi/\rho\\
 *  -b_{f2}\Xi'\Xi\mathrm{H}/\rho^3 & b_{f2}\mathrm{H}'(1+\Xi^2)/\rho^3 &
 *  b_{r2}\mathrm{H}/\rho\\
 *  -b_{f2}\Xi'\Xi/\rho^3 & -b_{f2}\mathrm{H}'\mathrm{H}/\rho^3 & b_{r2}/\rho\\
 *  \end{bmatrix}
 *  \f]
 *
 *  \note This differs from the choice in SpEC where it is demanded that the
 *  surfaces touch at the center, which leads to \f$L = R_{other}\f$.
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
   * \param radius_of_spherical_surface Radius of the spherical surface
   * \param radius_of_other_surface Distance from the origin to one of the
   * corners which lie on the other surface, which may be anything between flat
   * and spherical.
   * \param orientation_of_wedge The orientation of the desired wedge relative
   * to the orientation of the default wedge which is a wedge that has its
   * curved surfaces pierced by the upper-z axis. The logical xi and eta
   * coordinates point in the cartesian x and y directions, respectively.
   * \param sphericity_of_other_surface Value between 0 and 1 which determines
   * whether the other surface is flat (value of 0), spherical (value of 1) or
   * somewhere in between
   * \param with_equiangular_map Determines whether to apply a tangent function
   * mapping to the logical coordinates (for `true`) or not (for `false`).
   * \param halves_to_use Determines whether to use the logical xi
   * coordinates in the [0,1] interval (value of `UpperOnly`) of the full wedge,
   * the coordinates in the [-1,0] interval (value of `LowerOnly`) of the
   * full wedge, or the full wedge entirely (value of `Both`). Half wedges are
   * currently only useful in constructing domains for binary systems.
   */
  Wedge3D(double radius_of_other_surface, double radius_of_spherical_surface,
          OrientationMap<3> orientation_of_wedge,
          double sphericity_of_other_surface, bool with_equiangular_map,
          WedgeHalves halves_to_use = WedgeHalves::Both) noexcept;

  Wedge3D() = default;
  ~Wedge3D() = default;
  Wedge3D(Wedge3D&&) = default;
  Wedge3D(const Wedge3D&) = default;
  Wedge3D& operator=(const Wedge3D&) = default;
  Wedge3D& operator=(Wedge3D&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> inverse(
      const std::array<T, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept;

  double radius_of_other_surface_{std::numeric_limits<double>::signaling_NaN()};
  double radius_of_spherical_surface_{
      std::numeric_limits<double>::signaling_NaN()};
  OrientationMap<3> orientation_of_wedge_{};
  double sphericity_of_other_surface_{
      std::numeric_limits<double>::signaling_NaN()};
  bool with_equiangular_map_ = false;
  WedgeHalves halves_to_use_ = WedgeHalves::Both;
};
bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept;
}  // namespace CoordinateMaps
