// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class CylindricalEndcap.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
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
 * \brief Map from 3D unit right cylinder to a volume that connects
 *  portions of two spherical surfaces.
 *
 * \image html CylindricalEndcap.svg "A cylinder maps to the shaded region."
 *
 * \details Consider two spheres with centers \f$C_1\f$ and \f$C_2\f$,
 * and radii \f$R_1\f$ and \f$R_2\f$. Let sphere 1 be intersected by a
 * plane normal to the \f$z\f$ axis and located at \f$z = z_\mathrm{P}\f$.
 * Also let there be a projection point \f$P\f$.
 *
 * CylindricalEndcap maps a 3D unit right cylinder (with coordinates
 * \f$(\bar{x},\bar{y},\bar{z})\f$ such that \f$-1\leq\bar{z}\leq 1\f$
 * and \f$\bar{x}^2+\bar{y}^2 \leq 1\f$) to the shaded area
 * in the figure above (with coordinates \f$(x,y,z)\f$).  The "bottom"
 * of the cylinder \f$\bar{z}=-1\f$ is mapped to the portion of sphere
 * 1 that has \f$z \geq z_\mathrm{P}\f$.  Curves of constant
 * \f$(\bar{x},\bar{y})\f$ are mapped to portions of lines that pass
 * through \f$P\f$. Along each of these curves, \f$\bar{z}=-1\f$ is
 * mapped to a point on sphere 1 and \f$\bar{z}=+1\f$ is mapped to a
 * point on sphere 2.
 *
 * CylindricalEndcap is intended to be composed with Wedge2D maps to
 * construct a portion of a cylindrical domain for a binary system.
 *
 * CylindricalEndcap is described briefly in the Appendix of
 * \cite Buchman:2012dw.
 * CylindricalEndcap is used to construct the blocks labeled 'CA
 * wedge', 'EA wedge', 'CB wedge', 'EE wedge', and 'EB wedge' in
 * Figure 20 of that paper.
 *
 * CylindricalEndcap is implemented using `FocallyLiftedMap`
 * and `FocallyLiftedInnerMaps::Endcap`; see those classes for details.
 *
 * ### Restrictions on map parameters.
 *
 * We demand that Sphere 1 is fully contained inside Sphere 2. It is
 * possible to construct a valid map without this assumption, but the
 * assumption simplifies the code, and the expected use cases obey
 * this restriction.
 *
 * We also demand that \f$z_\mathrm{P} > C_1^2\f$, that is, the plane
 * in the above and below figures lies to the right of \f$C_1^2\f$.
 * This restriction not strictly necessary but is made for simplicity.
 *
 * The map is invertible only for some choices of the projection point
 * \f$P\f$.  Given the above restrictions, the allowed values of
 * \f$P\f$ are illustrated by the following diagram:
 *
 * \image html CylindricalEndcap_Allowed.svg "Allowed region for P." width=75%
 *
 * The plane \f$z=z_\mathrm{P}\f$ intersects sphere 1 on a circle. The
 * cone with apex \f$C_1\f$ that intersects that circle has opening
 * angle \f$2\theta\f$ as shown in the above figure. Construct another
 * cone, the "invertibility cone", with apex \f$S\f$ chosen such that
 * the two cones intersect at right angles on the circle; thus the
 * opening angle of the invertibility cone is \f$\pi-2\theta\f$.  A
 * necessary condition for invertibility is that the projection point
 * \f$P\f$ lies inside the invertibility cone, but not between \f$S\f$
 * and sphere 1.  (If \f$P\f$ does not obey this condition, then from the
 * diagram one can find at least one line through \f$P\f$
 * that twice intersects the surface of sphere 1 with \f$z>z_\mathrm{P}\f$;
 * the inverse map is thus double-valued at those intersection points.)
 * Placing the projection point \f$P\f$ to the
 * right of \f$S\f$ (but inside the invertibility cone) is ok for
 * invertibility.
 *
 * In addition to invertibility and the two additional restrictions
 * already mentioned above, we demand a few more restrictions on the
 * map parameters to simplify the logic for the expected use cases and
 * to ensure that jacobians do not get too large. We demand:
 *
 * - \f$P\f$ is not too close to the edge of the invertibility cone.
 *   Here we demand that the angle between \f$P\f$ and \f$S\f$
 *   is less than \f$0.95 (\pi/2-\theta)\f$ (note that if this
 *   angle is exactly \f$\pi/2-\theta\f$ it is exactly on the
 *   invertibility cone); The 0.95 was chosen empirically based on
 *   unit tests.
 * - \f$P\f$ is contained in sphere 2.
 * - \f$P\f$ is to the left of \f$z_\mathrm{P}\f$.
 * - \f$z_\mathrm{P}\f$ is not too close to the center or the edge of sphere 1.
 *   Here we demand that \f$0.1 \leq \cos(\theta) \leq 0.9\f$, where
 *   the values 0.1 and 0.9 were chosen empirically based on unit tests.
 * - If a line segment is drawn between \f$P\f$ and any point on the
 *   intersection circle (the circle where sphere 1 intersects the
 *   plane \f$z=z_\mathrm{P}\f$), the angle between the line segment and
 *   the z-axis is smaller than \f$\pi/3\f$.
 *
 */
class CylindricalEndcap {
 public:
  static constexpr size_t dim = 3;
  CylindricalEndcap(const std::array<double, 3>& center_one,
                       const std::array<double, 3>& center_two,
                       const std::array<double, 3>& proj_center,
                       double radius_one, double radius_two,
                       double z_plane) noexcept;

  CylindricalEndcap() = default;
  ~CylindricalEndcap() = default;
  CylindricalEndcap(CylindricalEndcap&&) = default;
  CylindricalEndcap(const CylindricalEndcap&) = default;
  CylindricalEndcap& operator=(const CylindricalEndcap&) = default;
  CylindricalEndcap& operator=(CylindricalEndcap&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

 private:
  friend bool operator==(const CylindricalEndcap& lhs,
                         const CylindricalEndcap& rhs) noexcept;
  FocallyLiftedMap<FocallyLiftedInnerMaps::Endcap> impl_;
};
bool operator!=(const CylindricalEndcap& lhs,
                const CylindricalEndcap& rhs) noexcept;

}  // namespace domain::CoordinateMaps
