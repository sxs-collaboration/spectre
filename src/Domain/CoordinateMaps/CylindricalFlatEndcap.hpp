// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class CylindricalFlatEndcap.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatEndcap.hpp"
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
 *  a portion of a circle to a portion of a spherical surface.
 *
 * \image html CylindricalFlatEndcap.svg "A cylinder maps to the shaded region."
 *
 * \details Consider a 2D circle in 3D space that is normal to the
 * \f$z\f$ axis and has (3D) center \f$C_1\f$ and radius \f$R_1\f$.
 * Also consider a sphere with center \f$C_2\f$, and radius \f$R_2\f$.
 * Also let there be a projection point \f$P\f$.
 *
 * CylindricalFlatEndcap maps a 3D unit right cylinder (with coordinates
 * \f$(\bar{x},\bar{y},\bar{z})\f$ such that \f$-1\leq\bar{z}\leq 1\f$
 * and \f$\bar{x}^2+\bar{y}^2 \leq 1\f$) to the shaded area
 * in the figure above (with coordinates \f$(x,y,z)\f$).  The "bottom"
 * of the cylinder \f$\bar{z}=-1\f$ is mapped to the interior of the
 * circle of radius \f$R_1\f$.  Curves of constant
 * \f$(\bar{x},\bar{y})\f$ are mapped to portions of lines that pass
 * through \f$P\f$. Along each of these curves, \f$\bar{z}=-1\f$ is
 * mapped to a point on the circle and \f$\bar{z}=+1\f$ is mapped to a
 * point on the sphere.
 *
 * CylindricalFlatEndcap is intended to be composed with Wedge2D maps to
 * construct a portion of a cylindrical domain for a binary system.
 *
 * CylindricalFlatEndcap is described briefly in the Appendix of
 * \cite Buchman:2012dw.
 * CylindricalFlatEndcap is used to construct the blocks labeled 'MA
 * wedge' and 'MB wedge' in Figure 20 of that paper.
 *
 * CylindricalFlatEndcap is implemented using `FocallyLiftedMap`
 * and `FocallyLiftedInnerMaps::FlatEndcap`; see those classes for
 * details.
 *
 * ### Restrictions on map parameters.
 *
 * The following restrictions are made so that the map is not singular
 * or close to singular. It is possible to construct a valid map
 * without these assumptions, but the assumptions simplify the code and
 * avoid problematic edge cases, and the expected use cases obey
 * these restrictions.
 *
 * We demand that
 * - The plane containing the circle is below (i.e. at a smaller value
 *   of \f$z\f$ than) the sphere, by an amount at least 5% of the sphere
 *   radius \f$R_2\f$ but not more than 5 times the sphere radius \f$R_2\f$.
 * - \f$P\f$ is inside the sphere but not too close to its surface;
 *   specifically, we demand that \f$|P-C_2|\leq 0.95 R_2\f$.
 * - The ratio \f$R_1/R_2\f$ is between 10 and 1/10, inclusive.
 * - The x and y components of \f$C_2-C_1\f$ both have magnitudes
 *   smaller than or equal to \f$R_1+R_2\f$.
 *
 */
class CylindricalFlatEndcap {
 public:
  static constexpr size_t dim = 3;
  CylindricalFlatEndcap(const std::array<double, 3>& center_one,
                        const std::array<double, 3>& center_two,
                        const std::array<double, 3>& proj_center,
                        double radius_one, double radius_two);

  CylindricalFlatEndcap() = default;
  ~CylindricalFlatEndcap() = default;
  CylindricalFlatEndcap(CylindricalFlatEndcap&&) = default;
  CylindricalFlatEndcap(const CylindricalFlatEndcap&) = default;
  CylindricalFlatEndcap& operator=(const CylindricalFlatEndcap&) = default;
  CylindricalFlatEndcap& operator=(CylindricalFlatEndcap&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

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

  static bool is_identity() { return false; }

 private:
  friend bool operator==(const CylindricalFlatEndcap& lhs,
                         const CylindricalFlatEndcap& rhs);
  FocallyLiftedMap<FocallyLiftedInnerMaps::FlatEndcap> impl_;
};
bool operator!=(const CylindricalFlatEndcap& lhs,
                const CylindricalFlatEndcap& rhs);

}  // namespace domain::CoordinateMaps
