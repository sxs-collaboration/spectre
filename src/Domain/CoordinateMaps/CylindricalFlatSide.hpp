// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class CylindricalFlatSide.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatSide.hpp"
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
 * \brief Map from 3D unit right cylindrical shell to a volume that connects
 *  a portion of an annulus to a portion of a spherical surface.
 *
 * \image html CylindricalFlatSide.svg "A cylinder maps to the shaded region."
 *
 * \details Consider a 2D annulus in 3D space that is normal to the
 * \f$z\f$ axis and has (3D) center \f$C_1\f$, inner radius
 * \f$R_\mathrm{in}\f$ and outer radius \f$R_\mathrm{out}\f$
 * Also consider a sphere with center \f$C_2\f$, and radius \f$R_2\f$.
 * Also let there be a projection point \f$P\f$.
 *
 * CylindricalFlatSide maps a 3D unit right cylindrical shell (with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$1 \leq \bar{x}^2+\bar{y}^2 \leq
 * 4\f$) to the shaded area in the figure above (with coordinates
 * \f$(x,y,z)\f$).  The "bottom" of the cylinder \f$\bar{z}=-1\f$ is
 * mapped to the interior of the annulus with radii
 * \f$R_\mathrm{in}\f$ and \f$R_\mathrm{out}\f$.  Curves of constant
 * \f$(\bar{x},\bar{y})\f$ are mapped to portions of lines that pass
 * through \f$P\f$. Along each of these curves, \f$\bar{z}=-1\f$ is
 * mapped to a point inside the annulus and \f$\bar{z}=+1\f$ is mapped to a
 * point on the sphere.
 *
 * CylindricalFlatSide is intended to be composed with Wedge2D maps to
 * construct a portion of a cylindrical domain for a binary system.
 *
 * CylindricalFlatSide is described briefly in the Appendix of
 * \cite Buchman:2012dw.
 * CylindricalFlatSide is used to construct the blocks labeled 'ME
 * cylinder' in Figure 20 of that paper.
 *
 * CylindricalFlatSide is implemented using `FocallyLiftedMap`
 * and `FocallyLiftedInnerMaps::FlatSide`; see those classes for
 * details.
 *
 * ### Restrictions on map parameters.
 *
 * We demand that:
 * - The sphere is at a larger value of \f$z\f$ (plus 5
 *   percent of the sphere radius) than the plane containing the
 *   annulus.
 * - The projection point \f$z_\mathrm{P}\f$ is
 *   inside the sphere and more than 15 percent away from the boundary
 *   of the sphere.
 * - The center of the annulus is contained in the circle that results from
 *   projecting the sphere into the \f$xy\f$ plane.
 * - The outer radius of the annulus is larger than 5 percent of the distance
 *   between the center of the annulus and the projection point.
 * - The inner radius of the annulus is less than 95 percent of the outer
 *   radius, larger than 5 percent of the outer radius, and larger than one
 *   percent of the distance between the center of the annulus and the
 *   projection point.  The last condition means that the angle subtended by
 *   the inner radius with respect to the projection point is not too small.
 *
 * It is possible to construct a valid map without these assumptions,
 * but some of these assumptions simplify the code and others eliminate
 * edge cases where Jacobians become large or small.
 *
 */
class CylindricalFlatSide {
 public:
  static constexpr size_t dim = 3;
  CylindricalFlatSide(const std::array<double, 3>& center_one,
                      const std::array<double, 3>& center_two,
                      const std::array<double, 3>& proj_center,
                      const double inner_radius, const double outer_radius,
                      const double radius_two);

  CylindricalFlatSide() = default;
  ~CylindricalFlatSide() = default;
  CylindricalFlatSide(CylindricalFlatSide&&) = default;
  CylindricalFlatSide(const CylindricalFlatSide&) = default;
  CylindricalFlatSide& operator=(const CylindricalFlatSide&) = default;
  CylindricalFlatSide& operator=(CylindricalFlatSide&&) = default;

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

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  static bool is_identity() { return false; }

 private:
  friend bool operator==(const CylindricalFlatSide& lhs,
                         const CylindricalFlatSide& rhs);
  FocallyLiftedMap<FocallyLiftedInnerMaps::FlatSide> impl_;
};
bool operator!=(const CylindricalFlatSide& lhs, const CylindricalFlatSide& rhs);

}  // namespace domain::CoordinateMaps
