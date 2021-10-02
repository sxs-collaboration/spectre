// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class CylindricalSide.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedSide.hpp"
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
 * \brief Map from a 3D unit right cylindrical shell to a volume that connects
 *  portions of two spherical surfaces.
 *
 * \image html CylindricalSide.svg "2D slice showing mapped (shaded) region."
 *
 * \details Consider two spheres with centers \f$C_1\f$ and \f$C_2\f$,
 * and radii \f$R_1\f$ and \f$R_2\f$. Let sphere 1 be intersected by two
 * planes normal to the \f$z\f$ axis and located at \f$z = z_\mathrm{L}\f$
 * and \f$z = z_\mathrm{U}\f$, with \f$z_\mathrm{L} < z_\mathrm{U}\f$.
 * Also let there be a projection point \f$P\f$.
 *
 * Note that Sphere 1 and Sphere 2 are not equivalent, because the
 * mapped portion of Sphere 1 is bounded by planes of constant
 * \f$z\f$ but the mapped portion of Sphere 2 is not (except for
 * special choices of \f$C_1\f$, \f$C_2\f$, and \f$P\f$).
 *
 * CylindricalSide maps a 3D unit right cylindrical shell (with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$1 \leq \bar{x}^2+\bar{y}^2 \leq
 * 4\f$) to the shaded area in each panel of the figure above (with
 * coordinates \f$(x,y,z)\f$).  The figure shows two different allowed
 * possibilities: \f$R_1 > R_2\f$ and \f$R_2 > R_1\f$. Note that the
 * two portions of the shaded region in each panel of the figure
 * represent different portions of the same block; each panel of the
 * figure is to be understood as rotated around the \f$z\f$ axis. The
 * inner boundary of the cylindrical shell \f$\bar{x}^2+\bar{y}^2=1\f$
 * is mapped to the portion of sphere 1 that has \f$z_\mathrm{L} \leq
 * z \leq z_\mathrm{U}\f$.  Curves of constant \f$(\bar{z})\f$ along
 * the vector \f$(\bar{x},\bar{y})\f$ are mapped to portions of lines
 * that pass through \f$P\f$. Along each of these curves,
 * \f$\bar{x}^2+\bar{y}^2=1\f$ is mapped to a point on sphere 1 and
 * \f$\bar{x}^2+\bar{y}^2=4\f$ is mapped to a point on sphere 2.
 *
 * CylindricalSide is described briefly in the Appendix of
 * \cite Buchman:2012dw.  CylindricalSide is used to construct the blocks
 * labeled 'CA cylinder', 'EA cylinder', 'CB cylinder', 'EE cylinder',
 * and 'EB cylinder' in Figure 20 of that paper.  Note that 'CA
 * cylinder', 'CB cylinder', and 'EE cylinder' have Sphere 1 contained
 * in Sphere2, and 'EA cylinder' and 'EB cylinder' have Sphere 2
 * contained in Sphere 1.
 *
 * CylindricalSide is implemented using `FocallyLiftedMap`
 * and `FocallyLiftedInnerMaps::Side`; see those classes for
 * details.
 *
 * ### Restrictions on map parameters.
 *
 * We demand that:
 * - Either Sphere 1 is fully contained inside Sphere 2, or
 *   Sphere 2 is fully contained inside Sphere 1.
 * - \f$P\f$ is contained inside the smaller sphere, and
 *   between (or on) the planes defined by \f$z_\mathrm{L}\f$ and
 *   \f$z_\mathrm{U}\f$.
 * - If sphere 1 is contained in sphere 2:
 *   - \f$C_1^z - 0.95 R_1 \leq z_\mathrm{L}\f$
 *   - \f$z_\mathrm{U} \leq C_1^z + 0.95 R_1\f$
 * - If sphere 2 is contained in sphere 1:
 *   - \f$C_1^z - 0.95 R_1 \leq z_\mathrm{L} \leq C_1^z - 0.2 R_1\f$
 *   - \f$C_1^z + 0.2 R_1 \leq z_\mathrm{U} \leq C_1^z + 0.95 R_1\f$
 *
 */
class CylindricalSide {
 public:
  static constexpr size_t dim = 3;
  CylindricalSide(const std::array<double, 3>& center_one,
                  const std::array<double, 3>& center_two,
                  const std::array<double, 3>& proj_center, double radius_one,
                  double radius_two, double z_lower, double z_upper);

  CylindricalSide() = default;
  ~CylindricalSide() = default;
  CylindricalSide(CylindricalSide&&) = default;
  CylindricalSide(const CylindricalSide&) = default;
  CylindricalSide& operator=(const CylindricalSide&) = default;
  CylindricalSide& operator=(CylindricalSide&&) = default;

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
  friend bool operator==(const CylindricalSide& lhs,
                         const CylindricalSide& rhs);
  FocallyLiftedMap<FocallyLiftedInnerMaps::Side> impl_;
};
bool operator!=(const CylindricalSide& lhs, const CylindricalSide& rhs);

}  // namespace domain::CoordinateMaps
