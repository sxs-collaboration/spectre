// Distributed under the MIT License.
// See LICENSE.txt for details.

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
 * \brief Map from a 3D unit right cylinder to a volume that connects
 *  a flat circular disk (lying inside a sphere) to the far wall of
 *  that sphere.
 *
 * \details This is the "interior" counterpart to `CylindricalFlatEndcap`.
 * The two maps are identical in structure — both use `FocallyLiftedMap`
 * with `FocallyLiftedInnerMaps::FlatEndcap` — but differ in which
 * intersection of the projecting ray with the sphere is chosen:
 *
 * - `CylindricalFlatEndcap`: the flat disk lies *outside* the sphere
 *   (the sphere is between \f$P\f$ and the flat disk), so
 *   `source_is_between_focus_and_target = false`.
 * - `CylindricalFlatEndcapInterior`: the flat disk lies *inside* the
 *   sphere (the flat disk is between \f$P\f$ and the sphere's far wall),
 *   so `source_is_between_focus_and_target = true`.
 *
 * Consider a 2D circle in 3D space normal to the \f$z\f$ axis with
 * (3D) center \f$C_1\f$, a sphere with center \f$C_2\f$ and radius
 * \f$R_2\f$, and a projection point \f$P\f$.
 *
 * The parameter \f$z_\mathrm{extent}\f$ specifies the \f$z\f$-coordinate
 * (in the map's frame) of the rim circle where the spherical face of the
 * block meets the adjacent hollow-cylinder block.  This single number
 * determines the radius \f$R_1\f$ of the flat disk:
 *
 * \f{align}{
 *   t &= \frac{z_\mathrm{extent} - P_z}{C_1^z - P_z}, \\
 *   r_\mathrm{rim} &= \sqrt{R_2^2 - (z_\mathrm{extent} - C_2^z)^2}, \\
 *   R_1 &= \frac{r_\mathrm{rim}}{t}.
 * \f}
 *
 * CylindricalFlatEndcapInterior maps a 3D unit right cylinder
 * \f$(\bar{x},\bar{y},\bar{z})\f$ with \f$-1\leq\bar{z}\leq 1\f$ and
 * \f$\bar{x}^2+\bar{y}^2\leq 1\f$ so that:
 * - \f$\bar{z}=+1\f$ maps to the interior of the disk of radius
 *   \f$R_1\f$ centred at \f$C_1\f$.
 * - \f$\bar{z}=-1\f$ maps to the portion of the sphere on the *far* side
 *   of the flat disk from \f$P\f$.
 * - Curves of constant \f$(\bar{x},\bar{y})\f$ are portions of lines
 *   passing through \f$P\f$.
 * - The rim of the disk (\f$\bar{x}^2+\bar{y}^2=1\f$ on \f$\bar{z}=-1\f$)
 *   maps to the circle at \f$z = z_\mathrm{extent}\f$ on the sphere.
 *
 * Note that the \f$\bar{z}\f$ orientation is the *opposite* of
 * `CylindricalFlatEndcap`: here \f$\bar{z}=+1\f$ is the flat disk and
 * \f$\bar{z}=-1\f$ is the sphere.  This reversal is necessary to keep the
 * Jacobian determinant positive, because the flat disk sits at a larger
 * physical \f$z\f$ than the far sphere wall.
 *
 * CylindricalFlatEndcapInterior is intended for the Pill domain, where
 * the filled-cylinder endcap blocks have their flat face (\f$\bar{z}=+1\f$)
 * at the end of the inner cubed-cylinder region (inside the outer domain
 * sphere) and their spherical face (\f$\bar{z}=-1\f$) on the outer domain
 * sphere.
 *
 * ### Requirements on map parameters
 *
 * - \f$P\f$ is sufficiently inside the sphere:
 *   \f$|P - C_2| \leq 0.95\,R_2\f$.
 * - The flat disk lies inside the sphere, at least 5 % of \f$R_2\f$ below
 *   \f$C_2^z\f$ and at most 95 % of \f$R_2\f$ below \f$C_2^z\f$:
 *   \f[
 *     C_2^z - 0.95\,R_2 \;\leq\; C_1^z \;\leq\; C_2^z - 0.05\,R_2.
 *   \f]
 * - The flat disk is below the projection point: \f$C_1^z < P^z\f$.
 * - \f$z_\mathrm{extent}\f$ lies strictly on the sphere:
 *   \f$|z_\mathrm{extent} - C_2^z| < R_2\f$.
 * - The focal parameter satisfies \f$t > 1\f$ (the sphere is beyond the
 *   disk from \f$P\f$).
 * - The ratio \f$R_1/R_2\f$ is between 1/10 and 10.
 * - The entire flat disk rim lies strictly inside the sphere:
 *   \f[
 *     \sqrt{\bigl(\sqrt{(C_1^x-C_2^x)^2+(C_1^y-C_2^y)^2}+R_1\bigr)^2
 *           +(C_1^z-C_2^z)^2} < R_2.
 *   \f]
 *   This is required for the inverse to be defined everywhere: if any rim
 *   point were outside the sphere, the corresponding ray from \f$P\f$ would
 *   hit the sphere before reaching the disk (\f$t_\mathrm{sphere}<1\f$),
 *   violating the `source_is_between_focus_and_target` assumption.
 */
class CylindricalFlatEndcapInterior {
 public:
  static constexpr size_t dim = 3;

  /*!
   * \brief Construct the map.
   *
   * \param center_one Center of the flat disk (\f$C_1\f$).
   * \param center_two Center of the outer sphere (\f$C_2\f$).
   * \param proj_center Projection point \f$P\f$.
   * \param z_sphere_extent z-coordinate of the rim circle where the spherical
   *   face meets the adjacent hollow-cylinder block.  This determines the flat
   *   disk radius \f$R_1\f$ via the focal projection.
   * \param radius_two Radius of the outer sphere \f$R_2\f$.
   */
  CylindricalFlatEndcapInterior(const std::array<double, 3>& center_one,
                                const std::array<double, 3>& center_two,
                                const std::array<double, 3>& proj_center,
                                double z_sphere_extent, double radius_two);

  CylindricalFlatEndcapInterior() = default;
  ~CylindricalFlatEndcapInterior() = default;
  CylindricalFlatEndcapInterior(CylindricalFlatEndcapInterior&&) = default;
  CylindricalFlatEndcapInterior(const CylindricalFlatEndcapInterior&) = default;
  CylindricalFlatEndcapInterior& operator=(
      const CylindricalFlatEndcapInterior&) = default;
  CylindricalFlatEndcapInterior& operator=(CylindricalFlatEndcapInterior&&) =
      default;

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

  static constexpr bool supports_hessian{false};

 private:
  friend bool operator==(const CylindricalFlatEndcapInterior& lhs,
                         const CylindricalFlatEndcapInterior& rhs);
  FocallyLiftedMap<FocallyLiftedInnerMaps::FlatEndcap> impl_;
};

bool operator!=(const CylindricalFlatEndcapInterior& lhs,
                const CylindricalFlatEndcapInterior& rhs);

}  // namespace domain::CoordinateMaps
