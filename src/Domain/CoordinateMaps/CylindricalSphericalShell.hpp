// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
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
 * \brief Maps a cylindrical shell block to a region bounded by an inner
 * right cylinder and an outer spherical surface.
 *
 * \image html CylindricalSphericalShell.png "The shaded region is the image."
 *
 * \details The logical coordinates are:
 * - radial \f$\xi \in [-1,1]\f$: interpolates between the inner right
 *   cylinder and the outer sphere.
 * - angular \f$\eta \in (-\pi, \pi]\f$: azimuthal angle \f$\phi\f$ (S1
 *   periodic direction).  The inverse map returns
 *   \f$\eta = \mathrm{atan2}(z, y)\f$, which lies in \f$(-\pi, \pi]\f$.
 * - axial \f$\zeta \in [-1,1]\f$: blends between the lower and upper ends
 *   of the shell.
 *
 * The physical coordinates \f$(x, y, z)\f$ are computed as follows.
 * Let \f$\alpha = (\xi+1)/2\f$ and \f$\beta = (\zeta+1)/2\f$.  Define
 *
 * \f{align}{
 *   x_\mathrm{inner}(\beta) &= x^\mathrm{inner}_\mathrm{lower}
 *       + \beta\,(x^\mathrm{inner}_\mathrm{upper}
 *                - x^\mathrm{inner}_\mathrm{lower}), \\
 *   x_\mathrm{outer}(\beta) &= x^\mathrm{outer}_\mathrm{lower}
 *       + \beta\,(x^\mathrm{outer}_\mathrm{upper}
 *                - x^\mathrm{outer}_\mathrm{lower}), \\
 *   r_\mathrm{outer}(\beta) &= \sqrt{r_\mathrm{sphere}^2
 *       - x_\mathrm{outer}(\beta)^2}.
 * \f}
 *
 * Then
 *
 * \f{align}{
 *   x &= (1-\alpha)\,x_\mathrm{inner}(\beta)
 *         + \alpha\,x_\mathrm{outer}(\beta), \\
 *   r &= (1-\alpha)\,r_\mathrm{inner}
 *         + \alpha\,r_\mathrm{outer}(\beta), \\
 *   y &= r\cos\eta, \quad z = r\sin\eta.
 * \f}
 *
 * The six block faces have the following geometry:
 * - \f$\xi = -1\f$ (\f$\alpha=0\f$): the inner right cylinder at
 *   \f$r = r_\mathrm{inner}\f$, extending in \f$x\f$ from
 *   \f$x^\mathrm{inner}_\mathrm{lower}\f$ to
 *   \f$x^\mathrm{inner}_\mathrm{upper}\f$.
 * - \f$\xi = +1\f$ (\f$\alpha=1\f$): a portion of the sphere
 *   \f$r^2 + x^2 = r_\mathrm{sphere}^2\f$.
 * - \f$\zeta = \pm 1\f$: generically curved surfaces in Cartesian
 *   coordinates (ruled surfaces that blend linearly in \f$(x,r)\f$ between
 *   the inner cylinder edge and the outer sphere edge at the corresponding
 *   axial end).
 * - \f$\eta\f$ direction: periodic (connected by the cylindrical_shell
 *   topology).
 *
 * ### Requirements
 * - \f$x^\mathrm{inner}_\mathrm{lower} < x^\mathrm{inner}_\mathrm{upper}\f$
 * - \f$x^\mathrm{outer}_\mathrm{lower} < x^\mathrm{outer}_\mathrm{upper}\f$
 * - \f$r_\mathrm{inner} > 0\f$
 * - \f$|x^\mathrm{outer}_\mathrm{lower}|,
 *   |x^\mathrm{outer}_\mathrm{upper}| < r_\mathrm{sphere}\f$
 * - \f$r_\mathrm{inner} < r_\mathrm{outer}(\beta)\f$ for all
 *   \f$\beta \in [0,1]\f$ (equivalently,
 *   \f$r_\mathrm{inner} < \min(r_\mathrm{outer}(0), r_\mathrm{outer}(1))\f$).
 */
class CylindricalSphericalShell {
 public:
  static constexpr size_t dim = 3;

  /*!
   * \brief Construct the map.
   *
   * \param x_inner_lower Axial coordinate \f$x\f$ at the lower end of the
   *   inner cylinder (\f$\xi=-1, \zeta=-1\f$).
   * \param x_inner_upper Axial coordinate \f$x\f$ at the upper end of the
   *   inner cylinder (\f$\xi=-1, \zeta=+1\f$).
   * \param x_outer_lower Axial coordinate \f$x\f$ at the lower end of the
   *   outer spherical face (\f$\xi=+1, \zeta=-1\f$).
   * \param x_outer_upper Axial coordinate \f$x\f$ at the upper end of the
   *   outer spherical face (\f$\xi=+1, \zeta=+1\f$).
   * \param r_inner Radius of the inner right cylinder.
   * \param r_sphere Radius of the outer bounding sphere (centered at the
   *   origin).
   */
  CylindricalSphericalShell(double x_inner_lower, double x_inner_upper,
                            double x_outer_lower, double x_outer_upper,
                            double r_inner, double r_sphere);

  CylindricalSphericalShell() = default;
  ~CylindricalSphericalShell() = default;
  CylindricalSphericalShell(CylindricalSphericalShell&&) = default;
  CylindricalSphericalShell(const CylindricalSphericalShell&) = default;
  CylindricalSphericalShell& operator=(const CylindricalSphericalShell&) =
      default;
  CylindricalSphericalShell& operator=(CylindricalSphericalShell&&) = default;

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
  friend bool operator==(const CylindricalSphericalShell& lhs,
                         const CylindricalSphericalShell& rhs);
  double x_inner_lower_{};
  double x_inner_upper_{};
  double x_outer_lower_{};
  double x_outer_upper_{};
  double r_inner_{};
  double r_sphere_{};
};

bool operator!=(const CylindricalSphericalShell& lhs,
                const CylindricalSphericalShell& rhs);

}  // namespace domain::CoordinateMaps
