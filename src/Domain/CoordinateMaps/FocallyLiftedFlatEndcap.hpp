// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Contains FocallyLiftedInnerMaps
namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {
/*!
 * \brief A FocallyLiftedInnerMap that maps a 3D unit right cylinder
 *  to a volume that connects a portion of a plane and a spherical
 *  surface.
 *
 * \details The domain of the map is a 3D unit right cylinder with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$\bar{x}^2+\bar{y}^2 \leq
 * 1\f$.  The range of the map has coordinates \f$(x,y,z)\f$.
 *
 * Consider a 2D circle in 3D space that is normal to the \f$z\f$ axis
 * and has (3D) center \f$C^i\f$ and radius \f$R\f$.  `FlatEndcap`
 * provides the following functions:
 *
 * ### forward_map()
 * `forward_map()` maps \f$(\bar{x},\bar{y},\bar{z}=-1)\f$ to the interior
 * of the circle.  The arguments to `forward_map()`
 * are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 * `forward_map()` returns \f$x_0^i\f$,
 * the 3D coordinates on the circle, which are given by
 *
 * \f{align}
 * x_0^0 &= R \bar{x} + C^0,\\
 * x_0^1 &= R \bar{y} + C^1,\\
 * x_0^2 &= C^2.
 * \f}
 *
 * ### sigma
 *
 * \f$\sigma\f$ is a function that is zero on the plane
 * \f$x^i=x_0^i\f$ and unity at \f$\bar{z}=+1\f$ (corresponding to the
 * upper surface of the FocallyLiftedMap). We define
 *
 * \f{align}
 *  \sigma &= \frac{\bar{z}+1}{2}.
 * \f}
 *
 * ### deriv_sigma
 *
 * `deriv_sigma` returns
 *
 * \f{align}
 *  \frac{\partial \sigma}{\partial \bar{x}^j} &= (0,0,1/2).
 * \f}
 *
 * ### jacobian
 *
 * `jacobian` returns \f$\partial x_0^k/\partial \bar{x}^j\f$.
 * The arguments to `jacobian`
 * are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 *
 * Differentiating Eqs.(1--3) above yields
 *
 * \f{align*}
 * \frac{\partial x_0^0}{\partial \bar{x}} &= R,\\
 * \frac{\partial x_0^1}{\partial \bar{y}} &= R,
 * \f}
 * and all other components are zero.
 *
 * ### inverse
 *
 * `inverse` takes \f$x_0^i\f$ and \f$\sigma\f$ as arguments, and
 * returns \f$(\bar{x},\bar{y},\bar{z})\f$, or a default-constructed
 * `std::optional<std::array<double, 3>>` if
 * \f$x_0^i\f$ or \f$\sigma\f$ are outside the range of the map.
 * The formula for the inverse is straightforward:
 *
 * \f{align}
 *  \bar{x} &= \frac{x_0^0-C^0}{R},\\
 *  \bar{y} &= \frac{x_0^1-C^1}{R},\\
 *  \bar{z} &= 2\sigma - 1.
 * \f}
 *
 * If \f$\bar{z}\f$ is outside the range \f$[-1,1]\f$ or
 * if \f$\bar{x}^2+\bar{y}^2 > 1\f$ then we return
 * a default-constructed `std::optional<std::array<double, 3>>`
 *
 * ### lambda_tilde
 *
 * `lambda_tilde` takes as arguments a point \f$x^i\f$ and a projection point
 *  \f$P^i\f$, and computes \f$\tilde{\lambda}\f$, the solution to
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda}.\f}
 *
 * Since \f$x_0^i\f$ must lie on the plane \f$x_0^3=C^3\f$,
 *
 * \f{align} \tilde{\lambda} &= \frac{C^3-P^3}{x^3-P^3}.\f}
 *
 * For `FocallyLiftedInnerMaps::FlatEndcap`, \f$x^i\f$ is always between
 * \f$P^i\f$ and \f$x_0^i\f$, so \f$\tilde{\lambda}\ge 1\f$.
 * Therefore a default-constructed `std::optional<double>` is returned
 * if \f$\tilde{\lambda}\f$ is less than unity (meaning that \f$x^i\f$
 * is outside the range of the map).
 *
 * ### deriv_lambda_tilde
 *
 * `deriv_lambda_tilde` takes as arguments \f$x_0^i\f$, a projection point
 *  \f$P^i\f$, and \f$\tilde{\lambda}\f$, and
 *  returns \f$\partial \tilde{\lambda}/\partial x^i\f$. We have
 *
 * \f{align}
 * \frac{\partial\tilde{\lambda}}{\partial x^3} =
 * -\frac{C^3-P^3}{(x^3-P^3)^2} = -\frac{\tilde{\lambda}^2}{C^3-P^3},
 * \f}
 * and other components are zero.
 *
 * ### inv_jacobian
 *
 * `inv_jacobian` returns \f$\partial \bar{x}^i/\partial x_0^k\f$,
 *  where \f$\sigma\f$ is held fixed.
 *  The arguments to `inv_jacobian`
 *  are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 *
 * The nonzero components are
 * \f{align}
 * \frac{\partial \bar{x}}{\partial x_0^0} &= \frac{1}{R},\\
 * \frac{\partial \bar{y}}{\partial x_0^1} &= \frac{1}{R}.
 * \f}
 *
 * ### dxbar_dsigma
 *
 * `dxbar_dsigma` returns \f$\partial \bar{x}^i/\partial \sigma\f$,
 *  where \f$x_0^i\f$ is held fixed.
 *
 * From Eq. (6) we have
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial \sigma} &= (0,0,2).
 * \f}
 *
 */
class FlatEndcap {
 public:
  FlatEndcap(const std::array<double, 3>& center, double radius);

  FlatEndcap() = default;
  ~FlatEndcap() = default;
  FlatEndcap(FlatEndcap&&) = default;
  FlatEndcap(const FlatEndcap&) = default;
  FlatEndcap& operator=(const FlatEndcap&) = default;
  FlatEndcap& operator=(FlatEndcap&&) = default;

  template <typename T>
  void forward_map(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          target_coords,
      const std::array<T, 3>& source_coords) const;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double sigma_in) const;

  template <typename T>
  void jacobian(const gsl::not_null<
                    tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
                    jacobian_out,
                const std::array<T, 3>& source_coords) const;

  template <typename T>
  void inv_jacobian(const gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3,
                                                 Frame::NoFrame>*>
                        inv_jacobian_out,
                    const std::array<T, 3>& source_coords) const;

  template <typename T>
  void sigma(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> sigma_out,
             const std::array<T, 3>& source_coords) const;

  template <typename T>
  void deriv_sigma(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          deriv_sigma_out,
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  void dxbar_dsigma(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          dxbar_dsigma_out,
      const std::array<T, 3>& source_coords) const;

  std::optional<double> lambda_tilde(
      const std::array<double, 3>& parent_mapped_target_coords,
      const std::array<double, 3>& projection_point,
      bool source_is_between_focus_and_target) const;

  template <typename T>
  void deriv_lambda_tilde(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          deriv_lambda_tilde_out,
      const std::array<T, 3>& target_coords, const T& lambda_tilde,
      const std::array<double, 3>& projection_point) const;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  static bool is_identity() { return false; }

 private:
  friend bool operator==(const FlatEndcap& lhs, const FlatEndcap& rhs);
  std::array<double, 3> center_{};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const FlatEndcap& lhs, const FlatEndcap& rhs);
}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
