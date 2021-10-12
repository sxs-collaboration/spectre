// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {
/*!
 * \brief A FocallyLiftedInnerMap that maps a 3D unit right cylindrical shell
 *  to a volume that connects portions of two spherical surfaces.
 *
 * \details The domain of the map is a 3D unit right cylinder with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$1\leq \bar{x}^2+\bar{y}^2 \leq
 * 4\f$.  The range of the map has coordinates \f$(x,y,z)\f$.
 *
 * Consider a sphere with center \f$C^i\f$ and radius \f$R\f$ that is
 * intersected by two planes normal to the \f$z\f$ axis located at
 * \f$z = z_\mathrm{L}\f$ and \f$z = z_\mathrm{U}\f$, with
 * \f$z_\mathrm{L} < z_\mathrm{U}\f$.
 * `Side` provides the following functions:
 *
 * ### forward_map()
 * `forward_map()` maps \f$(\bar{x},\bar{y},\bar{z})\f$ to a point on the inner
 * surface
 * \f$\bar{x}^2+\bar{y}^2=1\f$ by dividing \f$\bar{x}\f$ and \f$\bar{y}\f$
 * by \f$(1+\sigma)\f$, where \f$\sigma\f$ is the function given by Eq. (7)
 * below.
 * Then it maps that point to a point on the portion of the sphere with
 * \f$z_\mathrm{L} \leq z \leq z_\mathrm{U}\f$.
 * `forward_map()` returns
 * \f$x_0^i\f$, the 3D coordinates on that sphere, which are given by
 *
 * \f{align}
 * x_0^0 &= R \sin\theta \frac{\bar{x}}{1+\sigma} + C^0,\\
 * x_0^1 &= R \sin\theta \frac{\bar{y}}{1+\sigma} + C^1,\\
 * x_0^2 &= R \cos\theta + C^2.\\
 * \f}
 *
 * Here
 * \f{align}
 * \theta = \theta_\mathrm{max} +
 * (\theta_\mathrm{min}-\theta_\mathrm{max}) \frac{\bar{z}+1}{2},
 * \f}
 *
 * where
 * \f{align}
 * \cos(\theta_\mathrm{max}) &= (z_\mathrm{L}-C^2)/R,\\
 * \cos(\theta_\mathrm{min}) &= (z_\mathrm{U}-C^2)/R.
 * \f}
 *
 * Note that \f$\theta\f$ decreases with increasing \f$\bar{z}\f$,
 * which is the usual convention for a polar angle but might otherwise
 * cause confusion.
 *
 * ### sigma
 *
 * \f$\sigma\f$ is a function that is zero on the sphere
 * \f$x^i=x_0^i\f$ and unity at \f$\bar{x}^2+\bar{y}^2=4\f$
 * (corresponding to the upper surface of the FocallyLiftedMap). We define
 *
 * \f{align}
 *  \sigma &= \sqrt{\bar{x}^2+\bar{y}^2}-1.
 * \f}
 *
 * ### deriv_sigma
 *
 * `deriv_sigma` returns
 *
 * \f{align}
 *  \frac{\partial \sigma}{\partial \bar{x}^j} &=
 * \left(\frac{\bar{x}}{1+\sigma},
 *       \frac{\bar{y}}{1+\sigma},0\right).
 * \f}
 *
 * ### jacobian
 *
 * `jacobian` returns \f$\partial x_0^k/\partial \bar{x}^j\f$.
 * The arguments to `jacobian` are \f$(\bar{x},\bar{y},\bar{z})\f$.
 * Differentiating Eqs.(1--4) above yields
 *
 * \f{align*}
 * \frac{\partial x_0^0}{\partial \bar{x}} &= R \sin\theta
 * \frac{\bar{y}^2}{(1+\sigma)^3}, \\
 * \frac{\partial x_0^0}{\partial \bar{y}} &= -R \sin\theta
 * \frac{\bar{x}\bar{y}}{(1+\sigma)^3}, \\
 * \frac{\partial x_0^0}{\partial \bar{z}} &=
 * R \cos\theta \frac{\theta_\mathrm{min}-\theta_\mathrm{max}}{2(1+\sigma)}
 *   \bar{x},\\
 * \frac{\partial x_0^1}{\partial \bar{x}} &= -R \sin\theta
 * \frac{\bar{x}\bar{y}}{(1+\sigma)^3}, \\
 * \frac{\partial x_0^1}{\partial \bar{y}} &= R \sin\theta
 * \frac{\bar{x}^2}{(1+\sigma)^3}, \\
 * \frac{\partial x_0^1}{\partial \bar{z}} &=
 * R \cos\theta \frac{\theta_\mathrm{min}-\theta_\mathrm{max}}{2(1+\sigma)}
 *   \bar{y},\\
 * \frac{\partial x_0^2}{\partial \bar{x}} &= 0,\\
 * \frac{\partial x_0^2}{\partial \bar{y}} &= 0,\\
 * \frac{\partial x_0^2}{\partial \bar{z}} &=
 * - R \sin\theta \frac{\theta_\mathrm{min}-\theta_\mathrm{max}}{2}.
 * \f}
 *
 * ### inverse
 *
 * `inverse` takes \f$x_0^i\f$ and \f$\sigma\f$ as arguments, and
 * returns \f$(\bar{x},\bar{y},\bar{z})\f$, or a default-constructed
 * `std::optional<std::array<double, 3>>` if \f$x_0^i\f$ or \f$\sigma\f$ are
 * outside the range of the map.
 *
 * If \f$\sigma\f$ is outside the range \f$[0,1]\f$ then we return
 * a default-constructed `std::optional<std::array<double, 3>>`.
 *
 * To get \f$\bar{z}\f$ we invert Eq. (4):
 * \f{align}
 * \bar{z} &= 2\frac{\acos\left((x_0^2-C^2)/R\right)-\theta_\mathrm{max}}
 *            {\theta_\mathrm{min}-\theta_\mathrm{max}} - 1.
 * \f}
 *
 * If \f$\bar{z}\f$ is outside the range \f$[-1,1]\f$ then we return
 * a default-constructed `std::optional<std::array<double, 3>>`.
 *
 * To compute \f$\bar{x}\f$ and \f$\bar{y}\f$, we invert Eqs. (1--3) and
 * use \f$\sigma\f$:
 *
 * \f{align}
 *  \bar{x} &= \frac{(x_0^0-C^0) (1+\sigma)}{\rho},\\
 *  \bar{y} &= \frac{(x_0^1-C^1) (1+\sigma)}{\rho},
 * \f}
 *
 * where
 *
 * \f{align}
 * \rho = \sqrt{(x_0^0-C^0)^2+(x_0^1-C^1)^2}.
 * \f}
 *
 * ### lambda_tilde
 *
 * `lambda_tilde` takes as arguments a point \f$x^i\f$ and a projection point
 *  \f$P^i\f$, and computes \f$\tilde{\lambda}\f$, the solution to
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda}.\f}
 *
 * Since \f$x_0^i\f$ must lie on the sphere, \f$\tilde{\lambda}\f$ is the
 * solution of the quadratic equation
 *
 * \f{align}
 * |P^i + (x^i - P^i) \tilde{\lambda} - C^i |^2 - R^2 = 0.
 * \f}
 *
 * In solving the quadratic, we choose the larger root if
 * \f$x^2>z_\mathrm{P}\f$ and the smaller root otherwise. We demand
 * that the root is greater than unity.  If there is no such root,
 * this means that the point \f$x^i\f$ is not in the range of the map
 * so we return a default-constructed `std::optional<double>`.
 *
 * ### deriv_lambda_tilde
 *
 * `deriv_lambda_tilde` takes as arguments \f$x_0^i\f$, a projection point
 *  \f$P^i\f$, and \f$\tilde{\lambda}\f$, and
 *  returns \f$\partial \tilde{\lambda}/\partial x^i\f$.
 * By differentiating Eq. (14), we find
 *
 * \f{align}
 * \frac{\partial\tilde{\lambda}}{\partial x^j} &=
 * \tilde{\lambda}^2 \frac{C^j - x_0^j}{
 * (x_0^i - P^i)(x_{0i} - C_{i})} \nonumber \\
 * &= \tilde{\lambda}^2 \frac{C^j - x_0^j}{|x_0^i - P^i|^2
 * + (x_0^i - P^i)(P_i - C_{i})}.
 * \f}
 *
 * ### inv_jacobian
 *
 * `inv_jacobian` returns \f$\partial \bar{x}^i/\partial x_0^k\f$,
 *  where \f$\sigma\f$ is held fixed.
 * The arguments to `inv_jacobian` are \f$(\bar{x},\bar{y},\bar{z})\f$.
 *
 * Note from Eqs. (9--12) that \f$\bar{x}\f$ and \f$\bar{y}\f$
 * depend only on \f$x_0^0\f$ and \f$x_0^1\f$ but not on \f$x_0^2\f$.
 *
 * By differentiating Eqs. (9--12), we find
 *
 * \f{align*}
 * \frac{\partial \bar{x}}{\partial x_0^0} &=
 * \frac{\bar{y}^2}{(1+\sigma)\rho},\\
 * \frac{\partial \bar{x}}{\partial x_0^1} &=
 * - \frac{\bar{x}\bar{y}}{(1+\sigma)\rho},\\
 * \frac{\partial \bar{x}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{y}}{\partial x_0^0} &=
 * - \frac{\bar{x}\bar{y}}{(1+\sigma)\rho},\\
 * \frac{\partial \bar{y}}{\partial x_0^1} &=
 * \frac{\bar{x}^2}{(1+\sigma)\rho},\\
 * \frac{\partial \bar{y}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{z}}{\partial x_0^0} &= 0,\\
 * \frac{\partial \bar{z}}{\partial x_0^1} &= 0,\\
 * \frac{\partial \bar{z}}{\partial x_0^2} &=
 * -\frac{2}{\rho(\theta_\mathrm{min}-\theta_\mathrm{max})},
 * \f}
 *
 * where
 *
 * \f[
 *   \rho = R \sin\theta = R\sin\left(\theta_\mathrm{max} +
 * (\theta_\mathrm{min}-\theta_\mathrm{max}) \frac{\bar{z}+1}{2}\right),
 * \f]
 *
 * which is also equal to the quantity in Eq. (12).
 *
 * ### dxbar_dsigma
 *
 * `dxbar_dsigma` returns \f$\partial \bar{x}^i/\partial \sigma\f$,
 *  where \f$x_0^i\f$ is held fixed.
 *
 * From Eqs. (10) and (11) we have
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial \sigma} &=
 * \left(\frac{\bar{x}}{\sqrt{\bar{x}^2+\bar{y}^2}},
 *       \frac{\bar{y}}{\sqrt{\bar{x}^2+\bar{y}^2}},0\right).
 * \f}
 *
 */
class Side {
 public:
  static constexpr size_t dim = 3;
  Side(const std::array<double, 3>& center, const double radius,
       const double z_lower, const double z_upper);

  Side() = default;
  ~Side() = default;
  Side(Side&&) = default;
  Side(const Side&) = default;
  Side& operator=(const Side&) = default;
  Side& operator=(Side&&) = default;

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

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static bool is_identity() { return false; }

 private:
  friend bool operator==(const Side& lhs, const Side& rhs);
  std::array<double, 3> center_{
      make_array<3>(std::numeric_limits<double>::signaling_NaN())};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  double theta_min_{std::numeric_limits<double>::signaling_NaN()};
  double theta_max_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Side& lhs, const Side& rhs);
}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
