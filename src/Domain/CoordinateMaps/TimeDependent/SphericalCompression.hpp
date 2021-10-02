// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps::TimeDependent {
/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Time-dependent compression of a finite 3D spherical volume.
 *
 * \details Let \f$\xi^i\f$ be the unmapped coordinates, and let \f$\rho\f$ be
 * the Euclidean radius corresponding to these coordinates with respect to
 * some center \f$C^i\f$. The transformation implemented by this map is
 * equivalent to the following transformation: at each point, the mapped
 * coordinates are the same as the unmapped coordinates, except in
 * a spherical region \f$\rho \leq \rho_{\rm max}\f$, where instead coordinates
 * are mapped using a compression that is spherically symmetric about the center
 * \f$C^i\f$. The amount of compression decreases linearly from a maximum at
 * \f$\rho = \rho_{\rm min}\f$ to zero at \f$\rho = \rho_{\rm max}\f$. A
 * scalar domain::FunctionsOfTime::FunctionOfTime \f$\lambda_{00}(t)\f$ controls
 * the amount of compression.
 *
 * The mapped coordinates are a continuous function of the unmapped
 * coordinates, but the Jacobians are not continuous at \f$\rho_{\rm min}\f$
 * and \f$\rho_{\rm max}\f$. Therefore, \f$\rho_{\rm min}\f$ and \f$\rho_{\rm
 * max}\f$ should both be surfaces corresponding to block boundaries. Therefore,
 * this class implements the transformation described above as follows: the
 * if the template parameter `InteriorMap` is true, the map is the one
 * appropriate for \f$\rho < \rho_{\rm min}\f$, while if `InteriorMap` is false,
 * the map is the one appropriate for \f$\rho_{\rm min} \leq \rho \leq \rho_{\rm
 * max}\f$. To use this map, add it to the blocks where the transformation is
 * not the identity, using the appropriate template parameter, depending on
 * which region the block is in.
 *
 * \note Currently, this map only performs a compression. A generalization of
 * this map could also change the region's shape as well as its size, by
 * including more terms than the spherically symmetric term included here.
 *
 * ### Mapped coordinates
 *
 * The mapped coordinates
 * \f$x^i\f$ are related to the unmapped coordinates \f$\xi^i\f$
 * as follows:
 * \f{align}{
 * x^i &= \left\{\begin{array}{ll}\xi^i - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho^i}{\rho_{\rm min}}, & \rho < \rho_{\rm min}, \\
 * \xi^i - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}} \rho^i, &
 * \rho_{\rm min} \leq \rho \leq \rho_{\rm max}, \\
 * \xi^i, & \rho_{\rm max} < \rho,\end{array}\right.
 * \f}
 * where \f$\rho^i = \xi^i - C^i\f$ is the Euclidean radial position vector in
 * the unmapped coordinates with respect to the center \f$C^i\f$, \f$\rho =
 * \sqrt{\delta_{kl}\left(\xi^k - C^l\right)\left(\xi^l - C^l\right)}\f$ is the
 * Euclidean magnitude of \f$\rho^i\f$, and \f$\rho_j = \delta_{ij} \rho^i\f$.
 *
 * ### Frame velocity
 *
 * The frame velocity \f$v^i \equiv dx^i/dt\f$ is then
 * \f{align}{
 * v^i &= \left\{\begin{array}{ll} - \frac{\lambda_{00}^\prime(t)}{\sqrt{4\pi}}
 * \frac{\rho^i}{\rho_{\rm min}}, & \rho < \rho_{\rm min}, \\
 * - \frac{\lambda_{00}^\prime(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}} \rho^i,
 * & \rho_{\rm min} \leq \rho \leq \rho_{\rm max}, \\
 * 0, & \rho_{\rm max} < \rho,\end{array}\right.
 * \f} where \f$\lambda_{00}^\prime(t) \equiv d\lambda_{00}/dt\f$.
 *
 * ### Jacobian
 *
 * Differentiating the equations for \f$x^i\f$ gives the Jacobian
 * \f$\partial x^i / \partial \xi^j\f$. Using the result
 * \f{align}{
 * \frac{\partial \rho^i}{\partial \xi^j} &= \frac{\partial}{\partial \xi^j}
 * \left(\xi^i - C^i\right) = \frac{\partial \xi^i}{\partial \xi^j}
 * = \delta^i_{j}
 * \f}
 * and taking the derivatives yields
 * \f{align}{
 * \frac{\partial x^i}{\partial \xi^j} &= \left\{\begin{array}{ll}
 * \delta^i_j \left(1
 * - \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \frac{1}{\rho_{\rm min}}\right),
 * & \rho < \rho_{\rm min},\\
 * \delta^i_j
 * \left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * - \rho^i \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\partial}{\partial \xi^j}\left(
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right),
 * & \rho_{\rm min} \leq \rho < \rho_{\rm max},\\
 * \delta^i_j, & \rho_{\rm max} < \rho.\end{array}\right.
 * \f}
 * Inserting
 * \f{align}{
 * \frac{\partial}{\partial \xi^j}\left(
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * &= \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}}
 * \frac{\partial}{\partial \xi^j}\left(\frac{1}{\rho}\right)
 * = - \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}} \frac{1}{\rho^2}
 * \frac{\partial \rho}{\partial \xi^j}
 * \f}
 * and
 * \f{align}{
 * \frac{\partial \rho}{\partial \xi^j} &= \frac{\rho_j}{\rho}.
 * \f}
 * into the Jacobian yields
 * \f{align}{
 * \frac{\partial x^i}{\partial \xi^j} &= \left\{\begin{array}{ll}
 * \delta^i_j \left(1
 * - \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \frac{1}{\rho_{\rm min}}\right),
 * & \rho < \rho_{\rm min},\\
 * \delta^i_j
 * \left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * + \rho^i \rho_j \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}}\frac{1}{\rho^3},
 * & \rho_{\rm min} \leq \rho < \rho_{\rm max},\\
 * \delta^i_j, & \rho_{\rm max} < \rho.\end{array}\right.
 * \f}
 *
 * ### Inverse Jacobian
 *
 * This map finds the inverse Jacobian by first finding the Jacobian and then
 * numerically inverting it.
 *
 * ### Inverse map
 *
 * For \f$\lambda_{00}(t)\f$ that satisfy
 * \f{align}{
 * \rho_{\rm min} - \rho_{\rm max} < \lambda_{00}(t) / \sqrt{4\pi} <
 * \rho_{\rm min},
 * \f}
 * the map will be invertible and nonsingular. For simplicity, here we
 * enforce this condition, even though perhaps the map might be generalized to
 * handle cases that are still invertible but violate this condition. This
 * avoids the need to specially handle the cases
 * \f$\lambda_{00}(t) / \sqrt{4\pi} = \rho_{\rm min} - \rho_{\rm max}\f$
 * and \f$\lambda_{00}(t) / \sqrt{4\pi} = \rho_{\rm min}\f$, both of which
 * yield a singular map, and it also avoids cases where the map behaves
 * in undesirable ways (such as a larger \f$\lambda_{00}(t)\f$ leading to
 * an expansion and a coordinate inversion instead of a compression).
 *
 * After
 * requiring the above inequality to be satisfied, however, the inverse mapping
 * can be derived as follows. Let \f$r^i \equiv x^i - C^i\f$. In terms of
 * \f$r^i\f$, the map is \f{align}{ r^i &= \left\{\begin{array}{ll}\rho^i
 * \left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}\right), & \rho < \rho_{\rm min}, \\
 * \rho^i\left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right),
 * & \rho_{\rm min} \leq \rho \leq \rho_{\rm max}, \\
 * \rho^i, & \rho_{\rm max} < \rho.\end{array}\right.
 * \f}
 *
 * Taking the Euclidean magnitude of both sides and simplifying yields
 * \f{align}{
 * \frac{r}{\rho} &= \left\{\begin{array}{ll}
 * 1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}, & \rho < \rho_{\rm min}, \\
 * 1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max}/\rho - 1}{\rho_{\rm max} - \rho_{\rm min}},
 * & \rho_{\rm min} \leq \rho \leq \rho_{\rm max}, \\
 * 1, & \rho_{\rm max} < \rho,\end{array}\right.
 * \f}
 * which implies
 * \f{align}{
 * r^i = \rho^i \frac{r}{\rho} \Rightarrow \rho^i = r^i \frac{\rho}{r}.
 * \f}
 *
 * Inserting \f$\rho_{\rm min}\f$ or \f$\rho_{\rm max}\f$ then gives the
 * corresponding bounds in the mapped coordinates: \f{align}{
 * r_{\rm min} &= \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}},\\
 * r_{\rm max} &= \rho_{\rm max}.
 * \f}
 *
 * In the regime \f$\rho_{\rm min} \leq \rho < \rho_{\rm max}\f$, rearranging
 * yields a linear relationship between \f$\rho\f$ and \f$r\f$, which
 * can then be solved for \f$\rho(r)\f$:
 * \f{align}{
 * r &= \rho - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} - \rho}{\rho_{\rm max} - \rho_{\rm min}}\\
 * \Rightarrow r &= \rho \left(1 + \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}}.
 * \f}
 * Solving this linear equation for \f$\rho\f$ yields
 * \f{align}{
 * \rho &= \left(r+\frac{\lambda_{00}(t)}{\sqrt{4\pi}}\frac{\rho_{\rm
 * max}}{\rho_{\rm max}-\rho_{\rm min}}\right)
 * \left(1 + \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm max} - \rho_{\rm min}}\right)^{-1}.
 * \f}
 *
 * Inserting the expressions for \f$\rho\f$ into the equation
 * \f{align}{
 * \rho^i = r^i \frac{\rho}{r}
 * \f}
 * then gives
 * \f{align}{
 * \rho^i &= \left\{\begin{array}{ll}
 * r^i\left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}\right)^{-1},
 * & r < \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}},\\
 * r^i
 * \left(1+\frac{1}{r}\frac{\lambda_{00}(t)}{\sqrt{4\pi}}\frac{\rho_{\rm
 * max}}{\rho_{\rm max}-\rho_{\rm min}}\right)\left(1 +
 * \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \frac{1}{\rho_{\rm max} - \rho_{\rm
 * min}}\right)^{-1}, & \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \leq r
 * \leq \rho_{\rm max},\\
 * r^i, & \rho_{\rm max} < r.\end{array}\right.
 * \f}
 * Finally, inserting \f$\rho^i = \xi^i - C^i\f$ yields the inverse map:
 * \f{align}{
 * \xi^i &= \left\{\begin{array}{ll}
 * r^i\left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}\right)^{-1} + C^i,
 * & r < \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}},\\
 * r^i
 * \left(1+\frac{1}{r}\frac{\lambda_{00}(t)}{\sqrt{4\pi}}\frac{\rho_{\rm
 * max}}{\rho_{\rm max}-\rho_{\rm min}}\right)\left(1 +
 * \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \frac{1}{\rho_{\rm max} - \rho_{\rm
 * min}}\right)^{-1} + C^i, & \rho_{\rm min} -
 * \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \leq r
 * \leq \rho_{\rm max},\\
 * r^i + C^i = x^i, & \rho_{\rm max} < r.\end{array}\right.
 * \f}
 *
 */
template <bool InteriorMap>
class SphericalCompression {
 public:
  static constexpr size_t dim = 3;

  explicit SphericalCompression(std::string function_of_time_name,
                                double min_radius, double max_radius,
                                const std::array<double, 3>& center);
  SphericalCompression() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> frame_velocity(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  void pup(PUP::er& p);  // NOLINT

  static bool is_identity() { return false; }

 private:
  friend bool operator==(const SphericalCompression& lhs,
                         const SphericalCompression& rhs) {
    return lhs.f_of_t_name_ == rhs.f_of_t_name_ and
           lhs.min_radius_ == rhs.min_radius_ and
           lhs.max_radius_ == rhs.max_radius_ and lhs.center_ == rhs.center_;
  }
  std::string f_of_t_name_;
  double min_radius_ = std::numeric_limits<double>::signaling_NaN();
  double max_radius_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_;
};

template <bool InteriorMap>
bool operator!=(const SphericalCompression<InteriorMap>& lhs,
                const SphericalCompression<InteriorMap>& rhs) {
  return not(lhs == rhs);
}
}  // namespace domain::CoordinateMaps::TimeDependent
