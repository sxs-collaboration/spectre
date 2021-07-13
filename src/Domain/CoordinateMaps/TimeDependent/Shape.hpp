// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeTransitionFunctions.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"
#pragma once

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
 * \brief Given a point with cartesian coordinates \f$\xi^i\f$, let the polar
 * coordinates \f$(r, \theta, \phi)\f$ with respect to a center \f$x_c^i\f$ be
 * defined in the usual way: \f{align}{
 * \xi^0 - x_c^0 &= r sin(\theta) cos(\phi)\\
 * \xi^1 - x_c^1 &= r sin(\theta) sin(\phi)\\
 * \xi^2 - x_c^2 &= r cos(\theta)
 * \f}
 * The shape map distorts the distance \f$r\f$ between the point and the center
 * while leaving the angles \f$\theta\f$, \f$\phi\f$ between them preserved by
 * applying a spherical harmonic expansion with time-dependent coefficients
 * \f$\lambda_{lm}(t)\f$. An additional domain-dependent transition function
 * \f$f(r, \theta, \phi)\f$ ensures that the distortion falls off correctly to
 * zero at the boundary of the domain. The monopole and dipole coefficients \f$
 * \lambda_{00}, \lambda_{10}, \lambda_{11}\f$ are expected to be zero since
 * these degrees of freedoms are controlled by the `SphericalCompression` and
 * `Translation` map, respectively. The shape map maps the unmapped coordinates
 * \f$\xi^i\f$ to coordinates \f$x^i\f$:
 *
 * \f{equation}{
 * x^i = \xi^i - (\xi^i - x_c^i) f(r, \theta, \phi) \sum_{lm}
 * \lambda_{lm}(t)Y_{lm}(\theta, \phi).
 * \f}
 *
 * The inverse map is given by:
 * \f{equation}{
 * \xi^i = x_c^i + (x^i-x_c^i)*(r/\tilde{r}),
 * \f}
 * where \f$\tilde{r}\f$ is the radius of \f$\xi\f$, calculated by the
 * transition map. For more details, see
 * `TransitionFunc::original_radius_over_radius` in
 * `Domain/CoordinateMaps/Protocols.hpp`.
 *
 * The frame velocity \f$v^i\ = dx^i / dt\f$ is calculated trivially:
 * \f{equation}{
 * v^i = x^i - (x^i - x_c^i) f(r, \theta, \phi) \sum_{lm}
 * \dot{\lambda}_{lm}(t)Y_{lm}(\theta, \phi).
 * \f}
 *
 * The Jacobian is given by:
 * \f{equation}{
 * \partial_j x^i = \delta_j^i \left( 1 - f(r, \theta, \phi) \sum_{lm}
 * \lambda_{lm}(t)Y_{lm}(\theta, \phi)\right) + (\xi^i - x_c^i)
 * \left(\partial_jf(r, \theta, \phi) \sum_{lm} \lambda_{lm}(t)Y_{lm}(\theta,
 * \phi) + f(r, \theta, \phi) \sum_{lm} \lambda_{lm}(t) \partial_j
 * Y_{lm}(\theta, \phi) \right).
 * \f}
 *
 * The inverse Jacobian is computed by numerically inverting the Jacobian.
 *
 * For future optimization, the `interpolation_info` objects calculated in all
 * functions of this class could be cached. Since every element should evaluate
 * the same grid coordinates most time steps, this might greatly decrease
 * computation. Every element has their own clone of the shape map so the
 * caching could be done with member variables. Care must be taken that
 * `jacobian` currently calculates the `interpolation_info` with an order
 * higher.
 */
template <typename TransitionFunc>
class Shape {
 public:
  static_assert(
      tt::assert_conforms_to<TransitionFunc, protocols::TransitionFunc>);
  using FunctionsOfTimeMap = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  explicit Shape(const std::array<double, 3>& center, size_t l_max,
                 size_t m_max, TransitionFunc transition_func,
                 std::string function_of_time_name) noexcept;

  Shape() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const noexcept;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> frame_velocity(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT
  static bool is_identity() noexcept { return false; }
  static constexpr size_t dim = 3;

 private:
  std::string f_of_t_name_;
  std::array<double, 3> center_{};
  size_t l_max_ = 2;
  size_t m_max_ = 2;
  YlmSpherepack ylm_{2, 2};
  TransitionFunc transition_func_{};

  template <typename T>
  inline std::array<tt::remove_cvref_wrap_t<T>, 3> center_coordinates_(
      const std::array<T, 3>& coords) const noexcept {
    return {coords[0] - center_[0], coords[1] - center_[1],
            coords[2] - center_[2]};
  }

  // Checks that the vector of coefficients has the right size and that the
  // monopole and dipole coefficients are zero.
  void check_coefficients_(const DataVector& coefs) const noexcept;

  friend bool operator==(const Shape& lhs, const Shape& rhs) noexcept {
    return lhs.f_of_t_name_ == rhs.f_of_t_name_ and
           lhs.center_ == rhs.center_ and lhs.l_max_ == rhs.l_max_ and
           lhs.m_max_ == rhs.m_max_ and
           lhs.transition_func_ == rhs.transition_func_;
  }
};

template <typename TransitionFunc>
bool operator!=(const Shape<TransitionFunc>& lhs,
                const Shape<TransitionFunc>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace domain::CoordinateMaps::TimeDependent
