// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Equiangular.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Non-linear map from \f$\xi \in [A, B]\rightarrow x \in [a, b]\f$.
 *
 * The formula for the mapping is:
 * \f{align}
 * x &= \frac{a}{2} \left(1-\mathrm{tan}\left(
 *      \frac{\pi(2\xi-B-A)}{4(B-A)}\right)\right) +
 *      \frac{b}{2} \left(1+\mathrm{tan}\left(
 *      \frac{\pi(2\xi-B-A)}{4(B-A)}\right)\right)\\
 * \xi &= \frac{A}{2} \left(1-\frac{4}{\pi}\mathrm{arctan}\left(
 *        \frac{2x-a-b}{b-a}\right)\right)+
 *        \frac{B}{2} \left(1+\frac{4}{\pi}\mathrm{arctan}\left(
 *        \frac{2x-a-b}{b-a}\right)\right)
 * \f}
 *
 * \note The intermediate step in which a tangent map is applied can be more
 * clearly understood if we define the coordinates:
 * \f{align}
 * \xi_{logical} &:= \frac{2\xi-B-A}{B-A} \in [-1, 1]\\
 * \Xi &:= \mathrm{tan}\left(\frac{\pi\xi_{logical}}{4}\right) \in [-1, 1]
 * \f}
 *
 * This map is intended to be used with `Wedge2D` and `Wedge3D` when equiangular
 * coordinates are chosen for those maps. For more information on this choice
 * of coordinates, see the documentation for Wedge3D.
 */
class Equiangular {
 public:
  static constexpr size_t dim = 1;

  Equiangular(double A, double B, double a, double b) noexcept;

  Equiangular() = default;
  ~Equiangular() = default;
  Equiangular(const Equiangular&) = default;
  Equiangular(Equiangular&&) noexcept = default;
  Equiangular& operator=(const Equiangular&) = default;
  Equiangular& operator=(Equiangular&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> operator()(
      const std::array<T, 1>& source_coords) const noexcept;

  boost::optional<std::array<double, 1>> inverse(
      const std::array<double, 1>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jacobian(
      const std::array<T, 1>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jacobian(
      const std::array<T, 1>& source_coords) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  friend bool operator==(const Equiangular& lhs,
                         const Equiangular& rhs) noexcept;

  double A_{-1.0};
  double B_{1.0};
  double a_{-1.0};
  double b_{1.0};
  double length_of_domain_over_m_pi_4_{(B_ - A_) / M_PI_4};  // 4(B-A)/\pi
  double length_of_range_{2.0};                              // b-a
  double m_pi_4_over_length_of_domain_{M_PI_4 / (B_ - A_)};
  double one_over_length_of_range_{0.5};
  // The jacobian for the affine map with the same parameters.
  double linear_jacobian_times_m_pi_4_{length_of_range_ /
                                       length_of_domain_over_m_pi_4_};
  // The inverse jacobian for the affine map with the same parameters.
  double linear_inverse_jacobian_over_m_pi_4_{length_of_domain_over_m_pi_4_ /
                                              length_of_range_};
};

inline bool operator!=(const CoordinateMaps::Equiangular& lhs,
                       const CoordinateMaps::Equiangular& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace CoordinateMaps
}  // namespace domain
