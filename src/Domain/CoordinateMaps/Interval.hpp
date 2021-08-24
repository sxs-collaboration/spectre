// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Maps \f$\xi\f$ in the 1D interval \f$[A, B]\f$ to \f$x\f$ in the
 * interval \f$[a, b]\f$ according to a `domain::CoordinateMaps::Distribution`.
 *
 * \details The mapping takes a `domain::CoordinateMaps::Distribution` and
 * distributes the grid points accordingly.
 *
 * The formula for the mapping is, in case of a `Linear` distribution
 * \f{align}
 * x &= \frac{b}{B-A} (\xi-A) + \frac{a}{B-A} (B-\xi)\\
 * \xi &=\frac{B}{b-a} (x-a) + \frac{A}{b-a} (b-x)
 * \f}
 *
 * For every other distribution we use this linear mapping to map onto an
 * interval \f$[-1, 1]\f$, so we define:
 * \f{align}
 * f(\xi) &:= \frac{A+B-2\xi}{A-B} \in [-1, 1]\\
 * g(x) &:= \frac{a+b-2x}{a-b} \in [-1, 1]
 * \f}
 *
 * With this an `Equiangular` distribution is described by
 *
 * \f{align}
 * x &= \frac{a}{2} \left(1-\mathrm{tan}\left(\frac{\pi}{4}f(\xi)\right)\right)
 * + \frac{b}{2} \left(1+\mathrm{tan}\left(\frac{\pi}{4}f(\xi)\right)\right)\\
 * \xi &=
 * \frac{A}{2} \left(1-\frac{4}{\pi}\mathrm{arctan}\left(g(x)\right)\right) +
 * \frac{B}{2} \left(1+\frac{4}{\pi}\mathrm{arctan}\left(g(x)\right)\right)
 * \f}
 *
 * \note The equiangular distribution is intended to be used with the `Wedge`
 * map when equiangular coordinates are chosen for those maps. For more
 * information on this choice of coordinates, see the documentation for `Wedge`.
 *
 *
 * For both the `Logarithmic` and `Inverse` distribution, we first specify a
 * position for the singularity \f$c:=\f$`singularity_pos` outside the target
 * interval \f$[a, b]\f$.
 *
 * The `Logarithmic` distribution further requires the introduction of a
 * variable \f$\sigma\f$ dependent on whether \f$c\f$ is left (\f$ \sigma =
 * 1\f$) or right (\f$ \sigma = -1\f$) of the target interval. With this, the
 * `Logarithmic` distribution is described by
 *
 * \f{align}
 * x &= \sigma\, {\rm exp}\left(\frac{\ln(b-c)+\ln(a-c)}{2} + f(\xi)
 * \frac{\ln(b-c)-\ln(a-c)}{2}\right) + c\\
 * \xi &= \frac{B-A}{2}\frac{2\ln(\sigma [x-c]) - \ln(b-c) - \ln(a-c)}{\ln(b-c)
 * - \ln(a-c)} + \frac{B+A}{2} \f}
 *
 * and the `Inverse` distribution by
 *
 * \f{align}
 * x &= \frac{2(a-c)(b-c)}{a+b-2c+(a-b)f(\xi)} + c\\
 * \xi &= -\frac{A-B}{2(a-b)} \left(\frac{2(a-c)(b-c)}{x-c} - (a-c) -
 * (b-c)\right) + \frac{A+B}{2}
 * \f}
 */
class Interval {
 public:
  static constexpr size_t dim = 1;

  Interval(double A, double B, double a, double b, Distribution distribution,
           std::optional<double> singularity_pos = std::nullopt) noexcept;

  Interval() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> operator()(
      const std::array<T, 1>& source_coords) const noexcept;

  std::optional<std::array<double, 1>> inverse(
      const std::array<double, 1>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jacobian(
      const std::array<T, 1>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jacobian(
      const std::array<T, 1>& source_coords) const noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  bool is_identity() const noexcept { return is_identity_; }

 private:
  friend bool operator==(const Interval& lhs, const Interval& rhs) noexcept;

  double A_{std::numeric_limits<double>::signaling_NaN()};
  double B_{std::numeric_limits<double>::signaling_NaN()};
  double a_{std::numeric_limits<double>::signaling_NaN()};
  double b_{std::numeric_limits<double>::signaling_NaN()};
  Distribution distribution_{Distribution::Linear};
  std::optional<double> singularity_pos_{std::nullopt};
  bool is_identity_{false};
};

inline bool operator!=(const CoordinateMaps::Interval& lhs,
                       const CoordinateMaps::Interval& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace domain::CoordinateMaps
