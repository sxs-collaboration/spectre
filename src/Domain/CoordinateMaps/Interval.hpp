// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
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
 * which we use in each other distribution to map an interval to \f$[-1, 1]\f$
 * with
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
 * For both the `Logarithmic` and `Inverse` distribution, we first shift the
 * singularity out of the target interval by specifying
 * \f$c:=\f$`singularity_pos` to get
 *
 * for the `Logarithmic` distribution
 *
 * \f{align}
 * x &= {\rm exp}\left(\frac{\ln(b-c)+\ln(a-c)}{2} + f(\xi)
 * \frac{\ln(b-c)-\ln(a-c)}{2}\right) + c\\
 * f(\xi) &= \frac{2\ln(x-c) - \ln(b-c) - \ln(a-c)}{\ln(b-c) - \ln(a-c)}
 * \f}
 *
 * and for the `Inverse` distribution
 *
 * \f{align}
 * x &= \frac{2(a-c)(b-c)}{a+b-2c+(a-b)f(\xi)} + c\\
 * f(\xi) &= \frac{1}{a-b} \left(\frac{2(a-c)(b-c)}{x-c} - (a-c) - (b-c)\right)
 * \f}
 */
class Interval {
 public:
  static constexpr size_t dim = 1;

  Interval(double A, double B, double a, double b, Distribution distribution,
           double singularity_pos =
               std::numeric_limits<double>::signaling_NaN()) noexcept;

  Interval() = default;
  ~Interval() = default;
  Interval(const Interval&) = default;
  Interval(Interval&&) noexcept = default;
  Interval& operator=(const Interval&) = default;
  Interval& operator=(Interval&&) = default;

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
  double singularity_pos_{std::numeric_limits<double>::signaling_NaN()};
  bool is_identity_{false};
};

inline bool operator!=(const CoordinateMaps::Interval& lhs,
                       const CoordinateMaps::Interval& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace domain::CoordinateMaps
