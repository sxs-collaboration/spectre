// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <optional>

#include "Domain/CoordinateMaps/Protocols.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"

#pragma once

namespace PUP {
class er;
}  // namespace PUP

namespace domain::CoordinateMaps {
/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief A transition function that falls off as \f$f(r) = (ar + b) / r\f$. The
 * coefficients \f$a\f$ and \f$b\f$ are chosen so that the map falls off
 * linearly from 1 at `r_min` to 0 at `r_max`.
 */
class SphereTransition : public tt::ConformsTo<protocols::TransitionFunc> {
 public:
  SphereTransition(double r_min, double r_max)
      : r_min_(r_min),
        r_max_(r_max),
        a_(-1. / (r_max - r_min)),
        b_(r_max / (r_max - r_min)) {
    if (r_min <= 0.) {
      ERROR("The minimum radius must be greater than 0 but is " << r_min);
    }
    if (r_max <= r_min) {
      ERROR(
          "The maximum radius must be greater than the minimum radius but "
          "r_max =  "
          << r_max << ", and r_min = " << r_min);
    }
  }

  explicit SphereTransition() = default;

  template <typename T>
  T operator()(const std::array<T, 3>& source_coords) const {
    const T mag = magnitude(source_coords);
    check_magnitudes(mag);
    return a_ + b_ / mag;
  }

  std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double distorted_radius) const {
    const double mag = magnitude(target_coords);
    const double denom = 1. - distorted_radius * a_;
    // prevent zero division
    if (mag == 0. or denom == 0.) {
      return std::nullopt;
    }
    const double original_radius = (mag + distorted_radius * b_) / denom;

    return original_radius >= r_min_ and original_radius <= r_max_
               ? std::optional<double>{original_radius / mag}
               : std::nullopt;
  }

  template <typename T>
  T map_over_radius(const std::array<T, 3>& source_coords) const {
    const T mag = magnitude(source_coords);
    check_magnitudes(mag);
    return a_ / mag + b_ / square(mag);
  }

  template <typename T>
  std::array<T, 3> gradient(const std::array<T, 3>& source_coords) const {
    const T mag = magnitude(source_coords);
    check_magnitudes(mag);
    return -b_ * source_coords / cube(mag);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {
    p | r_min_;
    p | r_max_;
    p | a_;
    p | b_;
  }  // NOLINT

 private:
  friend bool operator==(const SphereTransition& lhs,
                         const SphereTransition& rhs) noexcept {
    // no need to check `a_` and `b_` as they are uniquely determined by
    // `r_min_` and `r_max_`.
    return lhs.r_min_ == rhs.r_min_ and lhs.r_max_ == rhs.r_max_;
  }

  // checks that the magnitudes are all between `r_min_` and `r_max_`
  template <typename T>
  void check_magnitudes(const T& mag) const {
    for (size_t i = 0; i < get_size(mag); ++i) {
      if (get_element(mag, i) < r_min_ or get_element(mag, i) > r_max_) {
        ERROR(
            "The sphere transition map was called with coordinates outside the "
            "set minimum and maximum radius. The minimum radius is "
            << r_min_ << ", the maximum radius is " << r_max_
            << ". The requested point has magnitude: " << get_element(mag, i));
      }
    }
  }

  double r_min_{};
  double r_max_{};
  double a_{};
  double b_{};
};

}  // namespace domain::CoordinateMaps
