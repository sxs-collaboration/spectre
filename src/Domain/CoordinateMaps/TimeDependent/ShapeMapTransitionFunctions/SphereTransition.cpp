// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"

#include <array>
#include <optional>
#include <pup.h>

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeString.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

SphereTransition::SphereTransition(const double r_min, const double r_max,
                                   const bool reverse)
    : r_min_(r_min), r_max_(r_max) {
  if (r_min <= 0.) {
    ERROR("The minimum radius must be greater than 0 but is " << r_min);
  }
  if (r_max <= r_min) {
    ERROR(
        "The maximum radius must be greater than the minimum radius but "
        "r_max =  "
        << r_max << ", and r_min = " << r_min);
  }
  a_ = -1.0 / (r_max - r_min);
  b_ = -a_ * r_max;
  if (reverse) {
    a_ *= -1.0;
    b_ = 1.0 - b_;
  }
}

double SphereTransition::operator()(
    const std::array<double, 3>& source_coords) const {
  return call_impl<double>(source_coords);
}
DataVector SphereTransition::operator()(
    const std::array<DataVector, 3>& source_coords) const {
  return call_impl<DataVector>(source_coords);
}

std::optional<double> SphereTransition::original_radius_over_radius(
    const std::array<double, 3>& target_coords,
    double radial_distortion) const {
  const double mag = magnitude(target_coords);
  const double denom = 1. - radial_distortion * a_;
  // prevent zero division
  if (equal_within_roundoff(mag, 0.) or equal_within_roundoff(denom, 0.)) {
    return std::nullopt;
  }
  const double original_radius = (mag + radial_distortion * b_) / denom;

  // If we are within r_min and r_max, doesn't matter if we are reverse, we just
  // return the value we calculated. If we are reverse and the point is outside
  // r_max or we aren't reversed and the point is inside r_min, then we return a
  // simplified formula since the transition function is 1 in this region. If
  // the above conditions aren't true, then we are in the region where the
  // transition function is 0, so we return 1.0. Otherwise we return nullopt.
  if ((original_radius + eps_) >= r_min_ and
      (original_radius - eps_) <= r_max_) {
    return std::optional<double>{original_radius / mag};
  } else if ((a_ > 0.0 and original_radius > r_max_) or
             (a_ < 0.0 and original_radius < r_min_)) {
    // a_ being positive is a sentinel for reverse (see constructor)
    return {1.0 + radial_distortion / mag};
  } else if ((a_ < 0.0 and original_radius > r_max_) or
             (a_ > 0.0 and original_radius < r_min_)) {
    return {1.0};
  } else {
    return std::nullopt;
  }
}

std::array<double, 3> SphereTransition::gradient(
    const std::array<double, 3>& source_coords) const {
  return gradient_impl<double>(source_coords);
}
std::array<DataVector, 3> SphereTransition::gradient(
    const std::array<DataVector, 3>& source_coords) const {
  return gradient_impl<DataVector>(source_coords);
}

template <typename T>
T SphereTransition::call_impl(const std::array<T, 3>& source_coords) const {
  const T mag = magnitude(source_coords);
  check_magnitudes(mag, false);
  // See https://github.com/sxs-collaboration/spectre/issues/6376 for why the
  // T{} is necessary inside the clamp.
  return blaze::clamp(T{a_ * mag + b_}, 0.0, 1.0);
}

template <typename T>
std::array<T, 3> SphereTransition::gradient_impl(
    const std::array<T, 3>& source_coords) const {
  const T mag = magnitude(source_coords);
  check_magnitudes(mag, true);
  return a_ * source_coords / mag;
}

bool SphereTransition::operator==(
    const ShapeMapTransitionFunction& other) const {
  if (dynamic_cast<const SphereTransition*>(&other) == nullptr) {
    return false;
  }
  const auto& derived = dynamic_cast<const SphereTransition&>(other);
  // no need to check `a_` and `b_` as they are uniquely determined by
  // `r_min_` and `r_max_`.
  return this->r_min_ == derived.r_min_ and this->r_max_ == derived.r_max_;
}

bool SphereTransition::operator!=(
    const ShapeMapTransitionFunction& other) const {
  return not(*this == other);
}

// if we need the point to be between the r_min and r_max, check that,
// otherwise just check that the radius is positive
template <typename T>
void SphereTransition::check_magnitudes(
    [[maybe_unused]] const T& mag,
    [[maybe_unused]] const bool check_bounds) const {
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(mag); ++i) {
    const bool point_is_bad = check_bounds
                                  ? (get_element(mag, i) + eps_ < r_min_ or
                                     get_element(mag, i) - eps_ > r_max_)
                                  : get_element(mag, i) <= 0.0;
    if (point_is_bad) {
      ERROR(
          "The sphere transition map was called with bad coordinates. The "
          "requested point has magnitude "
          << get_element(mag, i)
          << (check_bounds
                  ? (MakeString{} << " which is outside the set minimum and "
                                     "maxiumum radius. The minimum radius is "
                                  << r_min_ << ", the maximum radius is "
                                  << r_max_ << ".")
                  : (MakeString{} << " <= 0.0")));
    }
  }
#endif  // SPECTRE_DEBUG
}

void SphereTransition::pup(PUP::er& p) {
  ShapeMapTransitionFunction::pup(p);
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | r_min_;
    p | r_max_;
    p | a_;
    p | b_;
  }
}

SphereTransition::SphereTransition(CkMigrateMessage* const msg)
    : ShapeMapTransitionFunction(msg) {}

PUP::able::PUP_ID SphereTransition::my_PUP_ID = 0;

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
