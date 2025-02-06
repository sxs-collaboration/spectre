// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <pup.h>
#include <type_traits>
#include <vector>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

SphereTransition::SphereTransition(const double r_min, const double r_max,
                                   const bool reverse, const bool interior)
    : r_min_(r_min), r_max_(r_max), interior_(interior) {
  if (r_min <= 0.) {
    ERROR("The minimum radius must be greater than 0 but is " << r_min);
  }
  inverse_cube_r_min_ = 1.0 / cube(r_min_);
  if (r_max <= r_min) {
    ERROR(
        "The maximum radius must be greater than the minimum radius but "
        "r_max =  "
        << r_max << ", and r_min = " << r_min);
  }
  if (interior and reverse) {
    ERROR("Cannot be reverse while also in the interior.");
  }
  a_ = -1.0 / (r_max - r_min);
  b_ = -a_ * r_max;
  if (reverse) {
    a_ *= -1.0;
    b_ = 1.0 - b_;
  }
}

double SphereTransition::operator()(
    const std::array<double, 3>& source_coords,
    const std::optional<size_t>& one_over_radius_power) const {
  return call_impl<double>(source_coords, one_over_radius_power);
}

DataVector SphereTransition::operator()(
    const std::array<DataVector, 3>& source_coords,
    const std::optional<size_t>& one_over_radius_power) const {
  return call_impl<DataVector>(source_coords, one_over_radius_power);
}

std::optional<double> SphereTransition::original_radius_over_radius(
    const std::array<double, 3>& target_coords,
    double radial_distortion) const {
  const double mag = magnitude(target_coords);
  // If we are at the center, the radius is the same
  if (UNLIKELY(equal_within_roundoff(mag, 0.0))) {
    return interior_ ? std::optional{1.0} : std::nullopt;
  }

  // a_ being positive is a sentinel for reversed.
  // If we aren't reversed, check near or within r_min_.
  if (a_ < 0.0) {
    if (mag + radial_distortion < (1.0 - eps_) * r_min_) {
      if (not interior_) {
        return std::nullopt;
      }

      const double factor =
          1.0 / (inverse_cube_r_min_ * square(mag) * radial_distortion);

      // The following variable names are defined in Numerical Recipes (3rd
      // edition) pg 228. We choose to keep them as they appear in NR for better
      // comparison with the book.

      // Numerical Recipes (3rd edition) pg 228, eq 5.6.10
      const double R = 0.5 * factor;
      const double Q = factor / 3.0;
      const bool multiple_real_roots = square(R) < cube(Q);

      if (multiple_real_roots) {
        // Numerical Recipes (3rd edition) pg 228, eqs 5.6.11 - 5.6.12
        const double theta = acos(R / sqrt(cube(Q)));
        std::vector<double> roots{
            -2.0 * sqrt(Q) * cos(theta / 3.0),
            -2.0 * sqrt(Q) * cos((theta + 2.0 * M_PI) / 3.0),
            -2.0 * sqrt(Q) * cos((theta - 2.0 * M_PI) / 3.0)};

        // Radii are positive
        std::erase_if(roots, [](const double root) { return root < 0.0; });

        // Since the root of this is the original radius over target radius, it
        // will be of order unity and not something super large, so we take the
        // smallest of the positive roots. If this turns out to not be robust
        // enough we can change this.
        return *alg::min_element(roots);
      } else {
        // Numerical Recipes (3rd edition) pg 228, eqs 5.6.13 - 5.6.17
        const double A = -sgn(R) * cbrt(abs(R) + sqrt(square(R) - cube(Q)));
        const double B = A == 0.0 ? 0.0 : Q / A;

        return A + B;
      }
    } else if (equal_within_roundoff(mag + radial_distortion, r_min_)) {
      // IntMid: Since the point is on the boundary, we can't tell if this is
      // the interior or not, but either map will work.
      return std::optional{r_min_ / (r_min_ - radial_distortion)};
    }
  }

  // Beyond the range of validity for both reversed and not reversed
  if ((a_ < 0.0 and mag > (1.0 + eps_) * r_max_) or
      (a_ > 0.0 and mag < (1.0 - eps_) * r_min_)) {
    return std::nullopt;
  }

  // No distortion means our point is the same
  if (equal_within_roundoff(radial_distortion, 0.0)) {
    return std::optional{1.0};
  }

  // At the f=0 boundary.
  if ((a_ < 0.0 and equal_within_roundoff(mag, r_max_)) or
      (a_ > 0.0 and equal_within_roundoff(mag, r_min_))) {
    return std::optional{1.0};
  }

  const double denom = 1. - radial_distortion * a_;
  // prevent zero division
  if (UNLIKELY(equal_within_roundoff(denom, 0.))) {
    return std::nullopt;
  }

  const double original_radius = (mag + radial_distortion * b_) / denom;

  // Check at or beyond f=1 boundary for reversed
  if (a_ > 0.0) {
    if (equal_within_roundoff(original_radius, r_max_)) {
      return std::optional{1.0 + radial_distortion / mag};
    } else if (original_radius > (1.0 + eps_) * r_max_) {
      return std::nullopt;
    }
  }

  // We are within r_min and r_max and not at a boundary
  return std::optional{original_radius / mag};
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
T SphereTransition::call_impl(
    const std::array<T, 3>& source_coords,
    const std::optional<size_t>& one_over_radius_power) const {
  const T mag = magnitude(source_coords);

#ifdef SPECTRE_DEBUG
  if (UNLIKELY(one_over_radius_power.value_or(0_st) >= 3)) {
    for (size_t i = 0; i < get_size(source_coords[0]); i++) {
      if (equal_within_roundoff(get_element(mag, i), 0.0)) {
        const std::array<double, 3> point{get_element(source_coords[0], i),
                                          get_element(source_coords[1], i),
                                          get_element(source_coords[2], i)};
        ERROR("Trying to divide by a point "
              << point << " with radius zero in SphereTransition operator.");
      }
    }
  }
#endif

  if (interior_) {
    return inverse_cube_r_min_ *
           (one_over_radius_power.value_or(0_st) < 3
                ? T{integer_pow(mag,
                                static_cast<int>(
                                    2 - one_over_radius_power.value_or(0_st)))}
                : 1.0 /
                      integer_pow(mag, static_cast<int>(
                                           one_over_radius_power.value() - 2)));
  }

#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(source_coords[0]); i++) {
    const std::array<double, 3> point{get_element(source_coords[0], i),
                                      get_element(source_coords[1], i),
                                      get_element(source_coords[2], i)};
    if (UNLIKELY(get_element(mag, i) < (1.0 - eps_) * r_min_)) {
      ERROR("SphereTransition coord " << point << " with radius "
                                      << get_element(mag, i)
                                      << " is within r_min of " << r_min_
                                      << ", but the class was not constructed "
                                         "for the interior of the sphere.");
    } else if (UNLIKELY(get_element(mag, i) > (1.0 + eps_) * r_max_)) {
      ERROR("SphereTransition coord "
            << point << " with radius " << get_element(mag, i)
            << "is beyond r_max of " << r_max_ << ".");
    }
  }
#endif

  T result = (a_ * mag + b_);
  // Avoid roundoff
  result = blaze::clamp(result, 0.0, 1.0);

  return result /
         integer_pow(
             mag, static_cast<int>(1 + one_over_radius_power.value_or(0_st)));
}

template <typename T>
std::array<T, 3> SphereTransition::gradient_impl(
    const std::array<T, 3>& source_coords) const {
  // Short circuit for the interior
  if (interior_) {
    return 2.0 * inverse_cube_r_min_ * source_coords;
  }

#ifdef SPECTRE_DEBUG
  const T mag = magnitude(source_coords);

  for (size_t i = 0; i < get_size(source_coords[0]); i++) {
    const std::array<double, 3> point{get_element(source_coords[0], i),
                                      get_element(source_coords[1], i),
                                      get_element(source_coords[2], i)};
    if (UNLIKELY(equal_within_roundoff(get_element(mag, i), 0.0))) {
      ERROR("Trying to divide by a point "
            << point << " with radius zero in SphereTransition gradient.");
    }

    if (UNLIKELY(get_element(mag, i) < (1.0 - eps_) * r_min_)) {
      ERROR("SphereTransition gradient coord "
            << point << " with radius " << get_element(mag, i)
            << " is within r_min of " << r_min_
            << ", but the class was not constructed for the interior of the "
               "sphere.");
    } else if (UNLIKELY(get_element(mag, i) > (1.0 + eps_) * r_max_)) {
      ERROR("SphereTransition gradient coord "
            << point << " with radius " << get_element(mag, i)
            << "is beyond r_max of " << r_max_ << ".");
    }
  }
#endif

  // We can call the operator() and be sure it won't error because we did the
  // checks here as well.
  return source_coords *
         (a_ / dot(source_coords, source_coords) - (*this)(source_coords, {2}));
}

bool SphereTransition::operator==(
    const ShapeMapTransitionFunction& other) const {
  if (dynamic_cast<const SphereTransition*>(&other) == nullptr) {
    return false;
  }
  const auto& derived = dynamic_cast<const SphereTransition&>(other);
  // no need to check `a_` or `b_` as they are uniquely determined by `r_min_`
  // and `r_max_`.
  return this->r_min_ == derived.r_min_ and this->r_max_ == derived.r_max_ and
         this->interior_ == derived.interior_;
}

bool SphereTransition::operator!=(
    const ShapeMapTransitionFunction& other) const {
  return not(*this == other);
}

void SphereTransition::pup(PUP::er& p) {
  ShapeMapTransitionFunction::pup(p);
  size_t version = 1;
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

  if (p.isUnpacking()) {
    inverse_cube_r_min_ = 1.0 / cube(r_min_);
  }

  if (version >= 1) {
    p | interior_;
  } else if (p.isUnpacking()) {
    interior_ = false;
  }
}

SphereTransition::SphereTransition(CkMigrateMessage* const msg)
    : ShapeMapTransitionFunction(msg) {}

PUP::able::PUP_ID SphereTransition::my_PUP_ID = 0;  // NOLINT

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
