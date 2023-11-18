// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <pup.h>

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
template <typename T>
T Wedge::Surface::distance(const std::array<T, 3>& coords) const {
  // Short circuit if it's a sphere. Then the distance is trivially the radius
  // of this surface
  if (sphericity == 1.0) {
    return make_with_value<T>(coords[0], radius);
  }

  ASSERT(coords[2] != 0.0,
         "For wedge transition, rotated coords cannot have z component equal "
         "to zero.");

  // D = R * [ (1 - s) / (sqrt(3) * cos(theta)) + s]
  // cos(theta) = z / r
  return radius *
         ((1.0 - sphericity) / (coords[2] * sqrt(3.0)) * magnitude(coords) +
          sphericity);
}

void Wedge::Surface::pup(PUP::er& p) {
  p | radius;
  p | sphericity;
}

bool Wedge::Surface::operator==(const Wedge::Surface& other) const {
  return radius == other.radius and sphericity == other.sphericity;
}

bool Wedge::Surface::operator!=(const Wedge::Surface& other) const {
  return not(*this == other);
}

Wedge::Wedge(double inner_radius, double outer_radius, double inner_sphericity,
             double outer_sphericity, OrientationMap<3> orientation_map)
    : inner_surface_(Surface{inner_radius, inner_sphericity}),
      outer_surface_(Surface{outer_radius, outer_sphericity}),
      orientation_map_(std::move(orientation_map)),
      direction_(orientation_map_.inverse_map()(Direction<3>::upper_zeta())) {}

double Wedge::operator()(const std::array<double, 3>& source_coords) const {
  return call_impl<double>(source_coords);
}
DataVector Wedge::operator()(
    const std::array<DataVector, 3>& source_coords) const {
  return call_impl<DataVector>(source_coords);
}

std::optional<double> Wedge::original_radius_over_radius(
    const std::array<double, 3>& target_coords, double distorted_radius) const {
  const double radius = magnitude(target_coords);
  CAPTURE_FOR_ERROR(target_coords);
  CAPTURE_FOR_ERROR(distorted_radius);
  CAPTURE_FOR_ERROR(radius);
  CAPTURE_FOR_ERROR(orientation_map_);
  CAPTURE_FOR_ERROR(direction_);

  // Couple protections that would make a point completely outside of the domain
  // of validity for any wedge
  if (equal_within_roundoff(radius, 0.0) or
      equal_within_roundoff(distorted_radius, 1.0) or distorted_radius > 1.0) {
    return std::nullopt;
  }

  // First we check the extremal case of being outside the outermost radius. We
  // can check the outermost radius because its surface doesn't move. We can't
  // check the innermost surface because the inner bound is the origin which we
  // already checked above.
  if (radius - eps_ > outer_surface_.radius) {
    return std::nullopt;
  }

  // Rotate the coordinates so they are aligned with +z. This makes checking if
  // the target_coord is in the proper wedge very simple since we only have to
  // check if the target coord is in the direction of the +z wedge
  const std::array<double, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), target_coords);
  CAPTURE_FOR_ERROR(rotated_coords);

  // This comparison only works because the opening angle of the wedge is pi/2
  // in both x and y
  if (rotated_coords[2] <
      std::max(abs(rotated_coords[0]), abs(rotated_coords[1]))) {
    return std::nullopt;
  }

  // If distorted radius is 0, this means the map is the identity so the radius
  // and the original radius are equal. Also we don't want to divide by 0 below
  if (equal_within_roundoff(distorted_radius, 0.0)) {
    return std::optional{1.0};
  }

  const double outer_distance = outer_surface_.distance(rotated_coords);
  const double inner_distance = inner_surface_.distance(rotated_coords);
  const double distance_difference = outer_distance - inner_distance;

  // If we are at the overall outer distance, then the transition function is 0
  // so the map is again the identity so the radius and original radius are
  // equal. We can't check the overall inner distance because that has been
  // distorted so we don't know where the mapped inner distance is.
  if (equal_within_roundoff(radius, outer_distance)) {
    return std::optional{1.0};
  }

  const double linear_a = -1.0 / distance_difference;
  const double linear_b = -linear_a * outer_distance;

  const double denom = 1. - distorted_radius * linear_a;
  // prevent zero division
  if (equal_within_roundoff(denom, 0.)) {
    return std::nullopt;
  }
  const double original_radius = (radius + distorted_radius * linear_b) / denom;

  return (original_radius + eps_) >= inner_distance and
                 (original_radius - eps_) <= outer_distance
             ? std::optional<double>{original_radius / radius}
             : std::nullopt;
}

std::array<double, 3> Wedge::gradient(
    const std::array<double, 3>& source_coords) const {
  return gradient_impl<double>(source_coords);
}
std::array<DataVector, 3> Wedge::gradient(
    const std::array<DataVector, 3>& source_coords) const {
  return gradient_impl<DataVector>(source_coords);
}

template <typename T>
T Wedge::call_impl(const std::array<T, 3>& source_coords) const {
  const std::array<T, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), source_coords);
  check_distances(rotated_coords);
  T outer_distance = outer_surface_.distance(rotated_coords);

  return (outer_distance - magnitude(rotated_coords)) /
         (outer_distance - inner_surface_.distance(rotated_coords));
}

template <typename T>
std::array<T, 3> Wedge::gradient_impl(
    const std::array<T, 3>& source_coords) const {
  // If both surfaces are spherical then we short circuit because the distances
  // are constant and we only need to take a derivative of r.
  // (grad f)_i = -(x_i/r)/(D_out - D_in)
  const std::array<T, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), source_coords);
  check_distances(rotated_coords);
  if (inner_surface_.sphericity == 1.0 and outer_surface_.sphericity == 1.0) {
    const T one_over_denom = 1.0 / (magnitude(rotated_coords) *
                                    (outer_surface_.distance(rotated_coords) -
                                     inner_surface_.distance(rotated_coords)));

    return discrete_rotation(orientation_map_,
                             -rotated_coords * one_over_denom);
  }

  const T radius = magnitude(rotated_coords);

  const T one_over_radius = 1.0 / radius;
  T outer_distance = outer_surface_.distance(rotated_coords);
  const T one_over_denom =
      1.0 / (outer_distance - inner_surface_.distance(rotated_coords));
  T& outer_distance_minus_radius_over_denom = outer_distance;
  outer_distance_minus_radius_over_denom -= radius;
  outer_distance_minus_radius_over_denom *= one_over_denom;

  // Avoid small negative numbers if we are at outer boundary
  for (size_t i = 0; i < get_size(radius); i++) {
    if (equal_within_roundoff(
            get_element(outer_distance_minus_radius_over_denom, i), 0.0)) {
      get_element(outer_distance_minus_radius_over_denom, i) = 0.0;
    }
  }

  // Regardless of the sphericities below, we always need this factor in the
  // first term so we calculate it now.
  std::array<T, 3> result = -1.0 * rotated_coords * one_over_radius;

  const auto make_factor = [&one_over_radius](const Surface& surface) -> T {
    return (1.0 - surface.sphericity) * surface.radius * cube(one_over_radius) /
           sqrt(3.0);
  };

  // We can make some simplifications if either of the surfaces are spherical
  // because then the derivative of the distance is zero since it's constant. In
  // the first two branches, it's safe to assume the other sphericity isn't 1
  // because of the above check.
  if (outer_surface_.sphericity == 1.0) {
    T total_factor = make_factor(inner_surface_);
    total_factor *= outer_distance_minus_radius_over_denom;

    for (size_t i = 0; i < 2; i++) {
      gsl::at(result, i) -=
          total_factor * rotated_coords[2] * gsl::at(rotated_coords, i);
    }

    result[2] += total_factor * (square(radius) - square(rotated_coords[2]));
  } else if (inner_surface_.sphericity == 1.0) {
    T total_factor = make_factor(outer_surface_);
    total_factor *= (1.0 - outer_distance_minus_radius_over_denom);

    for (size_t i = 0; i < 2; i++) {
      gsl::at(result, i) -=
          total_factor * rotated_coords[2] * gsl::at(rotated_coords, i);
    }

    result[2] += total_factor * (square(radius) - square(rotated_coords[2]));
  } else {
    T inner_total_factor = make_factor(inner_surface_);
    inner_total_factor *= outer_distance_minus_radius_over_denom;
    T outer_total_factor = make_factor(outer_surface_);
    outer_total_factor *= (1.0 - outer_distance_minus_radius_over_denom);

    for (size_t i = 0; i < 2; i++) {
      gsl::at(result, i) -= (inner_total_factor + outer_total_factor) *
                            rotated_coords[2] * gsl::at(rotated_coords, i);
    }

    result[2] += (outer_total_factor + inner_total_factor) *
                 (square(radius) - square(rotated_coords[2]));
  }

  // Finally, need one more factor of D_out - D_in in the denominator and to
  // rotate it back to the proper orientation
  return discrete_rotation(orientation_map_, result * one_over_denom);
}

bool Wedge::operator==(const ShapeMapTransitionFunction& other) const {
  if (dynamic_cast<const Wedge*>(&other) == nullptr) {
    return false;
  }
  const Wedge& other_ref = *dynamic_cast<const Wedge*>(&other);
  return inner_surface_ == other_ref.inner_surface_ and
         outer_surface_ == other_ref.outer_surface_ and
         orientation_map_ == other_ref.orientation_map_;
}

bool Wedge::operator!=(const ShapeMapTransitionFunction& other) const {
  return not(*this == other);
}

// checks that the magnitudes are all between `r_min_` and `r_max_`
template <typename T>
void Wedge::check_distances([
    [maybe_unused]] const std::array<T, 3>& coords) const {
#ifdef SPECTRE_DEBUG
  const T mag = magnitude(coords);
  const T inner_distance = inner_surface_.distance(coords);
  const T outer_distance = outer_surface_.distance(coords);
  for (size_t i = 0; i < get_size(mag); ++i) {
    if (get_element(mag, i) + eps_ < get_element(inner_distance, i) or
        get_element(mag, i) - eps_ > get_element(outer_distance, i)) {
      ERROR(
          "The Wedge transition map was called with coordinates outside "
          "the set inner and outer surfaces. The inner radius and sphericity "
          "are (r="
          << inner_surface_.radius << ",s=" << inner_surface_.sphericity
          << ") and the outer radius and sphericity are (r="
          << outer_surface_.radius << ",s=" << outer_surface_.sphericity
          << "). The inner distance is " << get_element(inner_distance, i)
          << ", the outer distance is " << get_element(outer_distance, i)
          << ". The requested point has radius: " << get_element(mag, i));
    }
  }
#endif  // SPECTRE_DEBUG
}

void Wedge::pup(PUP::er& p) {
  ShapeMapTransitionFunction::pup(p);
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | inner_surface_;
    p | outer_surface_;
    p | orientation_map_;
    p | direction_;
  }
}

Wedge::Wedge(CkMigrateMessage* const msg) : ShapeMapTransitionFunction(msg) {}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID Wedge::my_PUP_ID = 0;

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
