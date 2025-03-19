// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <type_traits>
#include <vector>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
Wedge::Surface::Surface(const std::array<double, 3>& center_in,
                        const double radius_in, const double sphericity_in)
    : center(center_in),
      radius(radius_in),
      sphericity(sphericity_in),
      half_cube_length(radius / sqrt(3.0)) {}

void Wedge::Surface::pup(PUP::er& p) {
  p | center;
  p | radius;
  p | sphericity;
  p | half_cube_length;
}

size_t Wedge::axis_index(const Axis axis) {
  if (axis == Axis::Interior or axis == Axis::None) {
    ERROR("Cannot get the index of the axis " << axis);
  }
  return static_cast<size_t>(abs(static_cast<int>(axis)) - 1);
}

double Wedge::axis_sgn(const Axis axis) {
  if (axis == Axis::Interior or axis == Axis::None) {
    ERROR("Cannot get the sign of the axis " << axis);
  }
  return static_cast<double>(sgn(static_cast<int>(axis)));
}

std::ostream& operator<<(std::ostream& os, const Wedge::Axis axis) {
  if (axis == Wedge::Axis::Interior or axis == Wedge::Axis::None) {
    os << (axis == Wedge::Axis::Interior ? "Interior" : "None");
    return os;
  }

  os << (Wedge::axis_sgn(axis) < 0.0 ? "-" : "+") << Wedge::axis_index(axis);
  return os;
}

Wedge::Wedge(const std::array<double, 3>& inner_center,
             const double inner_radius, const double inner_sphericity,
             const std::array<double, 3>& outer_center,
             const double outer_radius, const double outer_sphericity,
             const Axis axis, const bool reverse)
    : inner_surface_(inner_center, inner_radius, inner_sphericity),
      outer_surface_(outer_center, outer_radius, outer_sphericity),
      projection_center_(inner_surface_.center - outer_surface_.center),
      axis_(axis),
      reverse_(reverse) {
  if (projection_center_ != std::array{0.0, 0.0, 0.0} and
      inner_surface_.sphericity != 1.0) {
    ERROR(
        "The sphericity of the inner surface must be exactly 1.0 if the "
        "centers of the inner and outer object are different.\ninner center: "
        << inner_surface_.center
        << "\nouter center: " << outer_surface_.center);
  }
  for (size_t i = 0; i < inner_surface_.center.size(); i++) {
    if (abs(gsl::at(projection_center_, i)) >=
        outer_surface_.half_cube_length) {
      ERROR("The inner surface center "
            << inner_surface_.center
            << " must be within the outer surface with center "
            << outer_surface_.center << " and half cube length "
            << outer_surface_.half_cube_length);
    }
  }
  if (axis_ == Axis::None) {
    ERROR(
        "Axis for Wedge shape map transition function cannot be 'None'. Please "
        "choose another.");
  }
  if (axis_ == Axis::Interior and
      (inner_surface_.sphericity != 1.0 or reverse)) {
    ERROR(
        "When the axis is 'Interior', the inner surface must be a sphere "
        "(sphericity 1.0) and reverse must be 'false'.");
  }
}

template <typename T>
std::array<T, 3> Wedge::compute_inner_surface_vector(
    const std::array<T, 3>& centered_coords, const T& centered_coords_magnitude,
    const std::optional<Axis>& potential_axis) const {
  // If the surface is a sphere, then the vector is trivially the radial vector
  if (inner_surface_.sphericity == 1.0) {
    return inner_surface_.radius * centered_coords / centered_coords_magnitude;
  }

  T non_spherical_contribution =
      (1.0 - inner_surface_.sphericity) / sqrt(3.0) * centered_coords_magnitude;
  // Do some debug checks so we don't divide by zero below if we are given an
  // axis
  if (potential_axis.has_value()) {
    const size_t index = axis_index(potential_axis.value());
#ifdef SPECTRE_DEBUG
    for (size_t i = 0; i < get_size(centered_coords[0]); i++) {
      if (get_element(gsl::at(centered_coords, index), i) == 0.0) {
        ERROR(
            "The Wedge transition map was called with coordinates outside of "
            "its wedge. The "
            << index << "-coordinate shouldn't be 0 but it is. Coordinate: ("
            << get_element(centered_coords[0], i) << ","
            << get_element(centered_coords[1], i) << ","
            << get_element(centered_coords[2], i) << ")");
      }
    }
#endif
    non_spherical_contribution /= abs(gsl::at(centered_coords, index));
  } else {
    non_spherical_contribution /=
        blaze::max(abs(centered_coords[0]), abs(centered_coords[1]),
                   abs(centered_coords[2]));
  }

  return inner_surface_.radius *
         (non_spherical_contribution + inner_surface_.sphericity) *
         centered_coords / centered_coords_magnitude;
}

template <typename T>
std::array<T, 3> Wedge::compute_outer_surface_vector(
    const std::array<T, 3>& centered_coords, const T& lambda) const {
  return lambda * centered_coords;
}

template <typename T>
T Wedge::lambda_cube(const std::array<T, 3>& centered_coords,
                     const std::optional<Axis>& potential_axis) const {
  const auto lambda_axis = [&centered_coords, &potential_axis,
                            this](const Axis axis) -> T {
    const size_t axis_idx = axis_index(axis);
    const double factor = axis_sgn(axis) * outer_surface_.half_cube_length -
                          gsl::at(projection_center_, axis_idx);
    T result{};
    // If we were given an axis to this function, then we can be confident that
    // all the coords of this axis aren't zero, so we don't have to check them
    if (potential_axis.has_value()) {
      result = factor / gsl::at(centered_coords, axis_idx);
    } else {
      // However, if we have to figure out the axis, it's possible some coords
      // for this axis are zero but some aren't so we have to check them all
      // individually to avoid dividing by zero. If a coordinate is zero, then
      // that point can't exist in the wedge whose coordinate is zero. Since we
      // take the min of all positive lambdas to find the correct one, we
      // overwrite this lambda with -max so it'll never be chosen
      result = make_with_value<T>(centered_coords[0], factor);
      for (size_t i = 0; i < get_size(result); i++) {
        if (get_element(gsl::at(centered_coords, axis_idx), i) == 0.0) {
          get_element(result, i) = -std::numeric_limits<double>::max();
        } else {
          get_element(result, i) /=
              get_element(gsl::at(centered_coords, axis_idx), i);
        }
      }
    }

    return result;
  };

  // If we specified an axis that we are on, then use that lambda. This saves
  // some computation.
  if (potential_axis.has_value()) {
    return lambda_axis(potential_axis.value());
  }

  // Otherwise we need to loop over all of them
  const std::array all_axes{Axis::PlusZ,  Axis::MinusZ, Axis::PlusY,
                            Axis::MinusY, Axis::PlusX,  Axis::MinusX};
  std::array<T, 6> candidate_lambdas{};

  for (size_t i = 0; i < all_axes.size(); i++) {
    gsl::at(candidate_lambdas, i) = lambda_axis(gsl::at(all_axes, i));
  }

  // We need to take the min of all the positive lambdas
  const double fill_value = std::numeric_limits<double>::max();
  T result = make_with_value<T>(candidate_lambdas[0], fill_value);
  std::vector<Axis> chosen_axes(get_size(result), Axis::None);
  for (size_t i = 0; i < get_size(result); i++) {
    for (size_t j = 0; j < candidate_lambdas.size(); j++) {
      if (get_element(gsl::at(candidate_lambdas, j), i) > 0.0 and
          get_element(gsl::at(candidate_lambdas, j), i) <
              get_element(result, i)) {
        get_element(result, i) = get_element(gsl::at(candidate_lambdas, j), i);
        chosen_axes[i] = gsl::at(all_axes, j);
      }
    }
  }

  using ::operator<<;
  ASSERT(not alg::any_of(chosen_axes,
                         [&](const Axis& a) { return a == Axis::None; }),
         "No candidate cube lambdas in the Wedge shape map transition function "
         "are positive.\nCandidate lambdas: "
             << candidate_lambdas
             << "\nThis is an internal error. Please file an issue.");

  return result;
}

template <typename T>
T Wedge::lambda_sphere(const std::array<T, 3>& centered_coords,
                       const T& centered_coords_magnitude) const {
  // quadratic equation is
  // a lambda^2 + b lambda + c = 0
  //
  // For this case
  // a = |x^i-P^i|^2
  // b = 2(x^i-P^i)(P^j-C^j)\delta_{ij}
  // c = |P^i-C^i|^2 - R^2

  const T a = square(centered_coords_magnitude);
  const T b = 2.0 * dot(centered_coords, projection_center_);
  const T c =
      make_with_value<T>(b, dot(projection_center_, projection_center_) -
                                square(outer_surface_.radius));

  CAPTURE_FOR_ERROR(a);
  CAPTURE_FOR_ERROR(b);
  CAPTURE_FOR_ERROR(c);

  // The root that we are looking for is positive (since a negative root would
  // be for the opposite side of the sphere). The epsilon is to handle slightly
  // negative values.
  return smallest_root_greater_than_value_within_roundoff(
      a, b, c, -std::numeric_limits<double>::epsilon() * 100.0);
}

template <typename T>
T Wedge::compute_lambda(const std::array<T, 3>& centered_coords,
                        const T& centered_coords_magnitude,
                        const std::optional<Axis>& potential_axis) const {
  T result{};
  // Avoid extra computation if possible
  if (outer_surface_.sphericity == 1.0) {
    result = lambda_sphere(centered_coords, centered_coords_magnitude);
  } else if (outer_surface_.sphericity == 0.0) {
    result = lambda_cube(centered_coords, potential_axis);
  } else {
    result = (1.0 - outer_surface_.sphericity) *
                 lambda_cube(centered_coords, potential_axis) +
             outer_surface_.sphericity *
                 lambda_sphere(centered_coords, centered_coords_magnitude);
  }

  return result;
}

template <typename T>
std::array<T, 3> Wedge::lambda_cube_gradient(
    const T& lambda_cube, const std::array<T, 3>& centered_coords) const {
  const size_t axis_idx = axis_index(axis_);

  // We don't have to worry about dividing by zero here because we only call the
  // gradient when we know our axis and so the coord of this axis can't be zero.
  std::array<T, 3> result =
      make_array<3>(make_with_value<T>(centered_coords[0], 0.0));
  gsl::at(result, axis_idx) = -lambda_cube / gsl::at(centered_coords, axis_idx);

  return result;
}

template <typename T>
std::array<T, 3> Wedge::lambda_sphere_gradient(
    const T& lambda_sphere, const std::array<T, 3>& centered_coords,
    const T& centered_coords_magnitude) const {
  return -1.0 * lambda_sphere *
         (lambda_sphere * centered_coords + projection_center_) /
         (lambda_sphere * square(centered_coords_magnitude) +
          dot(centered_coords, projection_center_));
}

template <typename T>
std::array<T, 3> Wedge::compute_lambda_gradient(
    const std::array<T, 3>& centered_coords,
    const T& centered_coords_magnitude) const {
  std::array<T, 3> result{};
  // Avoid extra computation if possible
  if (outer_surface_.sphericity == 1.0) {
    result = lambda_sphere_gradient(
        lambda_sphere(centered_coords, centered_coords_magnitude),
        centered_coords, centered_coords_magnitude);
  } else if (outer_surface_.sphericity == 0.0) {
    result = lambda_cube_gradient(lambda_cube(centered_coords, {axis_}),
                                  centered_coords);
  } else {
    result = (1.0 - outer_surface_.sphericity) *
                 lambda_cube_gradient(lambda_cube(centered_coords, {axis_}),
                                      centered_coords) +
             outer_surface_.sphericity *
                 lambda_sphere_gradient(
                     lambda_sphere(centered_coords, centered_coords_magnitude),
                     centered_coords, centered_coords_magnitude);
  }

  return result;
}

double Wedge::operator()(
    const std::array<double, 3>& source_coords,
    const std::optional<size_t>& one_over_radius_power) const {
  return call_impl<double>(source_coords, one_over_radius_power);
}

DataVector Wedge::operator()(
    const std::array<DataVector, 3>& source_coords,
    const std::optional<size_t>& one_over_radius_power) const {
  return call_impl<DataVector>(source_coords, one_over_radius_power);
}

template <typename T>
T Wedge::call_impl(const std::array<T, 3>& source_coords,
                   const std::optional<size_t>& one_over_radius_power) const {
  // The source coords are centered
  const T centered_coords_magnitude = magnitude(source_coords);

#ifdef SPECTRE_DEBUG
  using ::operator<<;
  if (UNLIKELY(one_over_radius_power.value_or(0_st) >= 3)) {
    for (size_t i = 0; i < get_size(source_coords[0]); i++) {
      if (UNLIKELY(equal_within_roundoff(
              get_element(centered_coords_magnitude, i), 0.0))) {
        const std::array<double, 3> point{get_element(source_coords[0], i),
                                          get_element(source_coords[1], i),
                                          get_element(source_coords[2], i)};
        ERROR("Trying to divide by a point "
              << point << " with radius zero in Wedge transition operator.");
      }
    }
  }
#endif

  // If the axis is the interior, short circuit
  if (axis_ == Axis::Interior) {
    return 1.0 / cube(inner_surface_.radius) *
           (one_over_radius_power.value_or(0_st) < 3
                ? T{integer_pow(centered_coords_magnitude,
                                static_cast<int>(
                                    2 - one_over_radius_power.value_or(0_st)))}
                : 1.0 / integer_pow(centered_coords_magnitude,
                                    static_cast<int>(
                                        one_over_radius_power.value() - 2)));
  }

  // First check if we are at the inner center to avoid dividing by zero below
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(source_coords[0]); i++) {
    if (UNLIKELY(equal_within_roundoff(
            get_element(centered_coords_magnitude, i), 0.0))) {
      const std::array<double, 3> point{get_element(source_coords[0], i),
                                        get_element(source_coords[1], i),
                                        get_element(source_coords[2], i)};
      ERROR(
          "The Wedge transition was called with a point that has zero "
          "centered "
          "radius, but the axis isn't the 'Interior'. Point is "
          << point << ", axis is " << axis_);
    }
  }
#endif

  // Now we can safely compute lambda without worrying about dividing by zero
  const T lambda =
      compute_lambda(source_coords, centered_coords_magnitude, {axis_});

  const T inner_distance =
      inner_surface_.sphericity == 1.0
          ? make_with_value<T>(centered_coords_magnitude, inner_surface_.radius)
          : magnitude(compute_inner_surface_vector(
                source_coords, centered_coords_magnitude, {axis_}));

  const std::array<T, 3> outer_surface_vector =
      compute_outer_surface_vector(source_coords, lambda);
  const T outer_distance = magnitude(outer_surface_vector);

  // Check if the point we were passed is within the transition region
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(source_coords[0]); i++) {
    const std::array<double, 3> point{get_element(source_coords[0], i),
                                      get_element(source_coords[1], i),
                                      get_element(source_coords[2], i)};
    if (get_element(centered_coords_magnitude, i) <
        (1.0 - eps_) * get_element(inner_distance, i)) {
      ERROR("Wedge transition called with centered coordinate "
            << point << " (with radius "
            << get_element(centered_coords_magnitude, i)
            << ") which is inside the inner surface ("
            << get_element(inner_distance, i)
            << ") while the axis was not 'Interior' (" << axis_ << ")");
    } else if (get_element(centered_coords_magnitude, i) >
               (1.0 + eps_) * get_element(outer_distance, i)) {
      ERROR("Wedge transition called with centered coordinate "
            << point << " (with radius "
            << get_element(centered_coords_magnitude, i)
            << ") which is outside the outer surface ("
            << get_element(outer_distance, i) << ").");
    }
  }
#endif

  T linear_transition_func = (outer_distance - centered_coords_magnitude) /
                             (outer_distance - inner_distance);

  if (reverse_) {
    linear_transition_func = 1.0 - linear_transition_func;
  }

  // Accounts for roundoff
  linear_transition_func = blaze::clamp(linear_transition_func, 0.0, 1.0);

  return linear_transition_func /
         integer_pow(
             centered_coords_magnitude,
             static_cast<int>(1 + one_over_radius_power.value_or(0_st)));
}

std::optional<double> Wedge::original_radius_over_radius(
    const std::array<double, 3>& target_coords,
    double radial_distortion) const {
  // The target coords are centered
  const double centered_coords_magnitude = magnitude(target_coords);
  CAPTURE_FOR_ERROR(target_coords);
  CAPTURE_FOR_ERROR(radial_distortion);
  CAPTURE_FOR_ERROR(centered_coords_magnitude);

  // There are a number of different special cases that can (and should) be
  // accounted for before we go to the general inverse formula. To denote all
  // these special cases we introduce the following notation. For the 3 possible
  // regions, we use Int (Interior), Ext (Exterior), and Mid
  // (middle/transition). For the boundaries between these regions we'll combine
  // them: IntMid (inner surface) and MidExt (outer surface). Then for reverse,
  // we'll add 'R+' (e.g. R+Ext accounts for a point in the exterior region when
  // the transition is reversed). The absence of 'R+' means this is the
  // "regular" transition.

  // Int: Protect against point that wouldn't have any radial distortion
  if (equal_within_roundoff(centered_coords_magnitude, 0.0)) {
    return (axis_ == Axis::Interior and not reverse_) ? std::optional{1.0}
                                                      : std::nullopt;
  }

  const double inner_distance =
      inner_surface_.sphericity == 1.0
          ? inner_surface_.radius
          : magnitude(compute_inner_surface_vector(
                target_coords, centered_coords_magnitude, std::nullopt));

  if (not reverse_) {
    // Check if we are within the interior region or at the inner surface. The
    // formula is simplified at the inner surface
    if (centered_coords_magnitude + radial_distortion <
        (1.0 - eps_) * inner_distance) {
      // Int: Unless the axis_ is Interior, this block doesn't contain this
      // point
      if (axis_ != Axis::Interior) {
        return std::nullopt;
      }

      const double factor =
          cube(inner_surface_.radius) /
          (square(centered_coords_magnitude) * radial_distortion);

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
    } else if (equal_within_roundoff(
                   centered_coords_magnitude + radial_distortion,
                   inner_distance)) {
      // IntMid: Since the point is on the boundary, we can't tell if this is
      // the interior or not, but either map will work.
      return std::optional{inner_distance /
                           (inner_distance - radial_distortion)};
    }
  }

  const double lambda =
      compute_lambda(target_coords, centered_coords_magnitude, std::nullopt);

  const double outer_distance =
      magnitude(compute_outer_surface_vector(target_coords, lambda));

  // Ext,R+Int: First we check the extremal cases where we are outside the
  // transition region. If we are reversed and inside the inner surface, we
  // return nullopt. If we aren't reversed and beyond the outer surface we
  // return nullopt because we could still be in the interior region.
  if ((not reverse_ and
       centered_coords_magnitude > (1.0 + eps_) * outer_distance) or
      (reverse_ and
       centered_coords_magnitude < (1.0 - eps_) * inner_distance)) {
    return std::nullopt;
  }

  // All remaining cases: If distorted radius is 0, this means the map is the
  // identity so the radius and the original radius are equal. Also we don't
  // want to divide by 0 below. We do this check after we check if the point is
  // beyond the outer distance because if a point is outside the distorted
  // frame, this function should return nullopt.
  if (equal_within_roundoff(radial_distortion, 0.0)) {
    return std::optional{1.0};
  }

  // MidExt,R+IntMid: If we are at the overall outer distance, then the
  // transition function is 0 so the map is again the identity so the radius and
  // original radius are equal. We can't check the overall inner distance
  // because that has been distorted so we don't know where the mapped inner
  // distance is. This logic is reversed if we are in reverse mode.
  if ((not reverse_ and
       equal_within_roundoff(centered_coords_magnitude, outer_distance)) or
      (reverse_ and
       equal_within_roundoff(centered_coords_magnitude, inner_distance))) {
    return std::optional{1.0};
  }

  const double denom = outer_distance - inner_distance +
                       (reverse_ ? -1.0 : 1.0) * radial_distortion;
  // prevent zero division
  if (equal_within_roundoff(denom, 0.)) {
    return std::nullopt;
  }

  const double original_radius_over_radius =
      (outer_distance - inner_distance +
       (reverse_ ? -inner_distance : outer_distance) * radial_distortion /
           centered_coords_magnitude) /
      denom;
  const double original_radius =
      original_radius_over_radius * centered_coords_magnitude;

  // R+MidExt,R+Ext: Extremal transition at outer boundary for reverse has a
  // simplified formula.
  if (reverse_) {
    if (equal_within_roundoff(original_radius, outer_distance)) {
      return {1.0 + radial_distortion / centered_coords_magnitude};
    } else if (original_radius > (1.0 + eps_) * outer_distance) {
      return std::nullopt;
    }
  }

  // Mid,R+Mid
  return {original_radius_over_radius};
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
std::array<T, 3> Wedge::gradient_impl(
    const std::array<T, 3>& source_coords) const {
  // The source coords are centered
  const T centered_coords_magnitude = magnitude(source_coords);

  // Short circuit if we are in the interior
  if (axis_ == Axis::Interior) {
    return 2.0 / cube(inner_surface_.radius) * source_coords;
  }

// First check if we are at the inner center to avoid dividing by zero below
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(centered_coords_magnitude); i++) {
    if (UNLIKELY(equal_within_roundoff(
            get_element(centered_coords_magnitude, i), 0.0))) {
      ERROR(
          "The gradient of the wedge transition was called with a point that "
          "has zero centered radius, but the axis isn't the 'Interior'. Point "
          "is "
          << inner_surface_.center << ", axis is " << axis_);
    }
  }
#endif

  const T one_over_centered_coords_magnitude = 1.0 / centered_coords_magnitude;

  const T lambda =
      compute_lambda(source_coords, centered_coords_magnitude, {axis_});

  const T inner_distance =
      inner_surface_.sphericity == 1.0
          ? make_with_value<T>(lambda, inner_surface_.radius)
          : magnitude(compute_inner_surface_vector(
                source_coords, centered_coords_magnitude, {axis_}));

  const std::array<T, 3> outer_surface_vector =
      compute_outer_surface_vector(source_coords, lambda);
  const T outer_distance = magnitude(outer_surface_vector);

// Check if the point we were passed is within the transition region
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(centered_coords_magnitude); i++) {
    const std::array<double, 3> point{get_element(source_coords[0], i),
                                      get_element(source_coords[1], i),
                                      get_element(source_coords[2], i)};
    if (get_element(centered_coords_magnitude, i) <
        (1.0 - eps_) * get_element(inner_distance, i)) {
      ERROR("Wedge transition called with centered coordinate "
            << point << " (with radius "
            << get_element(centered_coords_magnitude, i)
            << ") which is inside the inner surface ("
            << get_element(inner_distance, i)
            << ") while the axis was not 'Interior' (" << axis_ << ")");
    } else if (get_element(centered_coords_magnitude, i) >
               (1.0 + eps_) * get_element(outer_distance, i)) {
      ERROR("Wedge transition called with centered coordinate "
            << point << " (with radius "
            << get_element(centered_coords_magnitude, i)
            << ") which is outside the outer surface ("
            << get_element(outer_distance, i) << ").");
    }
  }
#endif

  // This can only be called if the projection center is 0, otherwise this
  // formula won't work. And if the projection center isn't 0, then we require
  // that the sphericity of the inner surface is 1, so we ASSERT that here
  // as well
  const auto inner_surface_gradient = [&]() -> std::array<T, 3> {
    using ::operator<<;
    ASSERT((projection_center_ == std::array{0.0, 0.0, 0.0}),
           "Should not be calculating the inner surface gradient when the "
           "projection center is non-zero. "
               << projection_center_);
    ASSERT(inner_surface_.sphericity != 1.0,
           "Should not be calculating the inner surface gradient when its "
           "sphericity is 1.0 because the gradient is exactly 0.");

    const size_t axis_idx = axis_index(axis_);
    const size_t axis_plus_one = (axis_idx + 1) % 3;
    const size_t axis_plus_two = (axis_idx + 2) % 3;

    const T& axis_coord = gsl::at(source_coords, axis_idx);

    std::array<T, 3> grad = make_array<3, T>(
        inner_surface_.radius * (1.0 - inner_surface_.sphericity) / sqrt(3.0) *
        one_over_centered_coords_magnitude / abs(axis_coord));

    // Dividing by axis_coord here takes care of the sgn(axis_coord) that would
    // have been necessary
    gsl::at(grad, axis_idx) *=
        -(square(centered_coords_magnitude) - square(axis_coord)) / axis_coord;
    gsl::at(grad, axis_plus_one) *= gsl::at(source_coords, axis_plus_one);
    gsl::at(grad, axis_plus_two) *= gsl::at(source_coords, axis_plus_two);

    return grad;
  };

  // We always need to compute the outer gradient regardless of the projection
  // center or the outer sphericity
  const std::array<T, 3> outer_gradient =
      lambda * source_coords * one_over_centered_coords_magnitude +
      compute_lambda_gradient(source_coords, centered_coords_magnitude) *
          centered_coords_magnitude;

  const T one_over_distance_difference =
      1.0 / (outer_distance - inner_distance);

  // Avoid allocating an array of 0 if the inner surface is a sphere
  if (inner_surface_.sphericity == 1.0) {
    return (outer_gradient -
            source_coords * one_over_centered_coords_magnitude -
            (outer_distance - centered_coords_magnitude) * outer_gradient *
                one_over_distance_difference) *
               one_over_distance_difference *
               one_over_centered_coords_magnitude * (reverse_ ? -1.0 : 1.0) -
           source_coords * (*this)(source_coords, {2});
  } else {
    const std::array<T, 3> inner_gradient = inner_surface_gradient();

    return (outer_gradient -
            source_coords * one_over_centered_coords_magnitude -
            (outer_distance - centered_coords_magnitude) *
                (outer_gradient - inner_gradient) *
                one_over_distance_difference) *
               one_over_distance_difference *
               one_over_centered_coords_magnitude * (reverse_ ? -1.0 : 1.0) -
           source_coords * (*this)(source_coords, {2});
  }
}

bool Wedge::operator==(const ShapeMapTransitionFunction& other) const {
  if (dynamic_cast<const Wedge*>(&other) == nullptr) {
    return false;
  }
  const auto surface_equal = [](const Surface& s1, const Surface& s2) {
    return s1.center == s2.center and s1.radius == s2.radius and
           s1.sphericity == s2.sphericity and
           s1.half_cube_length == s2.half_cube_length;
  };
  const Wedge& other_ref = *dynamic_cast<const Wedge*>(&other);
  return surface_equal(inner_surface_, other_ref.inner_surface_) and
         surface_equal(outer_surface_, other_ref.outer_surface_) and
         axis_ == other_ref.axis_ and reverse_ == other_ref.reverse_;
}

bool Wedge::operator!=(const ShapeMapTransitionFunction& other) const {
  return not(*this == other);
}

namespace {
// struct to allow unpacking version 0 of this class
struct LegacySurface {
  double radius{std::numeric_limits<double>::signaling_NaN()};
  double sphericity{std::numeric_limits<double>::signaling_NaN()};

  LegacySurface() = default;
  LegacySurface(const double radius_in, const double sphericity_in)
      : radius(radius_in), sphericity(sphericity_in) {}

  void pup(PUP::er& p) {
    p | radius;
    p | sphericity;
  }
};
}  // namespace

void Wedge::pup(PUP::er& p) {
  ShapeMapTransitionFunction::pup(p);
  size_t version = 2;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 1) {
    p | inner_surface_;
    p | outer_surface_;
    p | projection_center_;
    p | axis_;
    if (version >= 2) {
      p | reverse_;
    } else {
      reverse_ = false;
    }
  } else if (p.isUnpacking()) {
    LegacySurface inner_surface{};
    LegacySurface outer_surface{};
    size_t axis = 0;
    p | inner_surface;
    p | outer_surface;
    p | axis;

    // We don't know the centers of the objects so we just set them to 0
    inner_surface_ = Surface{
        {0.0, 0.0, 0.0},
        inner_surface.radius,
        inner_surface.sphericity,
    };
    outer_surface_ = Surface{
        {0.0, 0.0, 0.0},
        outer_surface.radius,
        outer_surface.sphericity,
    };
    // Unfortunately we can't recover the sign for the axis so we just set it
    // to positive. This is wrong though.
    axis_ = static_cast<Axis>(axis + 1);
    projection_center_ = std::array{0.0, 0.0, 0.0};
  }
}

Wedge::Wedge(CkMigrateMessage* const msg) : ShapeMapTransitionFunction(msg) {}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID Wedge::my_PUP_ID = 0;
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
