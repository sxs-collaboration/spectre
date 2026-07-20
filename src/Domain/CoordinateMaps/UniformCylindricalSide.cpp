// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <sstream>
#include <utility>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace domain::CoordinateMaps {

namespace {
// FunctionEvalTempVariables exists to remove code duplication between
// function_to_zero and deriv_function_to_zero.
struct FunctionEvalTempVariables {
  FunctionEvalTempVariables(double lambda,
                            const std::array<double, 3>& target_coords,
                            const std::array<double, 3>& center_one,
                            const std::array<double, 3>& center_two,
                            double radius_one, double radius_two,
                            double z_plane_minus_one, double z_plane_minus_two,
                            double z_plane_plus_one, double z_plane_plus_two);
  double one_plus_zbar_over_two;
  double r_one_cos_theta_one;
  double r_two_cos_theta_two;
  double r_one_sin_theta_one;
  double r_two_sin_theta_two;
};

FunctionEvalTempVariables::FunctionEvalTempVariables(
    const double lambda, const std::array<double, 3>& target_coords,
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, const double radius_one,
    const double radius_two, const double z_plane_minus_one,
    const double z_plane_minus_two, const double z_plane_plus_one,
    const double z_plane_plus_two)
    : one_plus_zbar_over_two(
          (target_coords[2] + lambda * (z_plane_minus_one - z_plane_minus_two) -
           z_plane_minus_one) /
          ((1.0 - lambda) * (z_plane_plus_one - z_plane_minus_one) +
           lambda * (z_plane_plus_two - z_plane_minus_two))),
      r_one_cos_theta_one(z_plane_minus_one - center_one[2] +
                          (z_plane_plus_one - z_plane_minus_one) *
                              one_plus_zbar_over_two),
      r_two_cos_theta_two(z_plane_minus_two - center_two[2] +
                          (z_plane_plus_two - z_plane_minus_two) *
                              one_plus_zbar_over_two),
      r_one_sin_theta_one(
          sqrt(square(radius_one) - square(r_one_cos_theta_one))),
      r_two_sin_theta_two(
          sqrt(square(radius_two) - square(r_two_cos_theta_two))) {}
}  // namespace

UniformCylindricalSide::UniformCylindricalSide(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, double radius_one,
    double radius_two, double z_plane_plus_one, double z_plane_minus_one,
    double z_plane_plus_two, double z_plane_minus_two)
    : center_one_(center_one),
      center_two_(center_two),
      radius_one_([&radius_one]() {
        // The equal_within_roundoff below has an implicit scale of 1,
        // so the ASSERT may trigger in the case where we really
        // want an entire domain that is very small.
        ASSERT(not equal_within_roundoff(radius_one, 0.0),
               "Cannot have zero radius_one");
        ASSERT(radius_one > 0.0, "Cannot have negative radius_one");
        return radius_one;
                  }()), // LCOV_EXCL_LINE
      radius_two_([&radius_two]() {
        // The equal_within_roundoff below has an implicit scale of 1,
        // so the ASSERT may trigger in the case where we really
        // want an entire domain that is very small.
        ASSERT(not equal_within_roundoff(radius_two, 0.0),
               "Cannot have zero radius_two");
        ASSERT(radius_two > 0.0, "Cannot have negative radius_two");
        return radius_two;
      }()), // LCOV_EXCL_LINE
      z_plane_plus_one_(z_plane_plus_one),
      z_plane_minus_one_(z_plane_minus_one),
      z_plane_plus_two_(z_plane_plus_two),
      z_plane_minus_two_(z_plane_minus_two) {
  // Assumptions made in the map.  Some of these can be relaxed,
  // as long as the unit test is changed to test them.
  // The ASSERTS here match the ones in UniformCylindricalEndcap.
  ASSERT(radius_one >= 0.08 * radius_two,
         "Radius_one = " << radius_one << " must be >= 0.08 * radius_two ="
                         << 0.08 * radius_two);
  ASSERT(z_plane_plus_two == z_plane_plus_one or
             z_plane_plus_two >= z_plane_plus_one + 0.03 * radius_two,
         "z_plane_plus_two must be >= z_plane_plus_one "
         "+ 0.03 * radius_two, or exactly equal to z_plane_plus_one, not "
             << z_plane_plus_two << " " << z_plane_plus_one << " "
             << z_plane_plus_one + 0.03 * radius_two);
  ASSERT(z_plane_minus_two == z_plane_minus_one or
             abs(z_plane_minus_two - z_plane_minus_one) >= 0.03 * radius_two,
         "z_plane_minus_two must differ from z_plane_minus_one by at least "
         "0.03 * radius_two, or be exactly equal, but z_plane_minus_two = "
             << z_plane_minus_two
             << ", z_plane_minus_one = " << z_plane_minus_one
             << ", and 0.03 * radius_two = " << 0.03 * radius_two);
  ASSERT(z_plane_plus_two != z_plane_plus_one or
             z_plane_minus_two != z_plane_minus_one,
         "Both pairs of z-planes cannot be simultaneously equal: "
         "z_plane_plus_one="
             << z_plane_plus_one << ", z_plane_plus_two=" << z_plane_plus_two
             << ", z_plane_minus_one=" << z_plane_minus_one
             << ", z_plane_minus_two=" << z_plane_minus_two);
  // In Class B (z_plane_minus_two > z_plane_minus_one), the outer sphere's
  // mapped band must still be non-inverted.  In Class A this is guaranteed
  // implicitly by the theta bounds on sphere_two, but those bounds are relaxed
  // in Class B, so we check it explicitly here (runs in both debug and
  // release).
  if (z_plane_minus_two > z_plane_minus_one) {
    ASSERT(z_plane_plus_two > z_plane_minus_two,
           "In Class B (z_plane_minus_two="
               << z_plane_minus_two
               << " > z_plane_minus_one=" << z_plane_minus_one
               << "), the outer sphere's upper cut z_plane_plus_two="
               << z_plane_plus_two
               << " must be strictly above z_plane_minus_two.");
  }

  // The code below defines several variables that are used only in ASSERTs.
  // We put that code in a #ifdef SPECTRE_DEBUG to avoid clang-tidy complaining
  // about unused variables in release mode.
#ifdef SPECTRE_DEBUG
  const double cos_theta_min_one =
      (z_plane_plus_one - center_one[2]) / radius_one;
  const double cos_theta_max_one =
      (z_plane_minus_one - center_one[2]) / radius_one;
  const double cos_theta_min_two =
      (z_plane_plus_two - center_two[2]) / radius_two;
  const double cos_theta_max_two =
      (z_plane_minus_two - center_two[2]) / radius_two;

  // For some reason, codecov thinks that the following lambda never
  // gets called, even though it is used in all of the ASSERT messages
  // below.
  // LCOV_EXCL_START
  const auto param_string = [this]() -> std::string {
    std::ostringstream buffer;
    buffer << "\nParameters to UniformCylindricalSide:\nradius_one="
           << radius_one_ << "\nradius_two=" << radius_two_
           << "\ncenter_one=" << center_one_ << "\ncenter_two=" << center_two_
           << "\nz_plane_plus_one=" << z_plane_plus_one_
           << "\nz_plane_plus_two=" << z_plane_plus_two_
           << "\nz_plane_minus_one=" << z_plane_minus_one_
           << "\nz_plane_minus_two=" << z_plane_minus_two_;
    return buffer.str();
  };
  // LCOV_EXCL_STOP

  const double cos_theta_min_one_lower_limit =
      z_plane_plus_one_ == z_plane_plus_two_
          ? cos(M_PI * 0.59)
          : (z_plane_minus_one_ == z_plane_minus_two_
                 // Equal lower planes (no lower cone): the upper cut can reach
                 // or slightly pass the equator of sphere_one.
                 ? cos(M_PI * 0.55)
                 : cos(M_PI * 0.4));

  const double cos_theta_max_one_upper_limit =
      z_plane_minus_one_ == z_plane_minus_two_
          ? cos(M_PI * 0.41)
          : (z_plane_minus_two_ > z_plane_minus_one_
                 // Class B: outer lower cut is above inner lower cut.
                 // The inner sphere's lower cut can be at or above the equator,
                 // so use a more lenient limit (theta_max_one > 72 deg).
                 ? cos(M_PI * 0.4)
                 : (z_plane_plus_one_ == z_plane_plus_two_ and
                            cos_theta_min_one < cos(M_PI * 0.4)
                        ? cos(M_PI * 0.7)
                        : cos(M_PI * 0.6)));

  const double cos_theta_min_two_upper_limit =
      z_plane_plus_one_ == z_plane_plus_two_ and
              cos_theta_min_one < cos(M_PI * 0.4)
          ? cos(M_PI * 0.25)
          : cos(M_PI * 0.15);

  const double cos_theta_max_two_lower_limit =
      z_plane_minus_one_ == z_plane_minus_two_ and
              cos_theta_max_one > cos(M_PI * 0.6)
          ? cos(M_PI * 0.75)
          : cos(M_PI * 0.85);

  ASSERT(cos_theta_min_one > cos_theta_min_one_lower_limit,
         "z_plane_plus_one is too close to the center of sphere_one: "
         "cos_theta_min_one must be > "
             << cos_theta_min_one_lower_limit << " but is " << cos_theta_min_one
             << " instead." << param_string());
  ASSERT(cos_theta_max_one < cos_theta_max_one_upper_limit,
         "z_plane_minus_one is too close to the center of sphere_one: "
         "cos_theta_max_one must be < "
             << cos_theta_max_one_upper_limit << " but is " << cos_theta_max_one
             << " instead." << param_string());
  ASSERT(cos_theta_min_one < cos(M_PI * 0.15),
         "z_plane_plus_one is too far from the center of sphere_one: "
         "cos_theta_min_one must be < "
             << cos(M_PI * 0.15) << " but is " << cos_theta_min_one
             << " instead." << param_string());
  ASSERT(cos_theta_max_one > cos(M_PI * 0.85),
         "z_plane_minus_one is too far from the center of sphere_one: "
         "cos_theta_max_one must be > "
             << cos(M_PI * 0.85) << " but is " << cos_theta_max_one
             << " instead." << param_string());
  ASSERT(cos_theta_min_two > (z_plane_plus_one_ == z_plane_plus_two_
                                  ? cos(M_PI * 0.75)
                                  : cos(M_PI * 0.4)),
         "z_plane_plus_two is too close to the south pole: theta/pi="
             << acos(cos_theta_min_two) / M_PI << param_string());
  // In Class B (z_minus_two > z_minus_one) the lower cut of sphere_two lies
  // above sphere_one's lower cut, so theta_max_two can be less than pi/2
  // (cut above equator of sphere_two). Use the same loose upper bound as
  // is used for theta_min values (0.15*pi from the pole).
  ASSERT(cos_theta_max_two < (z_plane_minus_one_ == z_plane_minus_two_
                                  ? cos(M_PI * 0.25)
                                  : (z_plane_minus_two_ > z_plane_minus_one_
                                         ? cos(M_PI * 0.15)
                                         : cos(M_PI * 0.6))),
         "z_plane_minus_two is too close to the north pole: theta/pi="
             << acos(cos_theta_max_two) / M_PI << param_string());
  ASSERT(cos_theta_min_two < cos_theta_min_two_upper_limit,
         "z_plane_plus_two is too close to the north pole: theta_min_two/pi="
             << acos(cos_theta_min_two) / M_PI << " but it should be >"
             << acos(cos_theta_min_two_upper_limit) / M_PI << param_string());
  ASSERT(cos_theta_max_two > cos_theta_max_two_lower_limit,
         "z_plane_minus_two is too close to the south pole: theta_max_two/pi="
             << acos(cos_theta_max_two) / M_PI << " but it should be < "
             << acos(cos_theta_max_two_lower_limit) / M_PI << param_string());

  const double dist_spheres = sqrt(square(center_one[0] - center_two[0]) +
                                   square(center_one[1] - center_two[1]) +
                                   square(center_one[2] - center_two[2]));

  ASSERT(dist_spheres + radius_one <= 0.98 * radius_two,
         "The map has been tested only for the case when "
         "sphere_one is sufficiently contained inside sphere_two, without the "
         "two spheres almost touching. Radius_one = "
             << radius_one << ", radius_two = " << radius_two
             << ", dist_spheres = " << dist_spheres
             << ", (dist_spheres+radius_one)/radius_two="
             << (dist_spheres + radius_one) / radius_two << param_string());

  const double horizontal_dist_spheres =
      sqrt(square(center_one[0] - center_two[0]) +
           square(center_one[1] - center_two[1]));

  if (z_plane_plus_two != z_plane_plus_one and
      z_plane_minus_two != z_plane_minus_one) {
    ASSERT(center_one[2] - radius_one <= center_two[2] + 0.2 * radius_two and
               center_one[2] + radius_one >= center_two[2] - 0.2 * radius_two,
           "The map has been tested only for the case when "
           "sphere_one is not a tiny thing at the top or bottom of sphere_two. "
           " Radius_one = "
               << radius_one << ", radius_two = " << radius_two
               << " center_one[2] = " << center_one[2]
               << " center_two[2] = " << center_two[2] << param_string());
  }

  // When the plus planes are equal and the minus planes differ, the minus plane
  // of sphere_two must be far enough below the shared upper boundary.
  if (z_plane_plus_two == z_plane_plus_one and
      z_plane_minus_two != z_plane_minus_one) {
    ASSERT(z_plane_minus_two <= z_plane_plus_two - 0.18 * radius_two,
           "The map has been tested only if the sphere_two planes are far "
           "enough apart. z_plane_plus_two="
               << z_plane_plus_two << " z_plane_minus_two=" << z_plane_minus_two
               << " radius_two=" << radius_two << param_string());
  }

  // When only the minus planes are equal (plus planes differ), the plus plane
  // of sphere_two must be far enough above the shared lower boundary.
  if (z_plane_minus_two == z_plane_minus_one and
      z_plane_plus_two != z_plane_plus_one) {
    ASSERT(z_plane_plus_two >= z_plane_minus_two + 0.18 * radius_two,
           "The map has been tested only if the sphere_two planes are far "
           "enough apart. z_plane_plus_two="
               << z_plane_plus_two << " z_plane_minus_two=" << z_plane_minus_two
               << " radius_two=" << radius_two << param_string());
  }

  ASSERT(horizontal_dist_spheres <=
             (z_plane_plus_two == z_plane_plus_one or
                      z_plane_minus_two == z_plane_minus_one
                  ? 0.0
                  : std::min(radius_one,
                             std::max(0.0, 0.95 * radius_two - radius_one))),
         "The map has been tested only for the case when "
         "sphere_one intersects the polar axis of sphere_two and is not "
         "too far from the edge of sphere_two (or in the "
         "case of equal plus or minus z_planes, when the two spheres have the "
         "same x and y centers)."
         " Radius_one = "
             << radius_one << ", radius_two = " << radius_two
             << ", dist_spheres = " << dist_spheres
             << ", horizontal_dist_spheres = " << horizontal_dist_spheres
             << param_string());

  const double theta_min_one = acos(cos_theta_min_one);
  const double theta_max_one = acos(cos_theta_max_one);
  const double theta_min_two = acos(cos_theta_min_two);
  const double theta_max_two = acos(cos_theta_max_two);

  // max_horizontal_dist_between_circles can be either sign;
  // therefore alpha can be in either the first or second quadrant.
  const double max_horizontal_dist_between_circles_plus =
      horizontal_dist_spheres + radius_one_ * sin(theta_min_one) -
      radius_two_ * sin(theta_min_two);
  const double max_horizontal_dist_between_circles_minus =
      horizontal_dist_spheres + radius_one_ * sin(theta_max_one) -
      radius_two_ * sin(theta_max_two);

  const double alpha_plus = atan2(z_plane_plus_two - z_plane_plus_one,
                                  max_horizontal_dist_between_circles_plus);
  // When z_plane_plus_one == z_plane_plus_two the two spheres share the same
  // upper z boundary; alpha_plus = 0 by definition and the cone-angle check
  // is vacuous.
  if (z_plane_plus_one_ != z_plane_plus_two_) {
    ASSERT(
        alpha_plus > 1.1 * theta_min_one and alpha_plus > 1.1 * theta_min_two,
        "Angle alpha_plus is too small: alpha_plus = "
            << alpha_plus << ", theta_min_one = " << theta_min_one
            << ", theta_min_two = " << theta_min_two
            << ", max_horizontal_dist_between_circles_plus = "
            << max_horizontal_dist_between_circles_plus
            << ", horizontal_dist_spheres = " << horizontal_dist_spheres
            << param_string());
  }

  // Use abs so alpha_minus is positive regardless of which class:
  // Class A (z_minus_two < z_minus_one): lower boundary goes downward.
  // Class B (z_minus_two > z_minus_one): lower boundary goes upward.
  // The tangent angle at S^-_1 is the same in both cases, so the same
  // formula applies for sphere_one. For sphere_two in Class B, the line
  // starts outside sphere_two and ends at S^-_2 (bottom of its mapped region);
  // by convexity of sphere_two the segment stays outside sphere_two, so the
  // sphere_two condition is automatically satisfied and need not be checked.
  const double alpha_minus = atan2(abs(z_plane_minus_one - z_plane_minus_two),
                                   max_horizontal_dist_between_circles_minus);
  // When z_plane_minus_one == z_plane_minus_two the two spheres share the same
  // lower z boundary; alpha_minus = 0 by definition and the cone-angle check
  // is vacuous.
  if (z_plane_minus_one_ != z_plane_minus_two_) {
    if (z_plane_minus_two_ < z_plane_minus_one_) {
      // Class A: lower boundary descends; sphere_two condition applies.
      ASSERT(alpha_minus > 1.1 * (M_PI - theta_max_one) and
                 alpha_minus > 1.1 * (M_PI - theta_max_two),
             "Angle alpha_minus is too small: alpha_minus = "
                 << alpha_minus << ", theta_max_one = " << theta_max_one
                 << ", theta_max_two = " << theta_max_two
                 << ", max_horizontal_dist_between_circles_minus = "
                 << max_horizontal_dist_between_circles_minus
                 << ", horizontal_dist_spheres = " << horizontal_dist_spheres
                 << param_string());
    } else {
      // Class B: lower boundary ascends; only sphere_one condition needed.
      ASSERT(alpha_minus > 1.1 * (M_PI - theta_max_one),
             "Angle alpha_minus is too small: alpha_minus = "
                 << alpha_minus << ", theta_max_one = " << theta_max_one
                 << ", max_horizontal_dist_between_circles_minus = "
                 << max_horizontal_dist_between_circles_minus
                 << ", horizontal_dist_spheres = " << horizontal_dist_spheres
                 << param_string());
    }
  }
#endif
}

template <typename T>
std::array<T, 3> UniformCylindricalSide::operator()(
    const std::array<T, 3>& source_coords) const {
  const T& xbar = source_coords[0];
  const T& ybar = source_coords[1];
  const T& zbar = source_coords[2];
  std::array<T, 3> target_coords{};
  T& x = target_coords[0];
  T& y = target_coords[1];
  T& z = target_coords[2];

  // Use y and z as temporary storage to avoid allocations,
  // before setting them to their actual values.
  z = sqrt(square(xbar) + square(ybar));  // z is rhobar = lambda+1 in the dox
  // y is temporarily R1 sin(theta_1)(1-lambda) + R2 sin(theta_2) lambda
  // Here R1 sin(theta_1) is calculated from R1 cos(theta_1) as
  // expressed in the dox
  y = sqrt(square(radius_two_) -
           square(z_plane_minus_two_ - center_two_[2] +
                  0.5 * (zbar + 1.0) *
                      (z_plane_plus_two_ - z_plane_minus_two_))) *
          (z - 1.0) +
      sqrt(square(radius_one_) -
           square(z_plane_minus_one_ - center_one_[2] +
                  0.5 * (zbar + 1.0) *
                      (z_plane_plus_one_ - z_plane_minus_one_))) *
          (2.0 - z);
  // Now compute actual x,y,z.
  x = y * xbar / z + center_two_[0] * (z - 1.0) + center_one_[0] * (2.0 - z);
  y = y * ybar / z + center_two_[1] * (z - 1.0) + center_one_[1] * (2.0 - z);
  z = (z_plane_minus_two_ +
       0.5 * (zbar + 1.0) * (z_plane_plus_two_ - z_plane_minus_two_)) *
          (z - 1.0) +
      (z_plane_minus_one_ +
       0.5 * (zbar + 1.0) * (z_plane_plus_one_ - z_plane_minus_one_)) *
          (2.0 - z);
  return target_coords;
}

std::optional<std::array<double, 3>> UniformCylindricalSide::inverse(
    const std::array<double, 3>& target_coords) const {
  // First do some easy checks that target_coords is in the range
  // of the map.  Need to accept points that are out of range by roundoff.

  // check z <= z_plane_plus_two_
  if (target_coords[2] > z_plane_plus_two_ and
      not equal_within_roundoff(target_coords[2], z_plane_plus_two_)) {
    return std::nullopt;
  }

  // check z >= lower z bound.
  // Class A (z_minus_two < z_minus_one): lower bound is z_minus_two.
  // Class B (z_minus_two > z_minus_one): lower bound is z_minus_one.
  const double z_lower_bound = std::min(z_plane_minus_one_, z_plane_minus_two_);
  if (target_coords[2] < z_lower_bound and
      not equal_within_roundoff(target_coords[2], z_lower_bound)) {
    return std::nullopt;
  }

  // check that point is in or on sphere_two
  const double r_minus_center_two =
      sqrt(square(target_coords[0] - center_two_[0]) +
           square(target_coords[1] - center_two_[1]) +
           square(target_coords[2] - center_two_[2]));
  if (r_minus_center_two > radius_two_ and
      not equal_within_roundoff(r_minus_center_two, radius_two_)) {
    return std::nullopt;
  }

  // check that point is outside or on sphere_one_
  const double r_minus_center_one =
      sqrt(square(target_coords[0] - center_one_[0]) +
           square(target_coords[1] - center_one_[1]) +
           square(target_coords[2] - center_one_[2]));
  if (r_minus_center_one < radius_one_ and
      not equal_within_roundoff(r_minus_center_one, radius_one_)) {
    return std::nullopt;
  }

  // Shared geometry computation for cone checks.
  const auto cone_rho_and_radius = [this, &target_coords](
                                       const double z_one, const double z_two,
                                       double& rho_tilde,
                                       double& circle_radius) {
    const double lambda_tilde = (target_coords[2] - z_one) / (z_two - z_one);
    rho_tilde = sqrt(square(target_coords[0] - center_one_[0] -
                            lambda_tilde * (center_two_[0] - center_one_[0])) +
                     square(target_coords[1] - center_one_[1] -
                            lambda_tilde * (center_two_[1] - center_one_[1])));
    circle_radius = sqrt(square(radius_one_) - square(z_one - center_one_[2])) *
                        (1.0 - lambda_tilde) +
                    sqrt(square(radius_two_) - square(z_two - center_two_[2])) *
                        lambda_tilde;
  };

  // Returns true if the point is strictly inside the cone (rho < radius).
  // Used in Class A: points inside the lower cone are below the lower boundary.
  auto point_inside_cone =
      [this, &cone_rho_and_radius](const double z_one, const double z_two) {
        double rho_tilde = 0.0;
        double circle_radius = 0.0;
        cone_rho_and_radius(z_one, z_two, rho_tilde, circle_radius);
        return rho_tilde < circle_radius and
               not equal_within_roundoff(
                   rho_tilde, circle_radius,
                   std::numeric_limits<double>::epsilon() * 100.0, radius_two_);
      };

  // Returns true if the point is strictly outside the cone (rho > radius).
  // Used in Class B: points outside the lower cone exceed the maximum lambda
  // for that z (would require zbar < -1), so they are not in the image.
  auto point_outside_cone = [this, &cone_rho_and_radius](const double z_one,
                                                   const double z_two) {
    double rho_tilde = 0.0;
    double circle_radius = 0.0;
    cone_rho_and_radius(z_one, z_two, rho_tilde, circle_radius);
    return rho_tilde > circle_radius and
           not equal_within_roundoff(
               rho_tilde, circle_radius,
               std::numeric_limits<double>::epsilon() * 100.0, radius_two_);
  };
  // Lower cone gate depends on class:
  //   Class A (z_minus_two < z_minus_one): cone descends; check z <=
  //   z_minus_one. Class B (z_minus_two > z_minus_one): cone ascends; check z
  //   in
  //     [z_minus_one, z_minus_two].
  const bool lower_cone_active =
      z_plane_minus_one_ != z_plane_minus_two_ and
      (z_plane_minus_two_ < z_plane_minus_one_
           ? target_coords[2] <= z_plane_minus_one_
           : (target_coords[2] >= z_plane_minus_one_ and
              target_coords[2] <= z_plane_minus_two_));
  // For Class A (z_minus_two < z_minus_one): reject if inside the lower cone
  // (point is below the lower boundary surface).
  // For Class B (z_minus_two > z_minus_one): reject if outside the lower cone
  // (point's rho exceeds the maximum achievable at that z, requiring lambda >
  // lambda_max which would give zbar < -1).
  const bool lower_cone_rejected =
      lower_cone_active and
      (z_plane_minus_two_ < z_plane_minus_one_
           ? point_inside_cone(z_plane_minus_one_, z_plane_minus_two_)
           : point_outside_cone(z_plane_minus_one_, z_plane_minus_two_));
  if ((target_coords[2] >= z_plane_plus_one_ and
       z_plane_plus_one_ != z_plane_plus_two_ and
       point_inside_cone(z_plane_plus_one_, z_plane_plus_two_)) or
      lower_cone_rejected) {
    return std::nullopt;
  }

  // To find lambda we will do numerical root finding.
  // function_to_zero is the function that is zero when lambda has the
  // correct value.
  const auto function_to_zero = [this, &target_coords](const double lambda) {
    const FunctionEvalTempVariables tmp(lambda, target_coords, center_one_,
                                        center_two_, radius_one_, radius_two_,
                                        z_plane_minus_one_, z_plane_minus_two_,
                                        z_plane_plus_one_, z_plane_plus_two_);
    return square(target_coords[0] - center_one_[0] -
                  lambda * (center_two_[0] - center_one_[0])) +
           square(target_coords[1] - center_one_[1] -
                  lambda * (center_two_[1] - center_one_[1])) -
           square((1.0 - lambda) * tmp.r_one_sin_theta_one +
                  lambda * tmp.r_two_sin_theta_two);
  };

  // Derivative with respect to lambda of function_to_zero.
  const auto deriv_function_to_zero = [this,
                                       &target_coords](const double lambda) {
    const FunctionEvalTempVariables tmp(lambda, target_coords, center_one_,
                                        center_two_, radius_one_, radius_two_,
                                        z_plane_minus_one_, z_plane_minus_two_,
                                        z_plane_plus_one_, z_plane_plus_two_);
    const double d_zbar_d_lambda_over_two =
        ((1.0 - tmp.one_plus_zbar_over_two) *
             (z_plane_minus_one_ - z_plane_minus_two_) -
         tmp.one_plus_zbar_over_two * (z_plane_plus_two_ - z_plane_plus_one_)) /
        ((1.0 - lambda) * (z_plane_plus_one_ - z_plane_minus_one_) +
         lambda * (z_plane_plus_two_ - z_plane_minus_two_));
    const double half_deriv_of_func_to_zero =
        (center_one_[0] - center_two_[0]) *
            (target_coords[0] - center_one_[0] -
             lambda * (center_two_[0] - center_one_[0])) +
        (center_one_[1] - center_two_[1]) *
            (target_coords[1] - center_one_[1] -
             lambda * (center_two_[1] - center_one_[1])) +
        -(tmp.r_one_sin_theta_one * (1.0 - lambda) +
          lambda * tmp.r_two_sin_theta_two) *
            (tmp.r_two_sin_theta_two - tmp.r_one_sin_theta_one -
             d_zbar_d_lambda_over_two *
                 (lambda * tmp.r_two_cos_theta_two *
                      (z_plane_plus_two_ - z_plane_minus_two_) /
                      tmp.r_two_sin_theta_two +
                  (1.0 - lambda) * tmp.r_one_cos_theta_one *
                      (z_plane_plus_one_ - z_plane_minus_one_) /
                      tmp.r_one_sin_theta_one));
    return 2.0 * half_deriv_of_func_to_zero;
  };

  // Find lambda via special-case checks or numerical root finding.
  double lambda = std::numeric_limits<double>::signaling_NaN();

  // Choose the minimum value of lambda.  Max value is always one.
  // These are not const because they can change with re-bracketing
  // below.
  // Class A (z_minus_two < z_minus_one): the minus-plane constraint gives a
  // lower bound on lambda (requiring zbar >= -1).
  // Class B (z_minus_two > z_minus_one): it gives an upper bound instead.
  // The formula (z_minus_one - z)/(z_minus_one - z_minus_two) evaluates to
  // the correct bound value in both cases; we just route it to lambda_min or
  // lambda_max depending on the class.
  const double minus_plane_bound =
      z_plane_minus_two_ == z_plane_minus_one_
          ? 0.0
          : (z_plane_minus_one_ - target_coords[2]) /
                (z_plane_minus_one_ - z_plane_minus_two_);
  double lambda_min = std::max(
      {0.0,
       z_plane_plus_two_ == z_plane_plus_one_
           ? 0.0
           : (target_coords[2] - z_plane_plus_one_) /
                 (z_plane_plus_two_ - z_plane_plus_one_),
       z_plane_minus_two_ < z_plane_minus_one_ ? minus_plane_bound : 0.0});
  double lambda_max = z_plane_minus_two_ > z_plane_minus_one_
                          ? std::min(1.0, minus_plane_bound)
                          : 1.0;

  // The root should always be bracketed.  However, if
  // lambda==lambda_min or lambda==lambda_max, the function may not be
  // exactly zero because of roundoff and then the root might not be
  // bracketed.
  double function_at_lambda_min = function_to_zero(lambda_min);
  double function_at_lambda_max = function_to_zero(lambda_max);

  if (lambda_max < std::numeric_limits<double>::epsilon() * 100.0) {
    // Class B only: lambda_max is within roundoff of zero, meaning the point
    // is at the inner sphere's lower boundary where lambda must be 0.
    lambda = 0.0;
  } else if (1.0 - lambda_min <
             std::numeric_limits<double>::epsilon() * 100.0) {
    // lambda_min is within roundoff of lambda_max, meaning that the
    // point must be within roundoff of a point on the outer sphere
    // where there is only one possible value of lambda.  Here we
    // treat this case as if lambda is exactly 1.0.
    lambda = 1.0;
  } else {
    // Here we do a root find.
    if (function_at_lambda_min * function_at_lambda_max > 0.0) {
      // Root not bracketed.
      // If the bracketing failure is due to roundoff, then
      // slightly adjust the bounds to increase the range. Otherwise, error.
      //
      // roundoff_ratio is the limiting ratio of the two function values
      // for which we should attempt to adjust the bounds.  It is ok for
      // roundoff_ratio to be much larger than actual roundoff, since it
      // doesn't hurt to expand the interval and try again when
      // otherwise we would just error.
      constexpr double roundoff_ratio = 1.e-3;
      if (abs(function_at_lambda_min) / abs(function_at_lambda_max) <
          roundoff_ratio) {
        // Slightly decrease lambda_min.  How far do we decrease it?
        // Figure that out by looking at the deriv of the function.
        const double deriv_function_at_lambda_min =
            deriv_function_to_zero(lambda_min);
        const double trial_lambda_min_increment =
            -2.0 * function_at_lambda_min / deriv_function_at_lambda_min;
        // new_lambda_min = lambda_min + trial_lambda_min_increment would be
        // a Newton-Raphson step except for the factor of 2 in
        // trial_lambda_min_increment. The factor of 2 is there to
        // over-compensate so that the new lambda_min brackets the
        // root.
        //
        // But sometimes the factor of 2 is not enough when
        // trial_lambda_min_increment is roundoff-small so that the
        // change in new_lambda_min is zero or only in the last one or
        // two bits.  If this is the case, then we set
        // lambda_min_increment to some small roundoff value with the
        // same sign as trial_lambda_min_increment.
        // Note that lambda is always between zero and one, so it is
        // ok to use an absolute epsilon.
        const double lambda_min_increment =
            abs(trial_lambda_min_increment) >
                    1000.0 * std::numeric_limits<double>::epsilon()
                ? trial_lambda_min_increment
                : std::copysign(1000.0 * std::numeric_limits<double>::epsilon(),
                                trial_lambda_min_increment);
        const double new_lambda_min = lambda_min + lambda_min_increment;
        const double function_at_new_lambda_min =
            function_to_zero(new_lambda_min);
        if (function_at_new_lambda_min * function_at_lambda_min > 0.0) {
          // Excluding code coverage, because it is not clear how to make
          // this error actually occur.
          // LCOV_EXCL_START
          ERROR(
              "Cannot find bracket after trying to adjust bracketing "
              "lambda_min due to roundoff : lambda_min="
              << lambda_min << " f(lambda_min)=" << function_at_lambda_min
              << " lambda_max=" << lambda_max << " f(lambda_max)="
              << function_at_lambda_max << " new_lambda_min=" << new_lambda_min
              << " f(new_lambda_min)=" << function_at_new_lambda_min
              << " df(new_lambda_min)=" << deriv_function_at_lambda_min
              << " new_lambda_min-lambda_min=" << new_lambda_min - lambda_min
              << "\n");
          // LCOV_EXCL_STOP
        }
        // Now the root is bracketed between lambda_min and new_lambda_min,
        // so replace lambda_max and lambda_min and then fall through to the
        // root finder.
        lambda_max = lambda_min;
        function_at_lambda_max = function_at_lambda_min;
        lambda_min = new_lambda_min;
        function_at_lambda_min = function_at_new_lambda_min;
      } else if (abs(function_at_lambda_max) / abs(function_at_lambda_min) <
                 roundoff_ratio) {
        // Slightly increase lambda_max.  How far do we increase it?
        // Figure that out by looking at the deriv of the function.
        const double deriv_function_at_lambda_max =
            deriv_function_to_zero(lambda_max);
        const double trial_lambda_max_increment =
            -2.0 * function_at_lambda_max / deriv_function_at_lambda_max;
        // new_lambda_max = lambda_max + trial_lambda_max_increment would be
        // a Newton-Raphson step except for the factor of 2 in
        // trial_lambda_max_increment. The factor of 2 is there to
        // over-compensate so that the new lambda_max brackets the
        // root.
        //
        // But sometimes the factor of 2 is not enough when
        // trial_lambda_max_increment is roundoff-small so that the
        // change in new_lambda_max is zero or only in the last one or
        // two bits.  If this is the case, then we set
        // lambda_max_increment to some small roundoff value with the
        // same sign as trial_lambda_max_increment.
        // Note that lambda is always between zero and one, so it is
        // ok to use an absolute epsilon.
        const double lambda_max_increment =
            abs(trial_lambda_max_increment) >
                    1000.0 * std::numeric_limits<double>::epsilon()
                ? trial_lambda_max_increment
                : std::copysign(1000.0 * std::numeric_limits<double>::epsilon(),
                                trial_lambda_max_increment);
        const double new_lambda_max = lambda_max + lambda_max_increment;
        const double function_at_new_lambda_max =
            function_to_zero(new_lambda_max);
        if (function_at_new_lambda_max * function_at_lambda_max > 0.0) {
          // Excluding code coverage, because it is not clear how to make
          // this error actually occur.
          // LCOV_EXCL_START
          ERROR(
              "Cannot find bracket after trying to adjust bracketing "
              "lambda_max due to roundoff : lambda_min="
              << lambda_min << " f(lambda_min)=" << function_at_lambda_min
              << " lambda_max=" << lambda_max << " f(lambda_max)="
              << function_at_lambda_max << " new_lambda_max=" << new_lambda_max
              << " f(new_lambda_max)=" << function_at_new_lambda_max
              << " df(new_lambda_max)=" << deriv_function_at_lambda_max
              << " new_lambda_max-lambda_max=" << new_lambda_max - lambda_min
              << "\n");
          // LCOV_EXCL_STOP
        }
        // Now the root is bracketed between lambda_max and new_lambda_max,
        // so replace lambda_max and lambda_min and then fall through to the
        // root finder.
        lambda_min = lambda_max;
        function_at_lambda_min = function_at_lambda_max;
        lambda_max = new_lambda_max;
        function_at_lambda_max = function_at_new_lambda_max;
      } else {
        // Excluding code coverage, because it is not clear how to make
        // this error actually occur.
        // LCOV_EXCL_START
        ERROR(
            "Root is not bracketed: "
            "lambda_min="
            << lambda_min << " f(lambda_min)=" << function_at_lambda_min
            << " lambda_max=" << lambda_max
            << " f(lambda_max)=" << function_at_lambda_max);
        // LCOV_EXCL_STOP
      }
    }
    // If we get here, root is bracketed. Use toms748, and use the
    // function evaluations that we have already done.

    // Lambda is between zero and 1, so the scale is unity, so therefore
    // abs and rel tolerance are equal.
    constexpr double abs_tol = 1.e-15;
    constexpr double rel_tol = 1.e-15;

    try {
      lambda =
          // NOLINTNEXTLINE(clang-analyzer-core)
          RootFinder::toms748(function_to_zero, lambda_min, lambda_max,
                              function_at_lambda_min, function_at_lambda_max,
                              abs_tol, rel_tol);
    } catch (std::exception&) { // LCOV_EXCL_LINE
      // This should never happen unless something is really wrong with
      // the root finder, so excluding this from code coverage.
      // LCOV_EXCL_START
      ERROR("Cannot find root after bracketing: lambda_min="
            << lambda_min << " f(lambda_min)=" << function_at_lambda_min
            << " lambda_max=" << lambda_max
            << " f(lambda_max)=" << function_at_lambda_max);
      // LCOV_EXCL_STOP
    }
  }

  // Now that we have lambda, construct inverse.
  const double one_plus_zbar_over_two =
      (target_coords[2] + lambda * (z_plane_minus_one_ - z_plane_minus_two_) -
       z_plane_minus_one_) /
      ((1.0 - lambda) * (z_plane_plus_one_ - z_plane_minus_one_) +
       lambda * (z_plane_plus_two_ - z_plane_minus_two_));
  const double rhobar = 1.0 + lambda;
  const double r_one_cos_theta_one =
      z_plane_minus_one_ - center_one_[2] +
      (z_plane_plus_one_ - z_plane_minus_one_) * one_plus_zbar_over_two;
  const double r_two_cos_theta_two =
      z_plane_minus_two_ - center_two_[2] +
      (z_plane_plus_two_ - z_plane_minus_two_) * one_plus_zbar_over_two;
  const double r_one_sin_theta_one =
      sqrt(square(radius_one_) - square(r_one_cos_theta_one));
  const double r_two_sin_theta_two =
      sqrt(square(radius_two_) - square(r_two_cos_theta_two));
  const double denom = 1.0 / ((1.0 - lambda) * r_one_sin_theta_one +
                              lambda * r_two_sin_theta_two);
  return {{rhobar *
               (target_coords[0] - center_one_[0] -
                lambda * (center_two_[0] - center_one_[0])) *
               denom,
           rhobar *
               (target_coords[1] - center_one_[1] -
                lambda * (center_two_[1] - center_one_[1])) *
               denom,
           2.0 * one_plus_zbar_over_two - 1.0}};
}

template <typename T>
tnsr::Ij<T, 3, Frame::NoFrame> UniformCylindricalSide::jacobian(
    const std::array<T, 3>& source_coords) const {
  const T& xbar = source_coords[0];
  const T& ybar = source_coords[1];
  const T& zbar = source_coords[2];

  auto jac =
      make_with_value<tnsr::Ij<T, 3, Frame::NoFrame>>(source_coords[0], 0.0);

  // Use jacobian components as temporary storage to avoid extra
  // memory allocations.
  // jac(2,2)=rhobar
  get<2, 2>(jac) = sqrt(square(xbar) + square(ybar));
  // jac(2,1)=R1 sin(theta1)
  get<2, 1>(jac) = sqrt(
      square(radius_one_) -
      square(z_plane_minus_one_ - center_one_[2] +
             0.5 * (zbar + 1.0) * (z_plane_plus_one_ - z_plane_minus_one_)));
  // jac(0,2)=cot theta1/rho
  get<0, 2>(jac) =
      (z_plane_minus_one_ - center_one_[2] +
       0.5 * (zbar + 1.0) * (z_plane_plus_one_ - z_plane_minus_one_)) /
      (get<2, 1>(jac) * get<2, 2>(jac));
  // jac(2,1)=R1 sin(theta1)/rho^3
  get<2, 1>(jac) /= cube(get<2, 2>(jac));
  // jac(2,0)=R2 sin(theta2)
  get<2, 0>(jac) = sqrt(
      square(radius_two_) -
      square(z_plane_minus_two_ - center_two_[2] +
             0.5 * (zbar + 1.0) * (z_plane_plus_two_ - z_plane_minus_two_)));
  // jac(1,2)=cot theta2/rho
  get<1, 2>(jac) =
      (z_plane_minus_two_ - center_two_[2] +
       0.5 * (zbar + 1.0) * (z_plane_plus_two_ - z_plane_minus_two_)) /
      (get<2, 0>(jac) * get<2, 2>(jac));
  // jac(2,0)=R2 sin(theta2)/rho^3
  get<2, 0>(jac) /= cube(get<2, 2>(jac));
  // jac(1,1)=lambda rho^2
  get<1, 1>(jac) = square(get<2, 2>(jac)) * (get<2, 2>(jac) - 1.0);

  // Now fill Jacobian values
  get<0, 0>(jac) =
      square(ybar) * get<2, 1>(jac) +
      (get<1, 1>(jac) + square(xbar)) * (get<2, 0>(jac) - get<2, 1>(jac)) +
      xbar * (center_two_[0] - center_one_[0]) / get<2, 2>(jac);
  get<1, 1>(jac) =
      square(xbar) * get<2, 1>(jac) +
      (get<1, 1>(jac) + square(ybar)) * (get<2, 0>(jac) - get<2, 1>(jac)) +
      ybar * (center_two_[1] - center_one_[1]) / get<2, 2>(jac);
  get<0, 1>(jac) = xbar * ybar * (get<2, 0>(jac) - 2.0 * get<2, 1>(jac)) +
                   ybar * (center_two_[0] - center_one_[0]) / get<2, 2>(jac);
  get<1, 0>(jac) = xbar * ybar * (get<2, 0>(jac) - 2.0 * get<2, 1>(jac)) +
                   xbar * (center_two_[1] - center_one_[1]) / get<2, 2>(jac);
  get<1, 2>(jac) =
      0.5 * (get<0, 2>(jac) * (z_plane_plus_one_ - z_plane_minus_one_) *
                 (get<2, 2>(jac) - 2.0) +
             get<1, 2>(jac) * (z_plane_plus_two_ - z_plane_minus_two_) *
                 (1.0 - get<2, 2>(jac)));
  get<0, 2>(jac) = get<1, 2>(jac) * xbar;
  get<1, 2>(jac) *= ybar;
  get<2, 1>(jac) = (z_plane_minus_two_ - z_plane_minus_one_ +
                    0.5 * (zbar + 1) *
                        (z_plane_plus_two_ - z_plane_minus_two_ -
                         z_plane_plus_one_ + z_plane_minus_one_)) /
                   get<2, 2>(jac);
  get<2, 0>(jac) = xbar * get<2, 1>(jac);
  get<2, 1>(jac) *= ybar;
  get<2, 2>(jac) =
      0.5 * ((2.0 - get<2, 2>(jac)) * (z_plane_plus_one_ - z_plane_minus_one_) +
             (get<2, 2>(jac) - 1.0) * (z_plane_plus_two_ - z_plane_minus_two_));
  return jac;
}

template <typename T>
tnsr::Ij<T, 3, Frame::NoFrame> UniformCylindricalSide::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  return determinant_and_inverse(jacobian(source_coords)).second;
}

void UniformCylindricalSide::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | center_one_;
    p | center_two_;
    p | radius_one_;
    p | radius_two_;
    p | z_plane_plus_one_;
    p | z_plane_minus_one_;
    p | z_plane_plus_two_;
    p | z_plane_minus_two_;
  }
}

bool operator==(const UniformCylindricalSide& lhs,
                const UniformCylindricalSide& rhs) {
  // don't need to compare theta_max_one_ or theta_max_two_
  // because they are uniquely determined from the other variables.
  return lhs.center_one_ == rhs.center_one_ and
         lhs.radius_one_ == rhs.radius_one_ and
         lhs.z_plane_plus_one_ == rhs.z_plane_plus_one_ and
         lhs.z_plane_minus_one_ == rhs.z_plane_minus_one_ and
         lhs.center_two_ == rhs.center_two_ and
         lhs.radius_two_ == rhs.radius_two_ and
         lhs.z_plane_plus_two_ == rhs.z_plane_plus_two_ and
         lhs.z_plane_minus_two_ == rhs.z_plane_minus_two_;
}

bool operator!=(const UniformCylindricalSide& lhs,
                const UniformCylindricalSide& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template std::array<DTYPE(data), 3> UniformCylindricalSide::operator()( \
      const std::array<DTYPE(data), 3>& source_coords) const;             \
  template tnsr::Ij<DTYPE(data), 3, Frame::NoFrame>                       \
  UniformCylindricalSide::jacobian(                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;             \
  template tnsr::Ij<DTYPE(data), 3, Frame::NoFrame>                       \
  UniformCylindricalSide::inv_jacobian(                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
