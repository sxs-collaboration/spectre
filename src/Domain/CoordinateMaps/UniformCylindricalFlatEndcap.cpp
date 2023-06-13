// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"

#include <boost/math/special_functions/sign.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <sstream>
#include <tuple>
#include <utility>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CylindricalEndcapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace domain::CoordinateMaps {

namespace {

// The function that the inverse map needs to root-find to find rhobar.
double this_function_is_zero_for_correct_rhobar(
    const double rhobar, const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, double radius_one,
    const double radius_two, const double theta_max_one,
    const std::array<double, 3>& target_coords) {
  const double r_one_cos_theta_one = radius_one * cos(rhobar * theta_max_one);
  const double lambda =
      (target_coords[2] - center_one[2] - r_one_cos_theta_one) /
      (center_two[2] - center_one[2] - r_one_cos_theta_one);
  return square(target_coords[0] - center_one[0] -
                lambda * (center_two[0] - center_one[0])) +
         square(target_coords[1] - center_one[1] -
                lambda * (center_two[1] - center_one[1])) -
         square((1.0 - lambda) * radius_one * sin(rhobar * theta_max_one) +
                lambda * radius_two * rhobar);
}

// min and max values of rhobar in the inverse function.
std::tuple<double, double> rhobar_min_max(
    const std::array<double, 3>& center_one, const double radius_one,
    const double theta_max_one, const std::array<double, 3>& target_coords) {
  // Choose the minimum value of rhobar so that lambda >=0, where
  // lambda is the quantity inside function_to_zero.  Note that the
  // denominator of lambda inside that function is always positive.
  const double rhobar_min =
      target_coords[2] - center_one[2] >= radius_one
          ? 0.0
          : acos((target_coords[2] - center_one[2]) / radius_one) /
                theta_max_one;
  return {rhobar_min, 1.0};
}

// Returns whether the map is invertible if the target is on sphere_one.
bool is_invertible_for_target_on_sphere_one(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, const double radius_one,
    const double radius_two, const double theta_max_one,
    const std::array<double, 3>& target_coords) {
  const auto [rhobar_min, rhobar_max] =
      rhobar_min_max(center_one, radius_one, theta_max_one, target_coords);
  if (equal_within_roundoff(rhobar_min, rhobar_max)) {
    // Special case where rhobar_min == rhobar_max to roundoff.  This
    // case occurs when the target point (assumed to be on sphere_one)
    // is at rhobar=1.  In this case there is nothing to test because
    // there is only one possible value of rhobar, so there cannot be
    // two roots for two different values of rhobar.
    return true;
  }
  const size_t num_pts = 1000;
  const double drhobar = (rhobar_max - rhobar_min) / (num_pts - 1.0);
  // Because the target point is on sphere_one, the only root should
  // be at rhobar = rhobar_min, where lambda=0.  The actual value of
  // the function at rhobar_min may be either sign because of
  // roundoff.  So ignore that point, and start at the next point.
  // Keep track of the sign, and look for sign changes as we progress
  // from point to point.
  std::optional<double> sign;
  for (size_t i = 1; i < num_pts; ++i) {
    const double rhobar = rhobar_min + i * drhobar;
    const double func = this_function_is_zero_for_correct_rhobar(
        rhobar, center_one, center_two, radius_one, radius_two, theta_max_one,
        target_coords);
    if (not sign.has_value()) {
      sign = func > 0.0 ? 1.0 : -1.0;
    } else if (func * sign.value() <= 0.0) {
      // There is another root, so function is not invertible.
      //
      // Note that the code should rarely get here if the map
      // parameters are in range, so it is difficult to add an error
      // test because I don't know how to make it get here on purpose;
      // therefore we turn off codecov.
      return false; // LCOV_EXCL_LINE
    }
  }
  return true;  // No roots found.
}
}  // namespace

bool is_uniform_cylindrical_flat_endcap_invertible_on_sphere_one(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, double radius_one,
    double radius_two, double theta_max_one) {
  if (center_one[1] == center_two[1] and center_one[0] == center_two[0]) {
    return true;
  }
  // Choose the worst direction, which is the direction of x-y-plane
  // projection of the difference between center_one and center_two.
  const double phi =
      atan2(center_one[1] - center_two[1], center_one[0] - center_two[0]);
  const size_t num_pts = 100;
  for (size_t i = 0; i < num_pts; ++i) {
    const double theta_one = theta_max_one * (i / (num_pts - 1.0));
    if (not is_invertible_for_target_on_sphere_one(
            center_one, center_two, radius_one, radius_two, theta_max_one,
            {{center_one[0] + radius_one * sin(theta_one) * cos(phi),
              center_one[1] + radius_one * sin(theta_one) * sin(phi),
              center_one[2] + radius_one * cos(theta_one)}})) {
      // Note that the code should rarely get here if the map
      // parameters are in range, so it is difficult to add an error
      // test because I don't know how to make it get here on purpose;
      // therefore we turn off codecov.
      return false; // LCOV_EXCL_LINE
    }
  }
  return true;
}

UniformCylindricalFlatEndcap::UniformCylindricalFlatEndcap(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, double radius_one,
    double radius_two, double z_plane_one)
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
        // codecov doesn't understand lambdas in constructors for some
        // reason, and says the next line is not covered even though it is.
      }()), // LCOV_EXCL_LINE
      radius_two_([&radius_two]() {
        // The equal_within_roundoff below has an implicit scale of 1,
        // so the ASSERT may trigger in the case where we really
        // want an entire domain that is very small.
        ASSERT(not equal_within_roundoff(radius_two, 0.0),
               "Cannot have zero radius_two");
        ASSERT(radius_two > 0.0, "Cannot have negative radius_two");
        return radius_two;
        // codecov doesn't understand lambdas in constructors for some
        // reason, and says the next line is not covered even though it is.
      }()), // LCOV_EXCL_LINE
      z_plane_one_(z_plane_one),
      theta_max_one_([&center_one, &radius_one, &z_plane_one]() {
        const double cos_theta_max = (z_plane_one - center_one[2]) / radius_one;
        ASSERT(abs(cos_theta_max) < 1.0,
               "Plane one must intersect sphere_one, and at more than one "
               "point. You probably specified a bad value of z_plane_one, "
               "radius_one, or the z component of center_one. "
               "Here z_plane_one="
                   << z_plane_one << ", radius_one = " << radius_one
                   << ", center_one=" << center_one
                   << ", and cos_theta_max = " << cos_theta_max);
        return acos(cos_theta_max);
      }()) {
  // The code below defines several variables that are used only in ASSERTs.
  // We put that code in a #ifdef SPECTRE_DEBUG to avoid clang-tidy complaining
  // about unused variables in release mode.
#ifdef SPECTRE_DEBUG

  // For some reason, codecov thinks that the following lambda never
  // gets called, even though it is in all of the ASSERT messages below.
  // LCOV_EXCL_START
  const auto param_string = [this]() -> std::string {
    std::ostringstream buffer;
    buffer << "\nParameters to UniformCylindricalFlatEndcap:\nradius_one="
           << radius_one_ << "\nradius_two=" << radius_two_
           << "\ncenter_one=" << center_one_ << "\ncenter_two=" << center_two_
           << "\nz_plane_one=" << z_plane_one_;
    return buffer.str();
  };
  // LCOV_EXCL_STOP

  // Assumptions made in the map.  The exact numbers for all of these
  // can be changed, as long as the unit test is changed to test them.
  ASSERT(center_two[2] >= center_one[2] + 1.05 * radius_one,
         "center_two[2] must be >= center_one[2] + 1.05 * radius_one, not "
             << center_two[2] << " " << center_one[2] + 1.05 * radius_one
             << param_string());
  ASSERT(center_two[2] <= center_one[2] + 5.0 * radius_one,
         "center_two[2] must be <= center_one[2] + 5.0 * radius_one, not "
             << center_two[2] << " " << center_one[2] + 5.0 * radius_one
             << param_string());
  ASSERT(radius_two >= 0.1 * radius_one * sin(theta_max_one_),
         "radius_two is too small: " << radius_two << " "
                                     << 0.1 * radius_one * sin(theta_max_one_)
                                     << param_string());
  ASSERT(theta_max_one_ < M_PI * 0.35,
         "z_plane_one is too close to the center of sphere_one: theta/pi = "
             << theta_max_one_ / M_PI << param_string());
  ASSERT(theta_max_one_ > M_PI * 0.075,
         "z_plane_one is too far from the center of sphere_one: theta/pi = "
             << theta_max_one_ / M_PI << param_string());
  ASSERT(
      is_uniform_cylindrical_flat_endcap_invertible_on_sphere_one(
          center_one_, center_two_, radius_one_, radius_two_, theta_max_one_),
      "The map is not invertible at at least one point on sphere_one."
      " center_one = "
          << center_one_ << " center_two = " << center_two_
          << " radius_one = " << radius_one_ << " radius_two = " << radius_two_
          << " theta_max_one = " << theta_max_one_ << param_string());

  const double horizontal_dist_spheres =
      sqrt(square(center_one[0] - center_two[0]) +
           square(center_one[1] - center_two[1]));

  ASSERT(horizontal_dist_spheres <= radius_one_ * sin(theta_max_one_),
         "Horizontal distance between centers is too large: "
             << horizontal_dist_spheres << " "
             << radius_one_ * sin(theta_max_one_) << param_string());

  // max_horizontal_dist_between_circles can be either sign;
  // therefore alpha can be in either the first or second quadrant.
  const double max_horizontal_dist_between_circles =
      horizontal_dist_spheres + radius_one_ * sin(theta_max_one_) - radius_two_;

  const double alpha =
      atan2(center_two[2] - z_plane_one, max_horizontal_dist_between_circles);
  ASSERT(alpha > 1.1 * theta_max_one_,
         "Angle alpha is too small: alpha = "
             << alpha << ", theta_max_one = " << theta_max_one_
             << ", max_horizontal_dist_between_circles = "
             << max_horizontal_dist_between_circles
             << ", horizontal_dist_spheres = " << horizontal_dist_spheres
             << param_string());
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
UniformCylindricalFlatEndcap::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];
  std::array<ReturnType, 3> target_coords{};
  ReturnType& x = target_coords[0];
  ReturnType& y = target_coords[1];
  ReturnType& z = target_coords[2];

  // Use y and z as temporary storage to avoid allocations,
  // before setting them to their actual values.
  z = sqrt(square(xbar) + square(ybar));  // rhobar in the dox
  y = radius_one_ *
          cylindrical_endcap_helpers::sin_ax_over_x(z, theta_max_one_) *
          (1.0 - zbar) +
      radius_two_ * (1.0 + zbar);

  x = 0.5 * (y * xbar + center_one_[0] * (1.0 - zbar) +
             center_two_[0] * (1.0 + zbar));
  y = 0.5 * (y * ybar + center_one_[1] * (1.0 - zbar) +
             center_two_[1] * (1.0 + zbar));
  z = 0.5 *
      ((radius_one_ * cos(theta_max_one_ * z) + center_one_[2]) * (1.0 - zbar) +
       center_two_[2] * (1.0 + zbar));
  return target_coords;
}

std::optional<std::array<double, 3>> UniformCylindricalFlatEndcap::inverse(
    const std::array<double, 3>& target_coords) const {
  // First do some easy checks that target_coords is in the range
  // of the map.  Need to accept points that are out of range by roundoff.

  // check z >= z_plane_one_
  if (target_coords[2] < z_plane_one_ and
      not equal_within_roundoff(target_coords[2], z_plane_one_)) {
    return std::nullopt;
  }

  // check that target point is at smaller z than circle_two
  if (target_coords[2] > center_two_[2] and
      not equal_within_roundoff(target_coords[2], center_two_[2])) {
    return std::nullopt;
  }

  // check that point is outside or on sphere_one_
  const double r_minus_center_one_squared =
      square(target_coords[0] - center_one_[0]) +
      square(target_coords[1] - center_one_[1]) +
      square(target_coords[2] - center_one_[2]);
  if (r_minus_center_one_squared < square(radius_one_) and
      not equal_within_roundoff(sqrt(r_minus_center_one_squared),
                                radius_one_)) {
    // sqrt above because we don't want to change the scale of
    // equal_within_roundoff too much.
    return std::nullopt;
  }

  // Check if the point is inside the cone
  const double lambda_tilde =
      (target_coords[2] - z_plane_one_) / (center_two_[2] - z_plane_one_);
  const double rho_tilde =
      sqrt(square(target_coords[0] - center_one_[0] -
                  lambda_tilde * (center_two_[0] - center_one_[0])) +
           square(target_coords[1] - center_one_[1] -
                  lambda_tilde * (center_two_[1] - center_one_[1])));
  const double circle_radius =
      radius_one_ * sin(theta_max_one_) * (1.0 - lambda_tilde) +
      radius_two_ * lambda_tilde;
  if (rho_tilde > circle_radius and
      not equal_within_roundoff(rho_tilde, circle_radius,
                                std::numeric_limits<double>::epsilon() * 100.0,
                                radius_two_)) {
    return std::nullopt;
  }

  // To find rhobar we will do numerical root finding.
  // function_to_zero is the function that is zero when rhobar has the
  // correct value.
  const auto function_to_zero = [this, &target_coords](const double rhobar) {
    return this_function_is_zero_for_correct_rhobar(
        rhobar, center_one_, center_two_, radius_one_, radius_two_,
        theta_max_one_, target_coords);
  };

  // Derivative of function_to_zero with respect to rhobar.
  const auto deriv_function_to_zero = [this,
                                       &target_coords](const double rhobar) {
    const double r_one_cos_theta_one =
        radius_one_ * cos(rhobar * theta_max_one_);
    const double lambda =
        (target_coords[2] - center_one_[2] - r_one_cos_theta_one) /
        (center_two_[2] - center_one_[2] - r_one_cos_theta_one);
    // deriv of r_one_cos_theta_one with respect to rho.
    const double d_r_one_cos_theta_one =
        radius_one_ * sin(rhobar * theta_max_one_) * theta_max_one_;
    const double dlambda_drho =
        d_r_one_cos_theta_one * (1.0 - lambda) /
        (center_two_[2] - center_one_[2] - r_one_cos_theta_one);
    return -2.0 * dlambda_drho *
               ((center_two_[0] - center_one_[0]) *
                    (target_coords[0] - center_one_[0] -
                     lambda * (center_two_[0] - center_one_[0])) +
                (center_two_[1] - center_one_[1]) *
                    (target_coords[1] - center_one_[1] -
                     lambda * (center_two_[1] - center_one_[1]))) -
           2.0 *
               ((1.0 - lambda) * radius_one_ * sin(rhobar * theta_max_one_) +
                lambda * radius_two_ * rhobar) *
               ((radius_two_ * rhobar -
                 radius_one_ * sin(rhobar * theta_max_one_)) *
                    dlambda_drho +
                (1.0 - lambda) * r_one_cos_theta_one * theta_max_one_ +
                lambda * radius_two_);
  };

  double rhobar_min{};
  double rhobar_max{};
  std::tie(rhobar_min, rhobar_max) =
      rhobar_min_max(center_one_, radius_one_, theta_max_one_, target_coords);

  // If rhobar is zero, then the root finding doesn't converge
  // well. This is because the function behaves like rhobar^2 for
  // small values of rhobar, so both the function and its derivative
  // are zero at rhobar==0.
  //
  // Note that rhobar==0 is not that uncommon, and will occur if there
  // are grid points on the symmetry axis.
  //
  // Similarly, rhobar==1 occurs if there are grid points on the block
  // boundary.
  //
  // So check for the special cases of rhobar==0 and rhobar==1
  // before doing any root finding.
  const auto this_is_small_if_rhobar_is_near_unity = [this, &target_coords]() {
    const double denom = 1.0 / (center_two_[2] - center_one_[2] -
                                radius_one_ * cos(theta_max_one_));
    const double lambda_if_rhobar_is_unity =
        (target_coords[2] - center_one_[2] -
         radius_one_ * cos(theta_max_one_)) *
        denom;
    // The function function_to_zero goes like
    // first_term + second_term (1-rhobar) when 1-rhobar is small,
    // where first_term goes to zero (faster than (1-rhobar)) when
    // 1-rhobar is small.
    // We don't have a good sense of scale for first_term, so we
    // check the size of first_term/|second_term|.
    const double temp = radius_one_ * sin(theta_max_one_) +
                        lambda_if_rhobar_is_unity *
                            (radius_two_ - radius_one_ * sin(theta_max_one_));
    const double first_term =
        square(target_coords[0] - center_one_[0] -
               lambda_if_rhobar_is_unity * (center_two_[0] - center_one_[0])) +
        square(target_coords[1] - center_one_[1] -
               lambda_if_rhobar_is_unity * (center_two_[1] - center_one_[1])) -
        square(temp);
    // delta_lambda is (lambda-lambda_if_rhobar_is_unity)/(1-rhobar)
    // to first order in 1-rhobar.
    const double delta_lambda =
        -radius_one_ * theta_max_one_ * sin(theta_max_one_) * square(denom) *
        (center_two_[2] - target_coords[2]);
    const double second_term =
        2.0 * delta_lambda *
            (lambda_if_rhobar_is_unity *
                 (square(center_two_[0] - center_one_[0]) +
                  square(center_two_[1] - center_one_[1])) -
             (target_coords[0] - center_one_[0]) *
                 (center_two_[1] - center_one_[1]) -
             (target_coords[1] - center_one_[1]) *
                 (center_two_[0] - center_one_[0])) -
        2.0 * temp *
            (-(1.0 - lambda_if_rhobar_is_unity) * radius_one_ * theta_max_one_ *
                 cos(theta_max_one_) -
             lambda_if_rhobar_is_unity * radius_two_ +
             delta_lambda * (radius_two_ - radius_one_ * sin(theta_max_one_)));
    return std::abs(first_term) / std::abs(second_term);
  };

  const auto this_is_small_if_rhobar_is_near_zero = [this, &target_coords]() {
    const double denom = 1.0 / (center_two_[2] - center_one_[2] - radius_one_);
    const double lambda_if_rhobar_is_zero =
        (target_coords[2] - center_one_[2] - radius_one_) * denom;
    // The function we are trying to make zero goes like
    // first_term + second_term rhobar^2, and first_term goes to zero
    // (faster than rhobar^2) when rhobar is small.  We don't have a good
    // sense of scale for first_term, so we check the size of
    // first_term/second_term.
    const double first_term =
        square(target_coords[0] - center_one_[0] -
               lambda_if_rhobar_is_zero * (center_two_[0] - center_one_[0])) +
        square(target_coords[1] - center_one_[1] -
               lambda_if_rhobar_is_zero * (center_two_[1] - center_one_[1]));
    // delta_lambda is (lambda-lambda_if_rhobar_is_zero)/rhobar^2
    // to first order in rhobar^2.
    const double delta_lambda = radius_one_ * square(theta_max_one_) *
                                square(denom) *
                                (center_two_[2] - target_coords[2]);
    const double second_term =
        2.0 * delta_lambda *
            (lambda_if_rhobar_is_zero *
                 (square(center_two_[0] - center_one_[0]) +
                  square(center_two_[1] - center_one_[1])) -
             (target_coords[0] - center_one_[0]) *
                 (center_two_[1] - center_one_[1]) -
             (target_coords[1] - center_one_[1]) *
                 (center_two_[0] - center_one_[0])) -
        square((1.0 - lambda_if_rhobar_is_zero) * radius_one_ * theta_max_one_ +
               lambda_if_rhobar_is_zero * radius_two_);
    return std::abs(first_term) / std::abs(second_term);
  };

  if (rhobar_min < 1.e-6 and this_is_small_if_rhobar_is_near_zero() < 1.e-14) {
    // Treat rhobar as zero, if it is zero to roundoff.
    // We empirically choose roundoff here to be 1.e-14.  We don't
    // really need the rhobar_min < 1.e-6 in the 'if' above, but we
    // add it so that we can save the computational expense of calling
    // this_is_small_if_rhobar_is_near_zero() when we already know
    // that rhobar isn't close to zero. We choose the relatively large
    // value of 1.e-6 to err on the side of caution.
    const double lambda =
        (target_coords[2] - center_one_[2] - radius_one_) /
        (center_two_[2] - center_one_[2] - radius_one_);
    return {{0.0, 0.0, 2.0 * lambda - 1.0}};
  } else if (1.0 - rhobar_max < 1.e-6 and
             this_is_small_if_rhobar_is_near_unity() < 1.e-14) {
    // Treat rhobar as unity, if it is unity to roundoff.
    // We empirically choose roundoff here to be 1.e-14.  We don't
    // really need the 1-rhobar_max < 1.e-6 in the 'if' above, but we
    // add it so that we can save the computational expense of calling
    // this_is_small_if_rhobar_is_near_unity() when we already know
    // that rhobar isn't close to unity. We choose the relatively
    // large value of 1.e-6 to err on the side of caution.
    const double lambda =
        (target_coords[2] - center_one_[2] -
         radius_one_ * cos(theta_max_one_)) /
        (center_two_[2] - center_one_[2] - radius_one_ * cos(theta_max_one_));
    const double denom =
        1.0 / ((1.0 - lambda) * radius_one_ * sin(theta_max_one_) +
               lambda * radius_two_);
    return {{(target_coords[0] - center_one_[0] -
              lambda * (center_two_[0] - center_one_[0])) *
                 denom,
             (target_coords[1] - center_one_[1] -
              lambda * (center_two_[1] - center_one_[1])) *
                 denom,
             2.0 * lambda - 1.0}};
  }

  // If we get here, then rhobar is not near unity or near zero.
  // So let's find rhobar.
  double rhobar = std::numeric_limits<double>::signaling_NaN();

  // The root should always be bracketed.  However, if
  // rhobar==rhobar_min or rhobar==rhobar_max, the function may not be
  // exactly zero because of roundoff and then the root might not be
  // bracketed.
  double function_at_rhobar_min = function_to_zero(rhobar_min);
  double function_at_rhobar_max = function_to_zero(rhobar_max);
  if (function_at_rhobar_min * function_at_rhobar_max > 0.0) {
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
    if (abs(function_at_rhobar_min) / abs(function_at_rhobar_max) <
        roundoff_ratio) {
      // Slightly decrease rhobar_min.  How far do we decrease it?
      // Figure that out by looking at the deriv of the function.
      const double deriv_function_at_rhobar_min =
          deriv_function_to_zero(rhobar_min);
      const double trial_rhobar_min_increment =
          -2.0 * function_at_rhobar_min / deriv_function_at_rhobar_min;
      // new_rhobar_min = rhobar_min + trial_rhobar_min_increment would be
      // a Newton-Raphson step except for the factor of 2 in
      // trial_rhobar_min_increment. The factor of 2 is there to
      // over-compensate so that the new rhobar_min brackets the
      // root.
      //
      // But sometimes the factor of 2 is not enough when
      // trial_rhobar_min_increment is roundoff-small so that the
      // change in new_rhobar_min is zero or only in the last one or
      // two bits.  If this is the case, then we set
      // rhobar_min_increment to some small roundoff value with the
      // same sign as trial_rhobar_min_increment.
      // Note that rhobar is always between zero and one, so it is
      // ok to use an absolute epsilon.
      const double rhobar_min_increment =
          abs(trial_rhobar_min_increment) >
                  100.0 * std::numeric_limits<double>::epsilon()
              ? trial_rhobar_min_increment
              : boost::math::copysign(
                    100.0 * std::numeric_limits<double>::epsilon(),
                    trial_rhobar_min_increment);
      const double new_rhobar_min = rhobar_min + rhobar_min_increment;
      const double function_at_new_rhobar_min =
          function_to_zero(new_rhobar_min);
      if (function_at_new_rhobar_min * function_at_rhobar_min > 0.0) {
        // This error should never happen except for a bug, so it
        // is hard to make an error test for it.
        // LCOV_EXCL_START
        ERROR(
            "Cannot find bracket after trying to adjust bracketing "
            "rhobar_min due "
            "to roundoff : rhobar_min="
            << rhobar_min << " f(rhobar_min)=" << function_at_rhobar_min
            << " rhobar_max=" << rhobar_max << " f(rhobar_max)="
            << function_at_rhobar_max << " new_rhobar_min=" << new_rhobar_min
            << " f(new_rhobar_min)=" << function_at_new_rhobar_min
            << " df(new_rhobar_min)=" << deriv_function_at_rhobar_min
            << " new_rhobar_min-rhobar_min=" << new_rhobar_min - rhobar_min
            << "\n");
        // LCOV_EXCL_STOP
      }
      // Now the root is bracketed between rhobar_min and new_rhobar_min,
      // so replace rhobar_max and rhobar_min and then fall through to the
      // root finder.
      rhobar_max = rhobar_min;
      function_at_rhobar_max = function_at_rhobar_min;
      rhobar_min = new_rhobar_min;
      function_at_rhobar_min = function_at_new_rhobar_min;
    } else {
      // This error should never happen except for a bug, so it
      // is hard to make an error test for it.
      // LCOV_EXCL_START
      ERROR("Root is not bracketed: rhobar_min="
            << rhobar_min << " f(rhobar_min)=" << function_at_rhobar_min
            << " rhobar_max=" << rhobar_max
            << " f(rhobar_max)=" << function_at_rhobar_max);
      // LCOV_EXCL_STOP
    }
  }
  // If we get here, root is bracketed. Use toms748, and use the
  // function evaluations that we have already done.

  // Rhobar is between zero and 1, so the scale is unity, so therefore
  // abs and rel tolerance are equal.
  constexpr double abs_tol = 1.e-15;
  constexpr double rel_tol = 1.e-15;

  try {
    rhobar =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(function_to_zero, rhobar_min, rhobar_max,
                            function_at_rhobar_min, function_at_rhobar_max,
                            abs_tol, rel_tol);
    // This error should never happen except for a bug, so it
    // is hard to make an error test for it.
    // LCOV_EXCL_START
  } catch (std::exception&) {
    ERROR("Cannot find root after bracketing: rhobar_min="
          << rhobar_min << " f(rhobar_min)=" << function_at_rhobar_min
          << " rhobar_max=" << rhobar_max
          << " f(rhobar_max)=" << function_at_rhobar_max);
    // LCOV_EXCL_STOP
  }

  // Now that we have rhobar, construct inverse.
  const double r_one_cos_theta_one = radius_one_ * cos(rhobar * theta_max_one_);
  const double lambda =
      (target_coords[2] - center_one_[2] - r_one_cos_theta_one) /
      (center_two_[2] - center_one_[2] - r_one_cos_theta_one);
  const double denom =
      1.0 / ((1.0 - lambda) * radius_one_ * sin(rhobar * theta_max_one_) +
             lambda * radius_two_ * rhobar);
  return {{rhobar *
               (target_coords[0] - center_one_[0] -
                lambda * (center_two_[0] - center_one_[0])) *
               denom,
           rhobar *
               (target_coords[1] - center_one_[1] -
                lambda * (center_two_[1] - center_one_[1])) *
               denom,
           2.0 * lambda - 1.0}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
UniformCylindricalFlatEndcap::jacobian(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];

  auto jac =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // Use jacobian components as temporary storage to avoid extra
  // memory allocations.
  get<2, 2>(jac) = sqrt(square(xbar) + square(ybar));
  get<2, 0>(jac) = 0.5 * radius_one_ *
                   cylindrical_endcap_helpers::sin_ax_over_x(get<2, 2>(jac),
                                                             theta_max_one_) *
                   (1.0 - zbar);
  get<1, 1>(jac) = get<2, 0>(jac) + 0.5 * radius_two_ * (1.0 + zbar);
  get<2, 1>(jac) = theta_max_one_ * get<2, 0>(jac);
  get<1, 2>(jac) =
      0.5 *
      (radius_two_ - radius_one_ * cylindrical_endcap_helpers::sin_ax_over_x(
                                       get<2, 2>(jac), theta_max_one_));
  get<1, 0>(jac) = 0.5 * radius_one_ *
                   cylindrical_endcap_helpers::one_over_x_d_sin_ax_over_x(
                       get<2, 2>(jac), theta_max_one_) *
                   (1.0 - zbar);

  // Now fill Jacobian values
  get<0, 0>(jac) = square(xbar) * get<1, 0>(jac) + get<1, 1>(jac);
  get<1, 1>(jac) = square(ybar) * get<1, 0>(jac) + get<1, 1>(jac);
  get<0, 1>(jac) = xbar * ybar * get<1, 0>(jac);
  get<1, 0>(jac) = get<0, 1>(jac);
  get<2, 0>(jac) = -xbar * get<2, 1>(jac);
  get<2, 1>(jac) = -ybar * get<2, 1>(jac);
  get<0, 2>(jac) =
      xbar * get<1, 2>(jac) + 0.5 * (center_two_[0] - center_one_[0]);
  get<1, 2>(jac) =
      ybar * get<1, 2>(jac) + 0.5 * (center_two_[1] - center_one_[1]);
  get<2, 2>(jac) = 0.5 * (center_two_[2] - center_one_[2] -
                          radius_one_ * cos(theta_max_one_ * get<2, 2>(jac)));

  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
UniformCylindricalFlatEndcap::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  return determinant_and_inverse(jacobian(source_coords)).second;
}

void UniformCylindricalFlatEndcap::pup(PUP::er& p) {
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
    p | z_plane_one_;
    p | theta_max_one_;
  }
}

bool operator==(const UniformCylindricalFlatEndcap& lhs,
                const UniformCylindricalFlatEndcap& rhs) {
  // don't need to compare theta_max_one_
  // because it is uniquely determined from the other variables.
  return lhs.center_one_ == rhs.center_one_ and
         lhs.radius_one_ == rhs.radius_one_ and
         lhs.z_plane_one_ == rhs.z_plane_one_ and
         lhs.center_two_ == rhs.center_two_ and
         lhs.radius_two_ == rhs.radius_two_;
}

bool operator!=(const UniformCylindricalFlatEndcap& lhs,
                const UniformCylindricalFlatEndcap& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  UniformCylindricalFlatEndcap::operator()(                                  \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  UniformCylindricalFlatEndcap::jacobian(                                    \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  UniformCylindricalFlatEndcap::inv_jacobian(                                \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
