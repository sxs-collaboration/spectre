// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalEndcap.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

namespace CylindricalEndcap_detail {

// Maps a unit disk to the z>z_plane portion of a sphere
// with a given center and radius.
// The source coordinates are Cartesian xbar,ybar,zbar.
// This is a 2D map, but has the interface of a 3D map.
// The map is independent of zbar=source_coords[2], and
// the 3 coordinates returned by operator() obey a constraint
// (namely that they lie on a 2-sphere).
class InnerSphereMap {
 public:
  InnerSphereMap(const std::array<double, 3>& center, double radius,
                 double z_plane) noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;
  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;
  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;
  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

 private:
  // Compute sin(ax)/x.
  // For small x, use sin(ax)/x = a(1 - a^2 x^2 / 6 + ...) to evaluate.
  // x is considered small if the first-ignored term in the series is roundoff.
  static double sin_ax_over_x(double x, double ax, double a) noexcept;
  static double sin_ax_over_x(double x, double a) noexcept {
    return sin_ax_over_x(x, a * x, a);
  }
  static DataVector sin_ax_over_x(const DataVector& x, double a) noexcept;

  // Compute 1/x d/dx [sin(ax)/x].
  // For small x, use sin(ax)/x = a(1 - a^2 x^2 / 6 + a^4 x^4 / 5! ...).
  // x is considered small if the first-ignored term in the series is roundoff.
  static double dlogx_sin_ax_over_x(double x, double ax, double a) noexcept;
  static double dlogx_sin_ax_over_x(double x, double a) noexcept {
    return dlogx_sin_ax_over_x(x, a * x, a);
  }
  static DataVector dlogx_sin_ax_over_x(const DataVector& x, double a) noexcept;

  const std::array<double, 3> center_;
  const double radius_;
  const double theta_;
};

InnerSphereMap::InnerSphereMap(const std::array<double, 3>& center,
                               double radius, double z_plane) noexcept
    : center_(center),
      radius_([&]() noexcept {
        // The equal_within_roundoff below has an implicit scale of 1,
        // so the ASSERT may trigger in the case where we really
        // want an entire domain that is very small.
        ASSERT(not equal_within_roundoff(radius, 0.0),
               "Cannot have zero radius");
        return radius;
      }()),
      theta_(acos((z_plane - center_[2]) / radius_)) {
  ASSERT(z_plane != center[2],
         "Plane must intersect sphere at more than one point");
}

double InnerSphereMap::sin_ax_over_x(double x, double ax, double a) noexcept {
  return square(ax) < 6.0 * std::numeric_limits<double>::epsilon()
             ? a
             : sin(ax) / x;
}
DataVector InnerSphereMap::sin_ax_over_x(const DataVector& x,
                                         double a) noexcept {
  DataVector result(x);
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = sin_ax_over_x(x[i], a * x[i], a);
  }
  return result;
}

double InnerSphereMap::dlogx_sin_ax_over_x(double x, double ax,
                                           double a) noexcept {
  return square(ax) < 10.0 * std::numeric_limits<double>::epsilon()
             ? -cube(a) / 3.0
             : (a * cos(ax) - sin(ax) / x) / square(x);
}
DataVector InnerSphereMap::dlogx_sin_ax_over_x(const DataVector& x,
                                               double a) noexcept {
  DataVector result(x);
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = dlogx_sin_ax_over_x(x[i], a * x[i], a);
  }
  return result;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> InnerSphereMap::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type& xbar = source_coords[0];
  const return_type& ybar = source_coords[1];
  const return_type rho = sqrt(square(xbar) + square(ybar));
  const return_type sin_factor = radius_ * sin_ax_over_x(rho, theta_);
  const return_type z = radius_ * cos(rho * theta_) + center_[2];
  const return_type x = sin_factor * xbar + center_[0];
  const return_type y = sin_factor * ybar + center_[1];
  return std::array<return_type, 3>{{std::move(x), std::move(y), std::move(z)}};
}

boost::optional<std::array<double, 3>> InnerSphereMap::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  const double x = target_coords[0] - center_[0];
  const double y = target_coords[1] - center_[1];
  const double z = target_coords[2] - center_[2];
  const double r = sqrt(square(x) + square(y) + square(z));
  // The equal_within_roundoff below has an implicit scale of 1,
  // so the inverse may fail if radius_ is very small on purpose,
  // e.g. if we really want a tiny tiny domain for some reason.
  if (not equal_within_roundoff(r, radius_)) {
    return boost::none;
  }

  // Compute sin^2(rho theta).
  const double sin_squared_rho_theta = (square(x) + square(y)) / square(r);
  // Compute sin(rho theta)/rho.
  // If sin^2(rho theta) is small,
  // use arcsin(q) = q(1 + q^2/6 + 3 q^4/40 + ...)
  // for q = sin(rho theta).
  double sin_rho_theta_over_rho = 0.0;
  if (square(sin_squared_rho_theta) <
      (40.0 / 3.0) * std::numeric_limits<double>::epsilon()) {
    sin_rho_theta_over_rho = theta_ * (1.0 - sin_squared_rho_theta / 6.0);
  } else {
    const double rho = asin(sqrt(sin_squared_rho_theta)) / theta_;
    sin_rho_theta_over_rho = sqrt(sin_squared_rho_theta) / rho;
  }

  // Note about the division in the next line: The above check of r
  // versus radius_ means that r cannot be zero unless the radius of
  // the sphere (a map parameter) is chosen to be zero, which would
  // make the map singular.  Also sin_rho_theta_over_rho cannot be
  // zero unless theta_ (a map parameter) is chosen to be zero, which
  // also would make the map singular.
  const double xbar = x / (r * sin_rho_theta_over_rho);
  const double ybar = y / (r * sin_rho_theta_over_rho);
  const double rho_squared = square(xbar) + square(ybar);
  if (rho_squared > 1.0 and not equal_within_roundoff(rho_squared, 1.0)) {
    return boost::none;
  }

  return std::array<double, 3>{{xbar, ybar, -1.0}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
InnerSphereMap::jacobian(const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type& xbar = source_coords[0];
  const return_type& ybar = source_coords[1];
  const return_type rho = sqrt(square(xbar) + square(ybar));
  const return_type sin_factor = radius_ * sin_ax_over_x(rho, theta_);
  const return_type d_sin_factor = radius_ * dlogx_sin_ax_over_x(rho, theta_);

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dz/dxbar
  get<2, 0>(jacobian_matrix) = -sin_factor * theta_ * xbar;
  // dz/dybar
  get<2, 1>(jacobian_matrix) = -sin_factor * theta_ * ybar;
  // dx/dxbar
  get<0, 0>(jacobian_matrix) = d_sin_factor * square(xbar) + sin_factor;
  // dx/dybar
  get<0, 1>(jacobian_matrix) = d_sin_factor * xbar * ybar;
  // dy/dxbar
  get<1, 0>(jacobian_matrix) = d_sin_factor * ybar * xbar;
  // dy/dybar
  get<1, 1>(jacobian_matrix) = d_sin_factor * square(ybar) + sin_factor;

  return jacobian_matrix;
}

// This is really a 2-dimensional inverse jacobian because zbar = -1
// and xbar,ybar depend only on y,z (given y,z we know x since
// x,y,z must be on the sphere).
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
InnerSphereMap::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type& xbar = source_coords[0];
  const return_type& ybar = source_coords[1];
  const return_type rho = sqrt(square(xbar) + square(ybar));
  // Let q = sin(rho theta)/rho
  const return_type q = sin_ax_over_x(rho, theta_);
  const return_type dlogrho_q = dlogx_sin_ax_over_x(rho, theta_);
  const return_type one_over_r_q = 1.0 / (q * radius_);
  const return_type tmp =
      one_over_r_q * dlogrho_q / (q + square(rho) * dlogrho_q);

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dxbar/dx
  get<0, 0>(inv_jacobian_matrix) = one_over_r_q - square(xbar) * tmp;
  // dxbar/dy
  get<0, 1>(inv_jacobian_matrix) = -xbar * ybar * tmp;
  // dybar/dx
  get<1, 0>(inv_jacobian_matrix) = get<0, 1>(inv_jacobian_matrix);
  // dybar/dy
  get<1, 1>(inv_jacobian_matrix) = one_over_r_q - square(ybar) * tmp;

  return inv_jacobian_matrix;
}

// Consider a sphere with center 'sphere_center' and radius 'radius',
// and let 'proj_center' and 'src_point' be two arbitrary points.
//
// Consider the line passing through 'proj_center' and 'src_point'.
// This ray intersects the sphere at either zero, one, or two points.
// If there are zero such points, then 'scale_factor' is undefined.
// If there is at least one intersection point, then
// 'scale_factor' is a scalar defined by the relation
// intersection_point - proj_center = scale_factor * (src_point - proj_center).
// See below for what happens when there are two intersection points.
//
// More detail:
//
// Let x = 'src_point', p = 'proj_center', c = 'sphere_center',
// and r = 'radius'.
//
// then if y = 'intersection_point' and lambda = 'scale_factor', then
//  y = p + (x-p) \lambda
//
// To solve for 'scale_factor', we note that y is on the surface of
// the sphere, so
//   | y - c |^2 = r^2
// or
//   | p-c + (x-p)\lambda |^2 = r^2.
//
// This is a quadratic equation for \lambda in terms of x, p, c, and r.
//
// For the forward map, we want a positive root \lambda that is
// greater than or equal to unity.  If there are two such roots (this
// occurs if all of sphere_2 is closer to src_point than to
// proj_center) we take the smaller one. The function scale_factor
// returns the desired root, and ASSERTS if there are not two real
// roots or if there is no positive root \lambda >= 1.
//
// For the inverse map, we provide a function called try_scale_factor
// that has additional options as to which root to choose.
// try_scale_factor returns boost::none if the roots are not as
// expected (i.e. if the inverse map was called for a point not in the
// range of the map).
template <typename T>
tt::remove_cvref_wrap_t<T> scale_factor(
    const std::array<tt::remove_cvref_wrap_t<T>, 3>& src_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius) noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  // quadratic equation is
  // a x^2 + b x + c = 0
  const return_type a = square(src_point[0] - proj_center[0]) +
                        square(src_point[1] - proj_center[1]) +
                        square(src_point[2] - proj_center[2]);
  const return_type b =
      2.0 *
      ((src_point[0] - proj_center[0]) * (proj_center[0] - sphere_center[0]) +
       (src_point[1] - proj_center[1]) * (proj_center[1] - sphere_center[1]) +
       (src_point[2] - proj_center[2]) * (proj_center[2] - sphere_center[2]));
  const double c = square(sphere_center[0] - proj_center[0]) +
                   square(sphere_center[1] - proj_center[1]) +
                   square(sphere_center[2] - proj_center[2]) - square(radius);
  return smallest_root_greater_than_value_within_roundoff(
      a, b, make_with_value<return_type>(a, c), 1.0);
}

boost::optional<double> try_scale_factor(
    const std::array<double, 3>& src_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius,
    const bool pick_larger_root,
    const bool pick_root_greater_than_one) noexcept {
  // We solve the quadratic for (scale_factor-1) instead of scale_factor to
  // avoid roundoff problems when scale_factor is very nearly equal to unity.
  // Note that scale_factor==1 will occur when src_point is on the sphere, which
  // happens when inverse-mapping the boundaries.

  // quadratic equation is
  // a x^2 + b x + c = 0
  const double a = square(src_point[0] - proj_center[0]) +
                   square(src_point[1] - proj_center[1]) +
                   square(src_point[2] - proj_center[2]);
  const double b =
      2.0 *
      ((src_point[0] - proj_center[0]) * (src_point[0] - sphere_center[0]) +
       (src_point[1] - proj_center[1]) * (src_point[1] - sphere_center[1]) +
       (src_point[2] - proj_center[2]) * (src_point[2] - sphere_center[2]));
  const double c = square(sphere_center[0] - src_point[0]) +
                   square(sphere_center[1] - src_point[1]) +
                   square(sphere_center[2] - src_point[2]) - square(radius);

  double x0 = std::numeric_limits<double>::signaling_NaN();
  double x1 = std::numeric_limits<double>::signaling_NaN();
  const int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
  if (num_real_roots == 2) {
    // We solved for scale_factor-1 above, so add 1 to get scale_factor.
    x0 += 1.0;
    x1 += 1.0;
    if (equal_within_roundoff(x0, 1.0)) {
      x0 = 1.0;
    }
    if (equal_within_roundoff(x1, 1.0)) {
      x1 = 1.0;
    }
    if (pick_root_greater_than_one) {
      // For the inverse map, we want the a scale_factor s such that
      // s >= 1. Note that gsl_poly_solve_quadratic returns x0 < x1.
      // have three cases:
      //  a) x0 < x1 < 1         ->   error
      //  b) x0 < 1 <= x1        ->   Choose x1
      //  c) 1 <= x0 < x1        ->   choose based on pick_larger_root
      if (x0 >= 1.0 and not pick_larger_root) {
        return x0;
      } else if (x1 >= 1.0) {
        return x1;
      } else {
        return boost::none;
      }
    } else {
      // For the inverse map, we want a scale_factor s such that 0 < s <= 1.
      // Note that gsl_poly_solve_quadratic returns x0 < x1.
      // So we have six cases:
      //  a) x0 < x1 <= 0        ->   error
      //  b) x0 <= 0 < x1 <= 1   ->   Choose x1
      //  c) x0 <= 0 and x1 > 1  ->   error
      //  d) 0 < x0 < x1 <= 1      ->   Choose according to pick_larger_root
      //  e) 0 < x0 <= 1 < x1      ->   Choose x0
      //  f) 1 < x0 < x1           ->   error
      if (x0 <= 0.0) {
        if (x1 > 0.0 and x1 <= 1.0) {
          return x1; // b)
        } else {
          return boost::none;  // a) and c)
        }
      } else if (x0 <= 1.0) {
        if (x1 > 1.0) {
          return x0;  // e)
        } else {
          return pick_larger_root ? x1 : x0;  // d)
        }
      } else {
        return boost::none;  // f)
      }
    }
  } else if (num_real_roots == 1) {
    // We solved for scale_factor-1 above, so add 1 to get scale_factor.
    x0 += 1.0;
    if (equal_within_roundoff(x0, 1.0)) {
      x0 = 1.0;
    }
    if (pick_root_greater_than_one) {
      if (x0 < 1.0) {
        return boost::none;
      } else {
        return x0;
      }
    } else {
      if (x0 <= 0.0 or x0 > 1.0) {
        return boost::none;
      } else {
        return x0;
      }
    }
  } else {
    return boost::none;
  }
}

// Let xbar^i be 'src_point' and lambda be 'scale_factor' as computed by
// the function scale_factor, and let 'intersection_point' be the computed
// intersection point derived from lambda and xbar^i.
//
// d_scale_factor_d_src_point computes partial lambda/partial xbar^i for
// all i.
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> d_scale_factor_d_src_point(
    const std::array<tt::remove_cvref_wrap_t<T>, 3>& intersection_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center,
    const tt::remove_cvref_wrap_t<T>& lambda) noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type lambda_over_denominator =
      square(lambda) / (square(intersection_point[0] - proj_center[0]) +
                        square(intersection_point[1] - proj_center[1]) +
                        square(intersection_point[2] - proj_center[2]) +
                        ((intersection_point[0] - proj_center[0]) *
                             (proj_center[0] - sphere_center[0]) +
                         (intersection_point[1] - proj_center[1]) *
                             (proj_center[1] - sphere_center[1]) +
                         (intersection_point[2] - proj_center[2]) *
                             (proj_center[2] - sphere_center[2])));
  auto result =
      make_with_value<std::array<return_type, 3>>(lambda_over_denominator, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) =
        lambda_over_denominator *
        (gsl::at(sphere_center, i) - gsl::at(intersection_point, i));
  }
  return result;
}
}  // namespace CylindricalEndcap_detail

CylindricalEndcap::CylindricalEndcap(const std::array<double, 3>& center_one,
                                     const std::array<double, 3>& center_two,
                                     const std::array<double, 3>& proj_center,
                                     double radius_one, double radius_two,
                                     double z_plane) noexcept
    : center_one_(center_one),
      center_two_(center_two),
      proj_center_(proj_center),
      radius_one_(radius_one),
      radius_two_(radius_two),
      z_plane_(z_plane) {

#ifdef SPECTRE_DEBUG
  // There are two types of sanity checks here on the map parameters.
  // 1) ASSERTS that guarantee that the map is invertible.
  // 2) ASSERTS that guarantee that the map parameters fall within
  //    the range tested by the unit tests (which is the range in which
  //    the map is expected to be used).
  //
  // There are two reasons why 1) and 2) are not the same:
  //
  // a) It is possible to choose parameters such that the map is
  //    invertible but the resulting geometry has very sharp angles,
  //    very large or ill-conditioned Jacobians, or both.  We want to
  //    avoid such cases.
  // b) We do not want to waste effort testing the map for parameters
  //    that we don't expect to be used.  For example, we demand
  //    here that proj_center and sphere_one are contained within
  //    sphere_two, but the map should still be valid for some choices
  //    of parameters where sphere_one and sphere_two are disjoint;
  //    allowing those parameter choices would involve much more
  //    complicated logic to determine whether the map produces shapes
  //    with sharp angles or large jacobians, and it would involve more
  //    complicated unit tests to cover those possibilities.

  // First test for invertibility.

  // Consider the intersection of sphere_one and the plane formed by
  // z_plane.  Call it circle_one.  Consider the cone with apex
  // center_one that intersects sphere_one on circle_one. This cone
  // has an opening angle 2*theta.  Call the cone 'cone_one'.
  const double cos_theta = (z_plane - center_one_[2]) / radius_one_;

  // Now consider a different cone, cone_new, constructed so that
  // cone_new and cone_one intersect each other on circle_one at right
  // angles.  The apex of cone_new is cone_new_apex, defined as
  // follows:
  const std::array<double, 3> cone_new_apex = {
      center_one_[0], center_one_[1], center_one_[2] + radius_one_ / cos_theta};
  // Cone_new opens in the -z direction with opening angle 2*(pi/2-theta).

  // A necessary condition for invertibility is that proj_center lies
  // either inside of cone_new or inside of the reflection of cone_new
  // (a cone with apex cone_new_apex but opening in the +x direction
  // with opening angle 2*(pi/2-theta)).

  // Determine the angle of proj_center relative to cone_new_apex. Call this
  // angle alpha.
  const double dist_cone_proj =
      sqrt(square(cone_new_apex[0] - proj_center_[0]) +
           square(cone_new_apex[1] - proj_center_[1]) +
           square(cone_new_apex[2] - proj_center_[2]));
  const double cos_alpha = (cone_new_apex[2] - proj_center[2]) / dist_cone_proj;

  // Now make sure that alpha < pi/2-theta.
  // The cone on either side of cone_new_apex is ok, so we use abs below.
  ASSERT(acos(abs(cos_alpha)) < abs(asin(cos_theta)),
         "The arguments passed into the CylindricalEndcap constructor "
         "yield a noninvertible map.");

  // Another necessary condition for invertibility is that proj_center
  // cannot lie between sphere_one and cone_new_apex.
  const double proj_radius_one = sqrt(square(center_one_[0] - proj_center_[0]) +
                                      square(center_one_[1] - proj_center_[1]) +
                                      square(center_one_[2] - proj_center_[2]));
  ASSERT(proj_center_[2] > cone_new_apex[2] or proj_center_[2] < z_plane_ or
             proj_radius_one < radius_one_,
         "The arguments passed into the CylindricalEndcap constructor "
         "yield a noninvertible map.");

  // Other sanity checks that may be relaxed if there is more logic
  // added and more unit tests to test these cases.

  ASSERT(proj_center_[2] < z_plane_,
         "CylindricalEndcap: The map hasn't been tested for this "
         "configuration. The map may still be invertible, but further "
         "testing would be needed to ensure that jacobians are not "
         "ill-conditioned.");

  ASSERT(abs(cos_theta) <= 0.9,
         "CylindricalEndcap: z_plane is too far from the center of sphere_one. "
             << "cos_theta = " << cos_theta
             << ". If |cos_theta| > 1 the map is singular.  If 0.9 < "
                "|cos_theta| < 1 then the map is not singular, but the "
                "jacobians are likely to be large and the map has not been "
                "tested for these parameters.");
  ASSERT(
      abs(cos_theta) >= 0.1,
      "CylindricalEndcap: z_plane is too close to the center of sphere_one. "
          << "cos_theta = " << cos_theta
          << ". The map is not singular, but the jacobians are likely to be "
             "large "
             "and the map has not been tested for this choice of parameters.");

  const double dist_spheres = sqrt(square(center_one_[0] - center_two_[0]) +
                                   square(center_one_[1] - center_two_[1]) +
                                   square(center_one_[2] - center_two_[2]));
  ASSERT(dist_spheres + radius_one < radius_two,
         "CylindricalEndcap: The map has been tested only for the case when "
         "sphere_one is contained inside sphere_two");

  const double proj_radius_two = sqrt(square(center_two_[0] - proj_center_[0]) +
                                      square(center_two_[1] - proj_center_[1]) +
                                      square(center_two_[2] - proj_center_[2]));
  ASSERT(proj_radius_two < radius_two_,
         "CylindricalEndcap: The map has been tested only for the case when "
         "proj_center is contained inside sphere_two");

  // Check if we are too close to singular.
  ASSERT(acos(abs(cos_alpha)) < abs(0.95 * asin(cos_theta)),
         "CylindricalEndcap: Parameters are close to where the map becomes "
         "non-invertible.  The map has not been tested for this case.");

  // Check if opening angle is small enough.
  const double max_opening_angle = M_PI / 3.0;
  const double max_proj_center_z =
      z_plane_ -
      radius_one_ * sqrt(1.0 - square(cos_theta)) / tan(max_opening_angle);
  ASSERT(proj_center_[2] < max_proj_center_z,
         "CylindricalEndcap: proj_center is too close to z_plane. The "
         "map has not been tested for this case.");
  const double tan_beta = sqrt(square(center_one_[0] - proj_center_[0]) +
                               square(center_one_[1] - proj_center_[1])) /
                          (max_proj_center_z - proj_center_[2]);
  ASSERT(tan_beta < tan(max_opening_angle),
         "CylindricalEndcap: opening angle is too large. The map has not "
         "been tested for this case.");
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> CylindricalEndcap::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  CylindricalEndcap_detail::InnerSphereMap inner_map(center_one_, radius_one_,
                                                     z_plane_);

  // lower_coords are the mapped coords on the surface of sphere 1.
  const std::array<return_type, 3> lower_coords = inner_map(source_coords);

  // upper_coords are the mapped coords on the surface of sphere 2.
  const auto lambda = CylindricalEndcap_detail::scale_factor<return_type>(
      lower_coords, proj_center_, center_two_, radius_two_);

  std::array<return_type, 3> upper_coords = lower_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  // mapped_coords goes linearly from lower_coords to upper_coords
  // as zbar goes from -1 to 1.
  const return_type& zbar = source_coords[2];
  auto mapped_coords = make_with_value<std::array<return_type, 3>>(zbar, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(mapped_coords, i) =
        gsl::at(lower_coords, i) +
        (gsl::at(upper_coords, i) - gsl::at(lower_coords, i)) * 0.5 *
            (zbar + 1.0);
  }
  return mapped_coords;
}

boost::optional<std::array<double, 3>> CylindricalEndcap::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  // If target_coords are outside of sphere_two then the point
  // is out of range.
  const double radius_squared = square(target_coords[0] - center_two_[0]) +
                                square(target_coords[1] - center_two_[1]) +
                                square(target_coords[2] - center_two_[2]);
  // The equal_within_roundoff below has an implicit scale of 1,
  // so the inverse may fail if radius_two_ is intentionally very
  // small, e.g. if we really want a tiny tiny domain.
  if (radius_squared > square(radius_two_) and
      not equal_within_roundoff(radius_squared, square(radius_two_))) {
    return boost::none;
  }

  // Try to find lambda_tilde going from target_coords to *sphere 1*.
  // This lambda_tilde should be positive and less than or equal to unity.
  // If there are two such roots, we choose based on where the points are.
  const bool choose_larger_root = target_coords[2] > proj_center_[2];
  const auto lambda_tilde = CylindricalEndcap_detail::try_scale_factor(
      target_coords, proj_center_, center_one_, radius_one_, choose_larger_root,
      false);

  // Cannot find scale factor, so we are out of range of the map.
  if (not lambda_tilde) {
    return boost::none;
  }

  // If lambda_tilde is negative, then we are on the wrong side of
  // the sphere.  If lambda_tilde is larger than unity, then we are
  // outside the sphere.
  if (lambda_tilde.get() <= 0.0 or
      (lambda_tilde.get() > 1.0 and
       not equal_within_roundoff(lambda_tilde.get(), 1.0))) {
    return boost::none;
  }

  // Try to find lambda_bar going from target_coords to *sphere 2*.
  // This lambda_bar should be positive and greater than or equal to unity.
  const auto lambda_bar = CylindricalEndcap_detail::try_scale_factor(
      target_coords, proj_center_, center_two_, radius_two_, false, true);

  // Cannot find scale factor, so we are out of range of the map.
  if (not lambda_bar) {
    return boost::none;
  }

  // Compute zbar in a roundoff-friendly way that should get the
  // correct values at zbar=1 and zbar=-1.
  double zbar = 0.0;
  if (equal_within_roundoff(lambda_tilde.get(), 1.0, 1.e-5)) {
    // Get zbar correct for zbar near -1
    zbar = 2.0 * (lambda_tilde.get() - 1.0) /
               (lambda_tilde.get() - lambda_bar.get()) -
           1.0;
  } else {
    // Get zbar correct for zbar near +1
    zbar = 2.0 * (lambda_bar.get() - 1.0) /
               (lambda_tilde.get() - lambda_bar.get()) +
           1.0;
  }

  std::array<double, 3> lower_coords = target_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(lower_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(target_coords, i) - gsl::at(proj_center_, i)) *
            lambda_tilde.get();
  }

  // Map lower_coords back to (xbar,ybar,zbar)
  // Here zbar will be -1.
  CylindricalEndcap_detail::InnerSphereMap inner_map(center_one_, radius_one_,
                                                     z_plane_);
  boost::optional<std::array<double, 3>> orig_coords =
      inner_map.inverse(lower_coords);

  if (orig_coords) {
    orig_coords.get()[2] = zbar;
  }

  // Root polishing.
  // Here we do a single Newton iteration to get the
  // inverse to agree with the forward map to the level of machine
  // roundoff that is required by the unit tests.
  // Without the root polishing, the unit tests occasionally fail
  // the 'inverse(map(x))=x' test at a level slightly above roundoff.
  if (orig_coords) {
    const auto inv_jac = inv_jacobian(orig_coords.get());
    const auto mapped_coords = operator()(orig_coords.get());
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(orig_coords.get(), i) +=
            (gsl::at(target_coords, j) - gsl::at(mapped_coords, j)) *
            inv_jac.get(i, j);
      }
    }
  }

  return orig_coords;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalEndcap::jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  CylindricalEndcap_detail::InnerSphereMap inner_map(center_one_, radius_one_,
                                                     z_plane_);

  // lower_coords are the mapped coords on the surface of sphere 1.
  const std::array<return_type, 3> lower_coords = inner_map(source_coords);
  const auto lambda = CylindricalEndcap_detail::scale_factor<return_type>(
      lower_coords, proj_center_, center_two_, radius_two_);

  // upper_coords are the mapped coords on the surface of sphere 2.
  std::array<return_type, 3> upper_coords = lower_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  const return_type& zbar = source_coords[2];

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dx^i/dzbar is easy, do it first.
  for (size_t i = 0; i < 3; ++i) {
    jacobian_matrix.get(i, 2) =
        0.5 * (gsl::at(upper_coords, i) - gsl::at(lower_coords, i));
  }

  // Do the easiest of the terms involving the inner map.
  const auto d_inner = inner_map.jacobian(source_coords);
  const return_type lambda_factor = 0.5 * (1.0 - zbar + lambda * (1.0 + zbar));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian_matrix.get(i, j) += lambda_factor * d_inner.get(i, j);
    }
  }

  // Do lambda term, which is the most complicated one.
  const auto d_lambda_d_lower_coords =
      CylindricalEndcap_detail::d_scale_factor_d_src_point<return_type>(
          upper_coords, proj_center_, center_two_, lambda);
  const return_type z_factor = 0.5 * (1.0 + zbar);
  for (size_t j = 0; j < 3; ++j) {
    auto temp = make_with_value<return_type>(z_factor, 0.0);
    for (size_t k = 0; k < 3; ++k) {
      temp += gsl::at(d_lambda_d_lower_coords, k) * d_inner.get(k, j);
    }
    temp *= z_factor;
    for (size_t i = 0; i < 3; ++i) {
      jacobian_matrix.get(i, j) +=
          temp * (gsl::at(lower_coords, i) - gsl::at(proj_center_, i));
    }
  }

  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalEndcap::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  CylindricalEndcap_detail::InnerSphereMap inner_map(center_one_, radius_one_,
                                                     z_plane_);

  // lower_coords are the mapped coords on the surface of sphere 1.
  const std::array<return_type, 3> lower_coords = inner_map(source_coords);

  // Lambda is the scale factor between lower coords and upper coords.
  const auto lambda = CylindricalEndcap_detail::scale_factor<return_type>(
      lower_coords, proj_center_, center_two_, radius_two_);

  // upper_coords are the mapped coords on the surface of sphere 2.
  std::array<return_type, 3> upper_coords = lower_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  // Derivative of lambda
  const auto d_lambda_d_lower_coords =
      CylindricalEndcap_detail::d_scale_factor_d_src_point<return_type>(
          upper_coords, proj_center_, center_two_, lambda);

  // Lambda_tilde is the scale factor between mapped coords and lower coords.
  // We can compute it with a shortcut because there is a relationship
  // between lambda, lambda_tilde, and zbar.
  const return_type& zbar = source_coords[2];
  const return_type lambda_tilde =
      1.0 / (1.0 - 0.5 * (1.0 - lambda) * (1.0 + zbar));

  // Derivative of lambda_tilde
  const auto d_lambda_tilde_d_mapped_coords =
      CylindricalEndcap_detail::d_scale_factor_d_src_point<return_type>(
          lower_coords, proj_center_, center_one_, lambda_tilde);

  // Derivatives of zbar with respect to lambda and lambda_tilde
  const return_type dzbar_dlambda = (1.0 + zbar) / (1.0 - lambda);
  const return_type dzbar_dlambdatilde =
      2.0 / square(lambda_tilde) / (1.0 - lambda);

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dzbar/dx^i
  for (size_t i = 0; i < 3; ++i) {
    auto tmp = make_with_value<return_type>(lambda, 0.0);
    for (size_t j = 0; j < 3; ++j) {
      tmp += gsl::at(d_lambda_d_lower_coords, j) *
             (gsl::at(lower_coords, j) - gsl::at(proj_center_, j));
    }
    inv_jacobian_matrix.get(2, i) =
        gsl::at(d_lambda_d_lower_coords, i) * dzbar_dlambda * lambda_tilde +
        gsl::at(d_lambda_tilde_d_mapped_coords, i) *
            (dzbar_dlambdatilde + tmp * dzbar_dlambda / lambda_tilde);
  }

  // dxbar/dx^i and dybar/dx^i
  const auto dxbar_dx_inner = inner_map.inv_jacobian(source_coords);

  for (size_t i = 0; i < 2; ++i) {  // Loop up to 2; we already did zbar
    auto tmp = make_with_value<return_type>(lambda, 0.0);
    for (size_t k = 0; k < 3; ++k) {
      tmp += (gsl::at(lower_coords, k) - gsl::at(proj_center_, k)) *
             dxbar_dx_inner.get(i, k);
    }
    for (size_t j = 0; j < 3; ++j) {
      inv_jacobian_matrix.get(i, j) =
          tmp * (gsl::at(d_lambda_tilde_d_mapped_coords, j) / lambda_tilde) +
          lambda_tilde * dxbar_dx_inner.get(i, j);
    }
  }
  return inv_jacobian_matrix;
}

void CylindricalEndcap::pup(PUP::er& p) noexcept {
  p | center_one_;
  p | center_two_;
  p | proj_center_;
  p | radius_one_;
  p | radius_two_;
  p | z_plane_;
}

bool operator==(const CylindricalEndcap& lhs,
                const CylindricalEndcap& rhs) noexcept {
  return lhs.center_one_ == rhs.center_one_ and
         lhs.center_two_ == rhs.center_two_ and
         lhs.proj_center_ == rhs.proj_center_ and
         lhs.radius_one_ == rhs.radius_one_ and
         lhs.radius_two_ == rhs.radius_two_ and lhs.z_plane_ == rhs.z_plane_;
}

bool operator!=(const CylindricalEndcap& lhs,
                const CylindricalEndcap& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  CylindricalEndcap::operator()(                                               \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  CylindricalEndcap::jacobian(const std::array<DTYPE(data), 3>& source_coords) \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  CylindricalEndcap::inv_jacobian(                                             \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace domain::CoordinateMaps
