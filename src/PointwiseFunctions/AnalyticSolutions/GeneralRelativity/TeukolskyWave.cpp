// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

namespace {

// The spherical-coordinate implementation is not valid near the origin.
// Evaluations at or below this radius from the wave center trigger an error.
constexpr double origin_exclusion_radius = 0.1;

// Points within this distance of the z axis (cylindrical_radius == 0) are
// treated as being exactly on the axis. In this case, the value on the
// axis is found by averaging the values at two small offsets from the
// axis in orthogonal directions.
constexpr double axis_coordinate_tolerance = 1.0e-14;

// When evaluating the axis limit (cylindrical_radius == 0), approach the
// axis from a transverse offset of this size from two orthogonal directions
// and then average the result to get the value on the limit.
constexpr double axis_limit_offset = 1.0e-8;

template <typename DataType>
std::pair<tnsr::ii<DataType, 3, Frame::Inertial>,
          tnsr::ii<DataType, 3, Frame::Inertial>>
perturbation_and_dt_perturbation(const DataType& centered_x,
                                 const DataType& centered_y,
                                 const DataType& centered_z, const double t,
                                 const double amplitude, const int mode,
                                 const bool even_parity, const bool ingoing,
                                 const double radius, const double width) {
  const DataType radius_from_center =
      sqrt(square(centered_x) + square(centered_y) + square(centered_z));
  const DataType cylindrical_radius =
      sqrt(square(centered_x) + square(centered_y));

  const DataType axis_mask =
      step_function(cylindrical_radius - axis_coordinate_tolerance);
  const DataType safe_cylindrical_radius =
      cylindrical_radius + (1.0 - axis_mask);

  const DataType cos_theta = centered_z / radius_from_center;
  const DataType sin_theta = cylindrical_radius / radius_from_center;
  const DataType cos_phi =
      axis_mask * centered_x / safe_cylindrical_radius + (1.0 - axis_mask);
  const DataType sin_phi = axis_mask * centered_y / safe_cylindrical_radius;
  const DataType sin_2phi = 2.0 * sin_phi * cos_phi;
  const DataType cos_2phi = square(cos_phi) - square(sin_phi);

  const DataType y_profile = radius_from_center - radius + (ingoing ? t : -t);
  const double minus_two_over_width_squared = -2.0 / square(width);
  const DataType profile = amplitude * exp(-square(y_profile) / square(width));
  const DataType profile_1 = minus_two_over_width_squared * y_profile * profile;
  const DataType profile_2 =
      minus_two_over_width_squared * (profile + y_profile * profile_1);
  const DataType profile_3 =
      minus_two_over_width_squared * (2.0 * profile_1 + y_profile * profile_2);
  const DataType profile_4 =
      minus_two_over_width_squared * (3.0 * profile_2 + y_profile * profile_3);
  const DataType profile_5 =
      minus_two_over_width_squared * (4.0 * profile_3 + y_profile * profile_4);

  auto spherical_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::NoFrame>>(centered_x, 0.0);
  auto dt_spherical_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::NoFrame>>(centered_x, 0.0);

  if (even_parity) {
    auto angular_rr = make_with_value<DataType>(centered_x, 0.0);
    auto angular_rtheta = make_with_value<DataType>(centered_x, 0.0);
    auto angular_rphi = make_with_value<DataType>(centered_x, 0.0);
    auto angular_thetaphi = make_with_value<DataType>(centered_x, 0.0);
    auto angular_theta_theta_1 = make_with_value<DataType>(centered_x, 0.0);
    auto angular_theta_theta_2 = make_with_value<DataType>(centered_x, 0.0);
    auto angular_phi_phi_1 = make_with_value<DataType>(centered_x, 0.0);
    auto angular_phi_phi_2 = make_with_value<DataType>(centered_x, 0.0);
    switch (mode) {
      case -2:
        angular_rr = square(sin_theta) * sin_2phi;
        angular_rtheta = sin_theta * cos_theta * sin_2phi;
        angular_rphi = sin_theta * cos_2phi;
        angular_theta_theta_1 = (1.0 + square(cos_theta)) * sin_2phi;
        angular_theta_theta_2 = -sin_2phi;
        angular_thetaphi = -cos_theta * cos_2phi;
        angular_phi_phi_1 = -angular_theta_theta_1;
        angular_phi_phi_2 = square(cos_theta) * sin_2phi;
        break;
      case -1:
        angular_rr = 2.0 * sin_theta * cos_theta * sin_phi;
        angular_rtheta = (square(cos_theta) - square(sin_theta)) * sin_phi;
        angular_rphi = cos_theta * cos_phi;
        angular_theta_theta_1 = -2.0 * sin_theta * cos_theta * sin_phi;
        angular_thetaphi = sin_theta * cos_phi;
        angular_phi_phi_1 = -angular_theta_theta_1;
        angular_phi_phi_2 = -2.0 * sin_theta * cos_theta * sin_phi;
        break;
      case 0:
        angular_rr = 2.0 - 3.0 * square(sin_theta);
        angular_rtheta = -3.0 * sin_theta * cos_theta;
        angular_theta_theta_1 = 3.0 * square(sin_theta);
        angular_theta_theta_2 = -1.0;
        angular_phi_phi_1 = -angular_theta_theta_1;
        angular_phi_phi_2 = 3.0 * square(sin_theta) - 1.0;
        break;
      case 1:
        angular_rr = 2.0 * sin_theta * cos_theta * cos_phi;
        angular_rtheta = (square(cos_theta) - square(sin_theta)) * cos_phi;
        angular_rphi = -cos_theta * sin_phi;
        angular_theta_theta_1 = -2.0 * sin_theta * cos_theta * cos_phi;
        angular_thetaphi = -sin_theta * sin_phi;
        angular_phi_phi_1 = -angular_theta_theta_1;
        angular_phi_phi_2 = -2.0 * sin_theta * cos_theta * cos_phi;
        break;
      case 2:
        angular_rr = square(sin_theta) * cos_2phi;
        angular_rtheta = sin_theta * cos_theta * cos_2phi;
        angular_rphi = -sin_theta * sin_2phi;
        angular_theta_theta_1 = (1.0 + square(cos_theta)) * cos_2phi;
        angular_theta_theta_2 = -cos_2phi;
        angular_thetaphi = cos_theta * sin_2phi;
        angular_phi_phi_1 = -angular_theta_theta_1;
        angular_phi_phi_2 = square(cos_theta) * cos_2phi;
        break;
      default:
        ERROR("Unsupported Teukolsky mode");
    }

    const DataType radial_a =
        3.0 *
        (profile_2 + (-3.0 * profile_1 + 3.0 * profile / radius_from_center) /
                         radius_from_center) /
        cube(radius_from_center);
    const DataType radial_b =
        -(-profile_3 + (3.0 * profile_2 + (-6.0 * profile_1 +
                                           6.0 * profile / radius_from_center) /
                                              radius_from_center) /
                           radius_from_center) /
        square(radius_from_center);
    const DataType radial_c =
        0.25 *
        (profile_4 + (-2.0 * profile_3 +
                      (9.0 * profile_2 + (-21.0 * profile_1 +
                                          21.0 * profile / radius_from_center) /
                                             radius_from_center) /
                          radius_from_center) /
                         radius_from_center) /
        radius_from_center;

    get<0, 0>(spherical_metric) = radial_a * angular_rr;
    get<0, 1>(spherical_metric) =
        radius_from_center * radial_b * angular_rtheta;
    get<0, 2>(spherical_metric) =
        radius_from_center * radial_b * angular_rphi * sin_theta;
    get<1, 1>(spherical_metric) =
        square(radius_from_center) *
        (radial_c * angular_theta_theta_1 + radial_a * angular_theta_theta_2);
    get<1, 2>(spherical_metric) = square(radius_from_center) *
                                  (radial_a - 2.0 * radial_c) *
                                  angular_thetaphi * sin_theta;
    get<2, 2>(spherical_metric) =
        square(radius_from_center) *
        (radial_c * angular_phi_phi_1 + radial_a * angular_phi_phi_2) *
        square(sin_theta);

    const double propagation_sign = ingoing ? 1.0 : -1.0;
    const DataType dt_radial_a =
        propagation_sign * 3.0 *
        (profile_3 + (-3.0 * profile_2 + 3.0 * profile_1 / radius_from_center) /
                         radius_from_center) /
        cube(radius_from_center);
    const DataType dt_radial_b =
        -propagation_sign *
        (-profile_4 +
         (3.0 * profile_3 +
          (-6.0 * profile_2 + 6.0 * profile_1 / radius_from_center) /
              radius_from_center) /
             radius_from_center) /
        square(radius_from_center);
    const DataType dt_radial_c =
        propagation_sign * 0.25 *
        (profile_5 +
         (-2.0 * profile_4 +
          (9.0 * profile_3 +
           (-21.0 * profile_2 + 21.0 * profile_1 / radius_from_center) /
               radius_from_center) /
              radius_from_center) /
             radius_from_center) /
        radius_from_center;

    get<0, 0>(dt_spherical_metric) = dt_radial_a * angular_rr;
    get<0, 1>(dt_spherical_metric) =
        radius_from_center * dt_radial_b * angular_rtheta;
    get<0, 2>(dt_spherical_metric) =
        radius_from_center * dt_radial_b * angular_rphi * sin_theta;
    get<1, 1>(dt_spherical_metric) =
        square(radius_from_center) * (dt_radial_c * angular_theta_theta_1 +
                                      dt_radial_a * angular_theta_theta_2);
    get<1, 2>(dt_spherical_metric) = square(radius_from_center) *
                                     (dt_radial_a - 2.0 * dt_radial_c) *
                                     angular_thetaphi * sin_theta;
    get<2, 2>(dt_spherical_metric) =
        square(radius_from_center) *
        (dt_radial_c * angular_phi_phi_1 + dt_radial_a * angular_phi_phi_2) *
        square(sin_theta);
  } else {
    auto angular_rtheta = make_with_value<DataType>(centered_x, 0.0);
    auto angular_rphi = make_with_value<DataType>(centered_x, 0.0);
    auto angular_theta_theta = make_with_value<DataType>(centered_x, 0.0);
    auto angular_thetaphi = make_with_value<DataType>(centered_x, 0.0);
    auto angular_phi_phi = make_with_value<DataType>(centered_x, 0.0);
    switch (mode) {
      case -2:
        angular_rtheta = 4.0 * sin_theta * sin_2phi;
        angular_rphi = 4.0 * sin_theta * cos_theta * cos_2phi;
        angular_theta_theta = -2.0 * cos_theta * sin_2phi;
        angular_thetaphi = -(2.0 - square(sin_theta)) * cos_2phi;
        angular_phi_phi = 2.0 * cos_theta * sin_2phi;
        break;
      case -1:
        angular_rtheta = -2.0 * cos_theta * sin_phi;
        angular_rphi = -2.0 * (square(cos_theta) - square(sin_theta)) * cos_phi;
        angular_theta_theta = -sin_theta * sin_phi;
        angular_thetaphi = -cos_theta * sin_theta * cos_phi;
        angular_phi_phi = sin_theta * sin_phi;
        break;
      case 0:
        angular_rphi = -4.0 * cos_theta * sin_theta;
        angular_thetaphi = -square(sin_theta);
        break;
      case 1:
        angular_rtheta = -2.0 * cos_theta * cos_phi;
        angular_rphi = 2.0 * (square(cos_theta) - square(sin_theta)) * sin_phi;
        angular_theta_theta = -sin_theta * cos_phi;
        angular_thetaphi = cos_theta * sin_theta * sin_phi;
        angular_phi_phi = sin_theta * cos_phi;
        break;
      case 2:
        angular_rtheta = 4.0 * sin_theta * cos_2phi;
        angular_rphi = -4.0 * sin_theta * cos_theta * sin_2phi;
        angular_theta_theta = -2.0 * cos_theta * cos_2phi;
        angular_thetaphi = (2.0 - square(sin_theta)) * sin_2phi;
        angular_phi_phi = 2.0 * cos_theta * cos_2phi;
        break;
      default:
        ERROR("Unsupported Teukolsky mode");
    }

    const DataType radial_k =
        (profile_2 + (-3.0 * profile_1 + 3.0 * profile / radius_from_center) /
                         radius_from_center) /
        square(radius_from_center);
    const DataType radial_l =
        (-profile_3 + (2.0 * profile_2 +
                       (-3.0 * profile_1 + 3.0 * profile / radius_from_center) /
                           radius_from_center) /
                          radius_from_center) /
        radius_from_center;

    get<0, 1>(spherical_metric) =
        radius_from_center * radial_k * angular_rtheta;
    get<0, 2>(spherical_metric) =
        radius_from_center * radial_k * angular_rphi * sin_theta;
    get<1, 1>(spherical_metric) =
        square(radius_from_center) * radial_l * angular_theta_theta;
    get<1, 2>(spherical_metric) =
        square(radius_from_center) * radial_l * angular_thetaphi * sin_theta;
    get<2, 2>(spherical_metric) = square(radius_from_center) * radial_l *
                                  angular_phi_phi * square(sin_theta);

    const double propagation_sign = ingoing ? 1.0 : -1.0;
    const DataType dt_radial_k =
        propagation_sign *
        (profile_3 + (-3.0 * profile_2 + 3.0 * profile_1 / radius_from_center) /
                         radius_from_center) /
        square(radius_from_center);
    const DataType dt_radial_l =
        propagation_sign *
        (-profile_4 +
         (2.0 * profile_3 +
          (-3.0 * profile_2 + 3.0 * profile_1 / radius_from_center) /
              radius_from_center) /
             radius_from_center) /
        radius_from_center;

    get<0, 1>(dt_spherical_metric) =
        radius_from_center * dt_radial_k * angular_rtheta;
    get<0, 2>(dt_spherical_metric) =
        radius_from_center * dt_radial_k * angular_rphi * sin_theta;
    get<1, 1>(dt_spherical_metric) =
        square(radius_from_center) * dt_radial_l * angular_theta_theta;
    get<1, 2>(dt_spherical_metric) =
        square(radius_from_center) * dt_radial_l * angular_thetaphi * sin_theta;
    get<2, 2>(dt_spherical_metric) = square(radius_from_center) * dt_radial_l *
                                     angular_phi_phi * square(sin_theta);
  }

  // Jacobian dx^{spherical}/dx^{Cartesian} used by to_different_frame.
  Jacobian<DataType, 3, Frame::Inertial, Frame::NoFrame> jacobian =
      make_with_value<Jacobian<DataType, 3, Frame::Inertial, Frame::NoFrame>>(
          centered_x, 0.0);
  get<0, 0>(jacobian) = sin_theta * cos_phi;
  get<0, 1>(jacobian) = sin_theta * sin_phi;
  get<0, 2>(jacobian) = cos_theta;
  get<1, 0>(jacobian) = cos_theta * cos_phi / radius_from_center;
  get<1, 1>(jacobian) = cos_theta * sin_phi / radius_from_center;
  get<1, 2>(jacobian) = -sin_theta / radius_from_center;
  const DataType sin_theta_for_phi = sin_theta + (1.0 - axis_mask);
  get<2, 0>(jacobian) = -sin_phi / (radius_from_center * sin_theta_for_phi);
  get<2, 1>(jacobian) = cos_phi / (radius_from_center * sin_theta_for_phi);

  auto perturbation = transform::to_different_frame(spherical_metric, jacobian);
  auto dt_perturbation =
      transform::to_different_frame(dt_spherical_metric, jacobian);
  return {std::move(perturbation), std::move(dt_perturbation)};
}

template <typename DataType>
std::pair<tnsr::ii<DataType, 3, Frame::Inertial>,
          tnsr::ii<DataType, 3, Frame::Inertial>>
metric_and_dt_spatial_metric(const DataType& x, const DataType& y,
                             const DataType& z, const double t,
                             const double amplitude, const int mode,
                             const bool even_parity, const bool ingoing,
                             const std::array<double, 3>& center,
                             const double radius, const double width,
                             const bool include_minkowski_background) {
  auto spatial_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.0);
  auto dt_spatial_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.0);
  if (include_minkowski_background) {
    get<0, 0>(spatial_metric) = 1.0;
    get<1, 1>(spatial_metric) = 1.0;
    get<2, 2>(spatial_metric) = 1.0;
  }

  const DataType centered_x = x - center[0];
  const DataType centered_y = y - center[1];
  const DataType centered_z = z - center[2];
  const DataType radius_from_center =
      sqrt(square(centered_x) + square(centered_y) + square(centered_z));
  const double minimum_radius_from_center = min(radius_from_center);
  if (minimum_radius_from_center <= origin_exclusion_radius) {
    ERROR(
        "TeukolskyWave cannot be evaluated at points with radius <= "
        << origin_exclusion_radius
        << " from its center because the implementation uses spherical "
           "coordinates, which are singular at the origin. Minimum radius was "
        << minimum_radius_from_center << ".");
  }
  const DataType cylindrical_radius =
      sqrt(square(centered_x) + square(centered_y));
  const DataType axis_mask =
      step_function(cylindrical_radius - axis_coordinate_tolerance);
  const DataType centered_x_axis_limit =
      centered_x + (1.0 - axis_mask) * axis_limit_offset;
  const DataType centered_y_axis_limit =
      centered_y + (1.0 - axis_mask) * axis_limit_offset;

  const auto perturbation_and_dt = perturbation_and_dt_perturbation(
      centered_x, centered_y, centered_z, t, amplitude, mode, even_parity,
      ingoing, radius, width);
  // On the axis, evaluate the smooth Cartesian limit by approaching from two
  // orthogonal transverse directions instead of fixing an arbitrary azimuth.
  const auto axis_limit_perturbation_and_dt_x =
      perturbation_and_dt_perturbation(centered_x_axis_limit, centered_y,
                                       centered_z, t, amplitude, mode,
                                       even_parity, ingoing, radius, width);
  const auto axis_limit_perturbation_and_dt_y =
      perturbation_and_dt_perturbation(centered_x, centered_y_axis_limit,
                                       centered_z, t, amplitude, mode,
                                       even_parity, ingoing, radius, width);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      const DataType axis_limit_perturbation =
          0.5 * (axis_limit_perturbation_and_dt_x.first.get(i, j) +
                 axis_limit_perturbation_and_dt_y.first.get(i, j));
      const DataType axis_limit_dt_perturbation =
          0.5 * (axis_limit_perturbation_and_dt_x.second.get(i, j) +
                 axis_limit_perturbation_and_dt_y.second.get(i, j));
      spatial_metric.get(i, j) +=
          axis_mask * perturbation_and_dt.first.get(i, j) +
          (1.0 - axis_mask) * axis_limit_perturbation;
      dt_spatial_metric.get(i, j) =
          axis_mask * perturbation_and_dt.second.get(i, j) +
          (1.0 - axis_mask) * axis_limit_dt_perturbation;
    }
  }

  return {std::move(spatial_metric), std::move(dt_spatial_metric)};
}

}  // namespace

namespace gr::Solutions {

TeukolskyWave::TeukolskyWave(CkMigrateMessage* /*msg*/) {}

TeukolskyWave::TeukolskyWave(double amplitude, const int mode,
                             std::string parity, std::string direction,
                             std::array<double, 3> center, const double radius,
                             const double width,
                             const Options::Context& context)
    : TeukolskyWave(amplitude, mode, std::move(parity), std::move(direction),
                    center, radius, width, true, context) {}

TeukolskyWave::TeukolskyWave(double amplitude, const int mode,
                             std::string parity, std::string direction,
                             std::array<double, 3> center, const double radius,
                             const double width,
                             const bool include_minkowski_background,
                             const Options::Context& context)
    : amplitude_(amplitude),
      mode_(mode),
      parity_(std::move(parity)),
      direction_(std::move(direction)),
      center_(center),
      radius_(radius),
      width_(width),
      include_minkowski_background_(include_minkowski_background) {
  if (mode_ < -2 or mode_ > 2) {
    PARSE_ERROR(context, "Mode must lie between -2 and 2, inclusive.");
  }
  if (parity_ != "even" and parity_ != "odd") {
    PARSE_ERROR(context, "Parity must be either 'even' or 'odd'.");
  }
  if (direction_ != "outgoing" and direction_ != "ingoing") {
    PARSE_ERROR(context, "Direction must be either 'outgoing' or 'ingoing'.");
  }
  if (width_ <= 0.0) {
    PARSE_ERROR(context, "Width must be greater than 0.");
  }
}

void TeukolskyWave::pup(PUP::er& p) {
  p | amplitude_;
  p | mode_;
  p | parity_;
  p | direction_;
  p | center_;
  p | radius_;
  p | width_;
  p | include_minkowski_background_;
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>> {
  return {Scalar<DataType>{
      make_with_value<DataType>(x, include_minkowski_background_ ? 1.0 : 0.0)}};
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> {
  return {Scalar<DataType>{make_with_value<DataType>(x, 0.0)}};
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>
    /*meta*/) const
    -> tuples::TaggedTuple<
        gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>> {
  return {
      make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<
        ::Tags::dt<gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>>
    /*meta*/) const
    -> tuples::TaggedTuple<
        ::Tags::dt<gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>> {
  return {
      make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double t,
    tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>>
    /*meta*/) const
    -> tuples::TaggedTuple<
        gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>> {
  return metric_and_dt_spatial_metric(get<0>(x), get<1>(x), get<2>(x), t,
                                      amplitude_, mode_, parity_ == "even",
                                      direction_ == "ingoing", center_, radius_,
                                      width_, include_minkowski_background_)
      .first;
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double t,
    tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<
        DataType, volume_dim, Frame::Inertial>>> /*meta*/) const
    -> tuples::TaggedTuple<::Tags::dt<
        gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>>> {
  return metric_and_dt_spatial_metric(get<0>(x), get<1>(x), get<2>(x), t,
                                      amplitude_, mode_, parity_ == "even",
                                      direction_ == "ingoing", center_, radius_,
                                      width_, include_minkowski_background_)
      .second;
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double t,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> {
  if (not include_minkowski_background_) {
    ERROR(
        "IncludeMinkowskiBackground must be true to compute "
        "sqrt(det(gamma)).");
  }
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, volume_dim>>(variables(
          x, t, tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim>>{}));
  const auto det_and_inverse = determinant_and_inverse(spatial_metric);
  auto sqrt_det_spatial_metric = make_with_value<Scalar<DataType>>(x, 0.0);
  get(sqrt_det_spatial_metric) = sqrt(get(det_and_inverse.first));
  return sqrt_det_spatial_metric;
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double t,
    tmpl::list<
        gr::Tags::ExtrinsicCurvature<DataType, volume_dim, Frame::Inertial>>
    /*meta*/) const
    -> tuples::TaggedTuple<
        gr::Tags::ExtrinsicCurvature<DataType, volume_dim, Frame::Inertial>> {
  if (not include_minkowski_background_) {
    ERROR(
        "IncludeMinkowskiBackground must be true to compute extrinsic "
        "curvature.");
  }
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  const auto dt_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>>(variables(
      x, t,
      tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>>{}));
  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = i; j < volume_dim; ++j) {
      extrinsic_curvature.get(i, j) = -0.5 * dt_spatial_metric.get(i, j);
    }
  }
  return extrinsic_curvature;
}

template <typename DataType>
auto TeukolskyWave::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double t,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataType, volume_dim,
                                              Frame::Inertial>> /*meta*/) const
    -> tuples::TaggedTuple<
        gr::Tags::InverseSpatialMetric<DataType, volume_dim, Frame::Inertial>> {
  if (not include_minkowski_background_) {
    ERROR(
        "IncludeMinkowskiBackground must be true to compute inverse spatial "
        "metric.");
  }
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, volume_dim>>(variables(
          x, t, tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim>>{}));
  return determinant_and_inverse(spatial_metric).second;
}

bool operator==(const TeukolskyWave& lhs, const TeukolskyWave& rhs) {
  return lhs.amplitude() == rhs.amplitude() and lhs.mode() == rhs.mode() and
         lhs.parity() == rhs.parity() and lhs.direction() == rhs.direction() and
         lhs.center() == rhs.center() and lhs.radius() == rhs.radius() and
         lhs.width() == rhs.width() and
         lhs.include_minkowski_background() ==
             rhs.include_minkowski_background();
}

bool operator!=(const TeukolskyWave& lhs, const TeukolskyWave& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t, tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const;      \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t, tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/) \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::Shift<                                \
      DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>                \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<gr::Tags::Shift<DTYPE(data), TeukolskyWave::volume_dim,       \
                                 Frame::Inertial>> /*meta*/) const;            \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<                     \
      DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>>               \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<::Tags::dt<gr::Tags::Shift<                                   \
          DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>> /*meta*/) \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::SpatialMetric<                        \
      DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>                \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<gr::Tags::SpatialMetric<                                      \
          DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>> /*meta*/)  \
      const;                                                                   \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::SpatialMetric<             \
      DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>>               \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<                           \
          DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>> /*meta*/) \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/) const; \
  template tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<                   \
      DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>                \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<gr::Tags::ExtrinsicCurvature<                                 \
          DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>> /*meta*/)  \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<                 \
      DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>>                \
  gr::Solutions::TeukolskyWave::variables(                                     \
      const tnsr::I<DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>&  \
          x,                                                                   \
      double t,                                                                \
      tmpl::list<gr::Tags::InverseSpatialMetric<                               \
          DTYPE(data), TeukolskyWave::volume_dim, Frame::Inertial>> /*meta*/)  \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace gr::Solutions
