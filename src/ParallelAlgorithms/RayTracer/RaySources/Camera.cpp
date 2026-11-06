// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <variant>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/GramSchmidtOrthonormalize.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace ray_tracing {

PerspectiveProjection::PerspectiveProjection(
    const std::array<double, 2> opening_angle_in)
    : opening_angle(opening_angle_in) {}

void PerspectiveProjection::pup(PUP::er& p) { p | opening_angle; }

OrthographicProjection::OrthographicProjection(
    const std::array<double, 2> extent_in)
    : extent(extent_in) {}

void OrthographicProjection::pup(PUP::er& p) { p | extent; }

Camera::Camera(const std::array<double, 3> position,
               const std::array<double, 3> focus,
               const std::array<double, 3> up, Projection::type projection,
               const std::array<size_t, 2> resolution, const bool center_rays,
               const double start_time, const double interval,
               const size_t num_frames, const double integration_time,
               const bool only_upper_half)
    : position_(position),
      coordinate_direction_{{0., focus[0] - position[0], focus[1] - position[1],
                             focus[2] - position[2]}},
      coordinate_up_{{0., up[0], up[1], up[2]}},
      four_velocity_{{1.0, 0.0, 0.0, 0.0}},
      direction_(coordinate_direction_),
      up_(coordinate_up_),
      right_{{0., direction_[2] * up_[3] - direction_[3] * up_[2],
              direction_[3] * up_[1] - direction_[1] * up_[3],
              direction_[1] * up_[2] - direction_[2] * up_[1]}},
      projection_(std::move(projection)),
      resolution_(resolution),
      center_rays_(center_rays),
      start_time_(start_time),
      interval_(interval),
      num_frames_(num_frames),
      integration_time_(integration_time),
      only_upper_half_(only_upper_half) {}

Camera::Camera(CkMigrateMessage* msg) : RaySource(msg) {}

std::unique_ptr<RaySource> Camera::get_clone() const {
  return std::make_unique<Camera>(*this);
}

size_t Camera::num_rays(const size_t /*frame*/) const {
  const size_t y_resolution =
      only_upper_half_ ? (resolution_[1] + 1_st) / 2_st : resolution_[1];
  return resolution_[0] * y_resolution;
}

std::array<double, 2> Camera::time_bounds(const size_t frame) const {
  const double time = start_time_ + static_cast<double>(frame) * interval_;
  return {{time - integration_time_, time}};
}

void Camera::initialize(const size_t frame,
                        const BackgroundSpacetime& background_spacetime) {
  time_ = start_time_ + static_cast<double>(frame) * interval_;
  const auto background_vars = background_spacetime.variables(position_, time_);
  lapse_ = get(get<gr::Tags::Lapse<double>>(background_vars));
  const auto inv_spacetime_metric = gr::inverse_spacetime_metric(
      get<gr::Tags::Lapse<double>>(background_vars),
      get<gr::Tags::Shift<double, 3, ::Frame::Inertial>>(background_vars),
      get<gr::Tags::InverseSpatialMetric<double, 3, ::Frame::Inertial>>(
          background_vars));
  Scalar<double> det_spacetime_metric{};
  determinant_and_inverse(make_not_null(&det_spacetime_metric),
                          make_not_null(&spacetime_metric_),
                          inv_spacetime_metric);
  // Construct the tetrad from the coordinate directions, so that it only
  // depends on the frame and not on previous calls to this function
  four_velocity_ = tnsr::A<double, 3>{{1.0, 0.0, 0.0, 0.0}};
  direction_ = coordinate_direction_;
  up_ = coordinate_up_;
  gram_schmidt_orthonormalize(
      std::array{make_not_null(&four_velocity_), make_not_null(&direction_),
                 make_not_null(&up_)},
      spacetime_metric_);
  cross_product(make_not_null(&right_), four_velocity_, direction_, up_,
                inv_spacetime_metric, det_spacetime_metric);
}

std::array<double, 2> Camera::screen_coordinates(size_t ray_index) const {
  const size_t x_index = ray_index % resolution_[0];
  const size_t y_index = ray_index / resolution_[0];
  if (center_rays_) {
    // Center rays in pixels
    return {(static_cast<double>(x_index) + 0.5) /
                static_cast<double>(resolution_[0]),
            (static_cast<double>(y_index) + 0.5) /
                static_cast<double>(resolution_[1])};
  } else {
    // Distribute rays from edge to edge (center only if single ray)
    return {resolution_[0] == 1 ? 0.5
                                : static_cast<double>(x_index) /
                                      static_cast<double>(resolution_[0] - 1),
            resolution_[1] == 1 ? 0.5
                                : static_cast<double>(y_index) /
                                      static_cast<double>(resolution_[1] - 1)};
  }
}

tnsr::i<double, 3> Camera::momentum(
    const tnsr::A<double, 3>& view_direction) const {
  // The ray arrives at the camera from the view direction, so in the camera
  // frame its four-momentum is p^a = u^a - n^a (null since u and n are
  // orthonormal), normalized such that the camera measures unit energy.
  const auto p_upper =
      tenex::evaluate<ti::A>(four_velocity_(ti::A) - view_direction(ti::A));
  const auto p_lower = raise_or_lower_index(p_upper, spacetime_metric_);
  // Normalize the spatial part as Pi_i = p_i / (alpha p^0)
  const double alpha_p0 = lapse_ * get<0>(p_upper);
  return tnsr::i<double, 3>{{get<1>(p_lower) / alpha_p0,
                             get<2>(p_lower) / alpha_p0,
                             get<3>(p_lower) / alpha_p0}};
}

tuples::tagged_tuple_from_typelist<typename Camera::tags> Camera::operator()(
    size_t ray_index,
    const BackgroundSpacetime& /*background_spacetime*/) const {
  const auto [x_frac, y_frac] = screen_coordinates(ray_index);
  // Screen coordinates in [-1, 1], with the y-axis pointing down
  const double x_screen = 2.0 * x_frac - 1.0;
  const double y_screen = 2.0 * y_frac - 1.0;
  return std::visit(
      Overloader{
          [&](const PerspectiveProjection& perspective) {
            const double x_angle =
                x_screen * tan(perspective.opening_angle[0] * M_PI / 360.0);
            const double y_angle =
                y_screen * tan(perspective.opening_angle[1] * M_PI / 360.0);
            const double norm = sqrt(1.0 + square(x_angle) + square(y_angle));
            const auto view_direction = tenex::evaluate<ti::A>(
                (direction_(ti::A) + x_angle * right_(ti::A) -
                 y_angle * up_(ti::A)) /
                norm);
            return tuples::tagged_tuple_from_typelist<typename Camera::tags>{
                time_, position_, momentum(view_direction), -integration_time_};
          },
          [&](const OrthographicProjection& orthographic) {
            const double x_offset = 0.5 * x_screen * orthographic.extent[0];
            const double y_offset = 0.5 * y_screen * orthographic.extent[1];
            tnsr::I<double, 3> position = position_;
            for (size_t i = 0; i < 3; ++i) {
              position.get(i) +=
                  x_offset * right_.get(i + 1) - y_offset * up_.get(i + 1);
            }
            return tuples::tagged_tuple_from_typelist<typename Camera::tags>{
                time_, std::move(position), momentum(direction_),
                -integration_time_};
          }},
      projection_);
}

void Camera::pup(PUP::er& p) {
  RaySource::pup(p);
  p | position_;
  p | coordinate_direction_;
  p | coordinate_up_;
  p | four_velocity_;
  p | direction_;
  p | up_;
  p | right_;
  p | projection_;
  p | resolution_;
  p | center_rays_;
  p | start_time_;
  p | interval_;
  p | num_frames_;
  p | integration_time_;
  p | only_upper_half_;
  // Also serialize cached quantities that are computed during initialization
  p | time_;
  p | spacetime_metric_;
  p | lapse_;
}

bool operator==(const Camera& lhs, const Camera& rhs) {
  return lhs.position_ == rhs.position_ and
         lhs.coordinate_direction_ == rhs.coordinate_direction_ and
         lhs.coordinate_up_ == rhs.coordinate_up_ and
         lhs.four_velocity_ == rhs.four_velocity_ and
         lhs.direction_ == rhs.direction_ and lhs.up_ == rhs.up_ and
         lhs.right_ == rhs.right_ and lhs.projection_ == rhs.projection_ and
         lhs.resolution_ == rhs.resolution_ and
         lhs.center_rays_ == rhs.center_rays_ and
         lhs.start_time_ == rhs.start_time_ and
         lhs.interval_ == rhs.interval_ and
         lhs.num_frames_ == rhs.num_frames_ and
         lhs.integration_time_ == rhs.integration_time_ and
         lhs.only_upper_half_ == rhs.only_upper_half_ and
         lhs.time_ == rhs.time_ and
         lhs.spacetime_metric_ == rhs.spacetime_metric_ and
         lhs.lapse_ == rhs.lapse_;
}

bool operator!=(const Camera& lhs, const Camera& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID Camera::my_PUP_ID = 0;

}  // namespace ray_tracing
