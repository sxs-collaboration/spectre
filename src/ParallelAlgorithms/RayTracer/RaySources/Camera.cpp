// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"

#include <array>
#include <memory>
#include <pup.h>
#include <pup_stl.h>

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
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ray_tracing {

Camera::Camera(const std::array<double, 3> position,
               const std::array<double, 3> focus,
               const std::array<double, 3> up,
               const std::array<double, 2> opening_angle,
               const std::array<size_t, 2> resolution, const bool center_rays,
               const double start_time, const double interval,
               const size_t num_frames, const double integration_time,
               const bool only_upper_half)
    : position_(position),
      four_velocity_{{1.0, 0.0, 0.0, 0.0}},
      direction_{{0., focus[0] - position[0], focus[1] - position[1],
                  focus[2] - position[2]}},
      up_{{0., up[0], up[1], up[2]}},
      right_{{0., direction_[2] * up_[3] - direction_[3] * up_[2],
              direction_[3] * up_[1] - direction_[1] * up_[3],
              direction_[1] * up_[2] - direction_[2] * up_[1]}},
      opening_angle_(opening_angle),
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
  const auto inv_spacetime_metric = gr::inverse_spacetime_metric(
      get<gr::Tags::Lapse<double>>(background_vars),
      get<gr::Tags::Shift<double, 3, ::Frame::Inertial>>(background_vars),
      get<gr::Tags::InverseSpatialMetric<double, 3, ::Frame::Inertial>>(
          background_vars));
  Scalar<double> det_spacetime_metric{};
  determinant_and_inverse(make_not_null(&det_spacetime_metric),
                          make_not_null(&spacetime_metric_),
                          inv_spacetime_metric);
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

tuples::tagged_tuple_from_typelist<typename Camera::tags> Camera::operator()(
    size_t ray_index,
    const BackgroundSpacetime& /*background_spacetime*/) const {
  const auto [x_frac, y_frac] = screen_coordinates(ray_index);
  const double x_angle =
      (2.0 * x_frac - 1.0) * tan(opening_angle_[0] * M_PI / 360.0);
  const double y_angle =
      (2.0 * y_frac - 1.0) * tan(opening_angle_[1] * M_PI / 360.0);
  const double norm = sqrt(1.0 + square(x_angle) + square(y_angle));
  const tnsr::a<double, 3> momentum = raise_or_lower_index(
      tenex::evaluate<ti::A>(
          four_velocity_(ti::A) -
          (direction_(ti::A) + x_angle * right_(ti::A) - y_angle * up_(ti::A)) /
              norm),
      spacetime_metric_);
  return tuples::tagged_tuple_from_typelist<typename Camera::tags>{
      time_, position_, tenex::evaluate<ti::i>(momentum(ti::i)),
      -integration_time_};
}

void Camera::pup(PUP::er& p) {
  RaySource::pup(p);
  p | position_;
  p | four_velocity_;
  p | direction_;
  p | up_;
  p | right_;
  p | opening_angle_;
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
}

bool operator==(const Camera& lhs, const Camera& rhs) {
  return lhs.position_ == rhs.position_ and
         lhs.four_velocity_ == rhs.four_velocity_ and
         lhs.direction_ == rhs.direction_ and lhs.up_ == rhs.up_ and
         lhs.right_ == rhs.right_ and
         lhs.opening_angle_ == rhs.opening_angle_ and
         lhs.resolution_ == rhs.resolution_ and
         lhs.center_rays_ == rhs.center_rays_ and
         lhs.start_time_ == rhs.start_time_ and
         lhs.interval_ == rhs.interval_ and
         lhs.num_frames_ == rhs.num_frames_ and
         lhs.integration_time_ == rhs.integration_time_ and
         lhs.only_upper_half_ == rhs.only_upper_half_ and
         lhs.time_ == rhs.time_ and
         lhs.spacetime_metric_ == rhs.spacetime_metric_;
}

bool operator!=(const Camera& lhs, const Camera& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID Camera::my_PUP_ID = 0;

}  // namespace ray_tracing
