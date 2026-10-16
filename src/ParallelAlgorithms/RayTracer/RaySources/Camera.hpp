// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

/*!
 * \brief A camera that emits rays ("takes pictures") at periodic frames in
 * time.
 *
 * The camera can step through frames in time, emitting rays at each frame.
 *
 * This implementation uses a pinhole camera model with perspective projection.
 * At each frame, it initializes a tetrad at its position using the Gram-Schmidt
 * procedure and then emits rays through a rectangular screen defined by the
 * opening angles and resolution (see Sec. II.B of \cite Bohn:2014xxa).
 *
 * Subclasses can implement different projections.
 */
class Camera : public RaySource {
 public:
  static constexpr Options::String help = "A camera that emits rays.";

  struct Position {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "Position of the camera.";
  };

  struct Focus {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "Focus point that the camera is looking at.";
  };

  struct Up {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "Up direction of the camera.";
  };

  struct OpeningAngle {
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Opening angles in the two screen directions in degrees. "
        "First dimension is horizontal / right and second dimension is "
        "vertical / up.";
  };

  struct Resolution {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help =
        "Number of rays in the two screen directions. "
        "First dimension is horizontal / right and second dimension is "
        "vertical / up.";
  };

  struct CenterRays {
    using type = bool;
    static constexpr Options::String help =
        "If true, rays are emitted in the center of each pixel. If false, rays "
        "are distributed from one edge of the screen to the other.";
  };

  struct StartTime {
    using type = double;
    static constexpr Options::String help = "Time of the first frame.";
  };

  struct Interval {
    using type = double;
    static constexpr Options::String help =
        "Time interval between frames. Can be negative to traverse frames "
        "backwards in time.";
  };

  struct NumFrames {
    using type = size_t;
    static constexpr Options::String help =
        "Number of frames. Frames are rendered at times StartTime + n * "
        "Interval for n in [0, NumFrames).";
  };

  struct IntegrationTime {
    using type = double;
    static constexpr Options::String help =
        "Maximum time to integrate the geodesics";
  };

  struct OnlyUpperHalf {
    using type = bool;
    static constexpr Options::String help =
        "Due to symmetry, only emit rays in the upper half of the camera";
  };

  using options = tmpl::list<Position, Focus, Up, OpeningAngle, Resolution,
                             CenterRays, StartTime, Interval, NumFrames,
                             IntegrationTime, OnlyUpperHalf>;

  Camera() = default;
  Camera(const Camera& /*rhs*/) = default;
  Camera& operator=(const Camera& /*rhs*/) = default;
  Camera(Camera&& /*rhs*/) = default;
  Camera& operator=(Camera&& /*rhs*/) = default;
  ~Camera() override = default;

  Camera(std::array<double, 3> position, std::array<double, 3> focus,
         std::array<double, 3> up, std::array<double, 2> opening_angle,
         std::array<size_t, 2> resolution, bool center_rays, double start_time,
         double interval, size_t num_frames, double integration_time,
         bool only_upper_half = false);

  const auto& position() const { return position_; }
  const auto& four_velocity() const { return four_velocity_; }
  const auto& direction() const { return direction_; }
  const auto& up() const { return up_; }
  const auto& right() const { return right_; }
  const auto& opening_angle() const { return opening_angle_; }
  const auto& resolution() const { return resolution_; }
  bool center_rays() const { return center_rays_; }
  double start_time() const { return start_time_; }
  double interval() const { return interval_; }
  double integration_time() const { return integration_time_; }
  bool only_upper_half() const { return only_upper_half_; }
  double time() const { return time_; }

  /// \cond
  explicit Camera(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Camera);
  /// \endcond

  std::unique_ptr<RaySource> get_clone() const override;

  size_t num_frames() const override { return num_frames_; }

  size_t num_rays(size_t frame) const override;

  std::array<double, 2> time_bounds(size_t frame) const override;

  void initialize(size_t frame,
                  const BackgroundSpacetime& background_spacetime) override;

  /*!
   * \brief Screen coordinates $\{\xi,\eta\} \in \[0,1\]^2$ of the ray. The
   * origin (0,0) is at the top-left of the screen, as is standard in computer
   * graphics.
   */
  std::array<double, 2> screen_coordinates(size_t ray_index) const;

  tuples::tagged_tuple_from_typelist<tags> operator()(
      size_t ray_index,
      const BackgroundSpacetime& background_spacetime) const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  friend bool operator==(const Camera& lhs, const Camera& rhs);

 protected:
  tnsr::I<double, 3> position_{};
  tnsr::A<double, 3> four_velocity_{};
  tnsr::A<double, 3> direction_{};
  tnsr::A<double, 3> up_{};
  tnsr::A<double, 3> right_{};
  std::array<double, 2> opening_angle_{};
  std::array<size_t, 2> resolution_{};
  bool center_rays_{false};
  double start_time_{};
  double interval_{};
  size_t num_frames_{};
  double integration_time_{};
  bool only_upper_half_{false};
  // State
  double time_{};
  tnsr::aa<double, 3> spacetime_metric_{};
};

bool operator!=(const Camera& lhs, const Camera& rhs);

}  // namespace ray_tracing
