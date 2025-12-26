// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <variant>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

/// A pinhole camera: rays fan out from the camera position.
struct PerspectiveProjection {
  /// Option group that names this projection in the input file
  struct Perspective {
    static constexpr Options::String help =
        "Pinhole camera. Rays fan out from the camera position.";
  };
  struct OpeningAngle {
    using group = Perspective;
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Opening angles in the two screen directions in degrees. "
        "First dimension is horizontal / right and second dimension is "
        "vertical / up.";
  };
  using options = tmpl::list<OpeningAngle>;
  static constexpr Options::String help = Perspective::help;

  PerspectiveProjection() = default;
  explicit PerspectiveProjection(std::array<double, 2> opening_angle_in);

  std::array<double, 2> opening_angle{};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  bool operator==(const PerspectiveProjection& rhs) const = default;
};

/// An orthographic camera: rays are emitted in parallel from the screen.
struct OrthographicProjection {
  /// Option group that names this projection in the input file
  struct Orthographic {
    static constexpr Options::String help =
        "Orthographic camera. Rays are emitted parallel to each other from a "
        "screen centered on the camera position. Useful to spawn rays with "
        "specific impact parameters.";
  };
  struct Extent {
    using group = Orthographic;
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Coordinate width of the screen in the two screen directions. "
        "First dimension is horizontal / right and second dimension is "
        "vertical / up.";
  };
  using options = tmpl::list<Extent>;
  static constexpr Options::String help = Orthographic::help;

  OrthographicProjection() = default;
  explicit OrthographicProjection(std::array<double, 2> extent_in);

  std::array<double, 2> extent{};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  bool operator==(const OrthographicProjection& rhs) const = default;
};

/*!
 * \brief A camera that emits rays ("takes pictures") at periodic frames in
 * time.
 *
 * The camera can step through frames in time, emitting rays at each frame.
 * At each frame, it initializes a tetrad at its position using the Gram-Schmidt
 * procedure and then emits rays through a rectangular screen with the given
 * resolution (see Sec. II.B of \cite Bohn:2014xxa). The tetrad is built from
 * the coordinate time direction, so the camera is static with respect to the
 * coordinates.
 *
 * The `Projection` determines how rays are emitted from the screen. Choose it
 * by name in the input file, e.g.:
 *
 * \code{.yaml}
 * Projection:
 *   Perspective:
 *     OpeningAngle: [60, 40]
 * \endcode
 *
 * - `Perspective`: A pinhole camera. All rays are emitted from the
 *   camera position and fan out over the opening angles.
 * - `Orthographic`: All rays are emitted in parallel along the viewing
 *   direction from a screen of the given extent centered on the camera
 *   position. The screen is spanned by the spatial part of the tetrad, and the
 *   rays use the tetrad at the camera position. Therefore, rays are only
 *   approximately parallel in curved spacetime. This projection is useful to
 *   spawn rays with specific impact parameters.
 *
 * The momentum of each ray is normalized as $\Pi_i = p_i / (\alpha p^0)$
 * (see `Tags::Momentum`), as required by `gr::geodesic_equation`.
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

  struct Projection {
    using type = std::variant<PerspectiveProjection, OrthographicProjection>;
    static constexpr Options::String help =
        "How rays are emitted from the screen. Specify either 'Perspective' "
        "for a pinhole camera or 'Orthographic' for parallel rays.";
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

  using options = tmpl::list<Position, Focus, Up, Projection, Resolution,
                             CenterRays, StartTime, Interval, NumFrames,
                             IntegrationTime, OnlyUpperHalf>;

  Camera() = default;
  Camera(const Camera& /*rhs*/) = default;
  Camera& operator=(const Camera& /*rhs*/) = default;
  Camera(Camera&& /*rhs*/) = default;
  Camera& operator=(Camera&& /*rhs*/) = default;
  ~Camera() override = default;

  Camera(std::array<double, 3> position, std::array<double, 3> focus,
         std::array<double, 3> up, Projection::type projection,
         std::array<size_t, 2> resolution, bool center_rays, double start_time,
         double interval, size_t num_frames, double integration_time,
         bool only_upper_half = false);

  const auto& position() const { return position_; }
  const auto& four_velocity() const { return four_velocity_; }
  const auto& direction() const { return direction_; }
  const auto& up() const { return up_; }
  const auto& right() const { return right_; }
  const auto& projection() const { return projection_; }
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

 private:
  // Momentum variable Pi_i for a ray that arrives at the camera from the
  // spatial direction `view_direction` (a unit vector orthogonal to the
  // four-velocity of the camera)
  tnsr::i<double, 3> momentum(const tnsr::A<double, 3>& view_direction) const;

  tnsr::I<double, 3> position_{};
  // Viewing and up directions in coordinate space, from which the tetrad is
  // constructed in each frame
  tnsr::A<double, 3> coordinate_direction_{};
  tnsr::A<double, 3> coordinate_up_{};
  // Tetrad at the camera position, constructed in `initialize`
  tnsr::A<double, 3> four_velocity_{};
  tnsr::A<double, 3> direction_{};
  tnsr::A<double, 3> up_{};
  tnsr::A<double, 3> right_{};
  Projection::type projection_{};
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
  double lapse_{};
};

bool operator!=(const Camera& lhs, const Camera& rhs);

}  // namespace ray_tracing
