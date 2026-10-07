// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

/*!
 * \brief A camera that emits parallel rays (orthographic projection).
 * Also useful to spawn rays with specific impact parameters.
 */
class ParallelRays : public Camera {
 public:
  static constexpr Options::String help =
      "A camera that emits parallel rays (orthographic projection).";

  struct Extent {
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Coordinate extent of the camera in the two screen directions. "
        "First dimension is horizontal / right, second dimension is vertical / "
        "up.";
  };

  using options = tmpl::replace<typename Camera::options, OpeningAngle, Extent>;

  using Camera::Camera;

  /// \cond
  explicit ParallelRays(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ParallelRays);
  /// \endcond

  std::unique_ptr<RaySource> get_clone() const override;

  tuples::tagged_tuple_from_typelist<tags> operator()(
      size_t ray_index,
      const BackgroundSpacetime& background_spacetime) const override;
};

}  // namespace ray_tracing
