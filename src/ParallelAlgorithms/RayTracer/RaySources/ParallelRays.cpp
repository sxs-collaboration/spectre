// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/RaySources/ParallelRays.hpp"

#include <memory>

#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace ray_tracing {

ParallelRays::ParallelRays(CkMigrateMessage* msg) : Camera(msg) {}

std::unique_ptr<RaySource> ParallelRays::get_clone() const {
  return std::make_unique<ParallelRays>(*this);
}

tuples::tagged_tuple_from_typelist<typename RaySource::tags>
ParallelRays::operator()(
    size_t ray_index,
    const BackgroundSpacetime& /*background_spacetime*/) const {
  const auto [x_frac, y_frac] = screen_coordinates(ray_index);
  // Reusing opening_angle_ to store the extent of the screen for parallel rays.
  const auto& extent = opening_angle_;
  auto position = tenex::evaluate<ti::I>(
      position_(ti::I) + (2.0 * x_frac - 1.0) * extent[0] * right_(ti::I) -
      (2.0 * y_frac - 1.0) * extent[1] * up_(ti::I));
  // The spacetime metric at the `position_` is used for all rays, so they are
  // only approximately parallel in curved spacetime.
  const auto momentum = raise_or_lower_index(direction_, spacetime_metric_);
  return tuples::tagged_tuple_from_typelist<typename RaySource::tags>{
      time_, std::move(position),
      // Negate momentum so it's future-directed. Then trace backwards in time.
      tenex::evaluate<ti::i>(-momentum(ti::i)), -integration_time_};
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID ParallelRays::my_PUP_ID = 0;

}  // namespace ray_tracing
