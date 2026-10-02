// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ray_tracing {

/*!
 * \brief Abstract base class for ray sources (initial conditions) in the ray
 * tracer.
 *
 * Derived classes have to provide the number of rays and their initial
 * conditions.
 *
 * Ray sources also have a concept of "frames" in time. This covers the obvious
 * case of a camera that takes pictures at discrete times, emitting rays at
 * those times, but also more general cases where the ray source emits rays in a
 * time-dependent way divided into chunks of time (e.g. to sample a time range
 * with rays uniformly in time).
 */
class RaySource : public PUP::able {
 protected:
  static constexpr size_t Dim = 3;
  using DataType = double;
  using Frame = ::Frame::Inertial;

  RaySource() = default;

 public:
  ~RaySource() override = default;

  /// \cond
  explicit RaySource(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(RaySource);
  /// \endcond

  /// Copies the ray source. Must call `initialize` on the clone
  /// before using it.
  virtual auto get_clone() const -> std::unique_ptr<RaySource> = 0;

  /// Number of frames (time chunks).
  virtual size_t num_frames() const = 0;

  /// Number of rays emitted in this frame.
  virtual size_t num_rays(size_t frame) const = 0;

  /// Time bounds that rays emitted in this frame can cover.
  virtual std::array<double, 2> time_bounds(size_t frame) const = 0;

  /// Initialize the ray source for the current frame with the background
  /// spacetime. Here, the ray source can set up its geometry, which may depend
  /// on the background spacetime.
  virtual void initialize(size_t frame,
                          const BackgroundSpacetime& background_spacetime) = 0;

  using tags =
      tmpl::list<::Tags::Time, Tags::Position<DataType, Dim, Frame>,
                 Tags::Momentum<DataType, Dim, Frame>, Tags::IntegrationTime>;

  /*!
   * \brief Returns the initial conditions for the ray with the given index
   * in the current frame.
   */
  virtual tuples::tagged_tuple_from_typelist<tags> operator()(
      size_t ray_index,
      const BackgroundSpacetime& background_spacetime) const = 0;
};

}  // namespace ray_tracing
