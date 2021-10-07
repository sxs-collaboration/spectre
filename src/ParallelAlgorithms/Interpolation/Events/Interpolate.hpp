// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <size_t VolumeDim>
class ElementId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain::Tags {
template <size_t VolumeDim>
struct Mesh;
}  // namespace domain::Tags
/// \endcond

namespace intrp {
namespace Events {
/// Does an interpolation onto InterpolationTargetTag by calling Actions on
/// the Interpolator and InterpolationTarget components.
template <size_t VolumeDim, typename InterpolationTargetTag, typename Tensors>
class Interpolate;

template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... Tensors>
class Interpolate<VolumeDim, InterpolationTargetTag, tmpl::list<Tensors...>>
    : public Event {
  /// \cond
  explicit Interpolate(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Interpolate);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Starts interpolation onto the given InterpolationTargetTag.";

  static std::string name() { return Options::name<InterpolationTargetTag>(); }

  Interpolate() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<typename InterpolationTargetTag::temporal_id,
                                   domain::Tags::Mesh<VolumeDim>, Tensors...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      const Mesh<VolumeDim>& mesh, const typename Tensors::type&... tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) const {
    interpolate<InterpolationTargetTag, tmpl::list<Tensors...>>(
        temporal_id, mesh, cache, array_index, tensors...);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }
};

/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... Tensors>
PUP::able::PUP_ID Interpolate<VolumeDim, InterpolationTargetTag,
                              tmpl::list<Tensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Events
}  // namespace intrp
