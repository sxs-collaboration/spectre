// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <type_traits>

#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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
struct FunctionsOfTime;
}  // namespace domain::Tags
/// \endcond

namespace intrp {
namespace Events {
/// Does an interpolation onto InterpolationTargetTag by calling Actions on
/// the Interpolator and InterpolationTarget components.
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename InterpolatorSourceVarTags>
class Interpolate;

template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... InterpolatorSourceVarTags>
class Interpolate<VolumeDim, InterpolationTargetTag,
                  tmpl::list<InterpolatorSourceVarTags...>> : public Event {
  /// \cond
  explicit Interpolate(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Interpolate);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Starts interpolation onto the given InterpolationTargetTag.";

  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }

  Interpolate() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags =
      tmpl::list<typename InterpolationTargetTag::temporal_id,
                 domain::Tags::Mesh<VolumeDim>, InterpolatorSourceVarTags...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      const Mesh<VolumeDim>& mesh,
      const typename InterpolatorSourceVarTags::
          type&... interpolator_source_vars,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) const {
    static_assert(
        std::is_same_v<typename Metavariables::interpolator_source_vars,
                       tmpl::list<InterpolatorSourceVarTags...>>);
    interpolate<InterpolationTargetTag>(temporal_id, mesh, cache, array_index,
                                        interpolator_source_vars...);
  }

  using is_ready_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(const double time, Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index,
                const Component* const component) const {
    return domain::functions_of_time_are_ready<domain::Tags::FunctionsOfTime>(
        cache, array_index, component, time);
  }

  bool needs_evolved_variables() const override { return true; }
};

/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... InterpolatorSourceVarTags>
PUP::able::PUP_ID
    Interpolate<VolumeDim, InterpolationTargetTag,
                tmpl::list<InterpolatorSourceVarTags...>>::my_PUP_ID =
        0;  // NOLINT
/// \endcond

}  // namespace Events
}  // namespace intrp
