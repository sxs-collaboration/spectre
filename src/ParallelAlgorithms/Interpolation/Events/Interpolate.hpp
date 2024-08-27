// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <type_traits>

#include "Options/String.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/ArrayCollection/PerformAlgorithmOnElement.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Events/GetComputeItemsOnSource.hpp"
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
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Events::Tags {
template <size_t Dim>
struct ObserverMesh;
}  // namespace Events::Tags
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
 public:
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

  using compute_tags_for_observation_box =
      detail::get_compute_items_on_source_or_default_t<InterpolationTargetTag,
                                                       tmpl::list<>>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<typename InterpolationTargetTag::temporal_id,
                                   ::Events::Tags::ObserverMesh<VolumeDim>,
                                   InterpolatorSourceVarTags...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      const Mesh<VolumeDim>& mesh,
      const typename InterpolatorSourceVarTags::
          type&... interpolator_source_vars,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/,
      const ObservationValue& /*observation_value*/) const {
    static_assert(
        std::is_same_v<typename Metavariables::interpolator_source_vars,
                       tmpl::list<InterpolatorSourceVarTags...>>);
    if constexpr (Parallel::is_dg_element_collection_v<ParallelComponent>) {
      const auto core_id = static_cast<int>(
          Parallel::local_synchronous_action<
              Parallel::Actions::GetItemFromDistributedOject<
                  typename ParallelComponent::element_collection_tag>>(
              Parallel::get_parallel_component<ParallelComponent>(cache))
              ->at(array_index)
              .get_core());
      interpolate<InterpolationTargetTag>(temporal_id, mesh, cache, array_index,
                                          core_id, interpolator_source_vars...);
    } else {
      interpolate<InterpolationTargetTag>(temporal_id, mesh, cache, array_index,
                                          std::nullopt,
                                          interpolator_source_vars...);
    }
  }

  using is_ready_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(const double time, Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index,
                const Component* const component) const {
    if constexpr (Parallel::is_dg_element_collection_v<Component>) {
      const auto element_location = static_cast<int>(
          Parallel::local_synchronous_action<
              Parallel::Actions::GetItemFromDistributedOject<
                  Parallel::Tags::ElementLocations<VolumeDim>>>(
              Parallel::get_parallel_component<Component>(cache))
              ->at(array_index));
      return domain::functions_of_time_are_ready_threaded_action_callback<
          domain::Tags::FunctionsOfTime,
          Parallel::Actions::PerformAlgorithmOnElement<false>>(
          cache, element_location, component, time, std::nullopt, array_index);
    } else {
      return domain::functions_of_time_are_ready_algorithm_callback<
          domain::Tags::FunctionsOfTime>(cache, array_index, component, time);
    }
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
