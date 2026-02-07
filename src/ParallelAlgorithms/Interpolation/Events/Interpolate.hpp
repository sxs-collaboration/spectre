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
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Events/GetComputeItemsOnSource.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
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
namespace Events::Tags {
template <size_t Dim>
struct ObserverMesh;
}  // namespace Events::Tags
namespace ah::Tags {
struct BlocksForInterpolation;
}  // namespace ah::Tags
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

  /// This constructor is not available through options
  explicit Interpolate(std::optional<std::string> dependency)
      : dependency_(std::move(dependency)) {}

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

    const auto& blocks_to_interpolate =
        Parallel::get<ah::Tags::BlocksForInterpolation>(cache);
    ASSERT(blocks_to_interpolate.contains(name()),
           "Blocks to interpolate doesn't contain target " << name());
    const auto& blocks_to_interpolate_for_this_target =
        blocks_to_interpolate.at(name());
    const auto& blocks =
        Parallel::get<domain::Tags::Domain<VolumeDim>>(cache).blocks();
    const auto& block_name = blocks[array_index.block_id()].name();

    // Only send data if this target needs this blocks data
    if (not blocks_to_interpolate_for_this_target.contains(block_name)) {
      return;
    }

    if constexpr (Parallel::is_dg_element_collection_v<ParallelComponent>) {
      const auto core_id = static_cast<int>(
          Parallel::local_synchronous_action<
              Parallel::Actions::GetItemFromDistributedOject<
                  typename ParallelComponent::element_collection_tag>>(
              Parallel::get_parallel_component<ParallelComponent>(cache))
              ->at(array_index)
              .get_core());
      interpolate<InterpolationTargetTag>(temporal_id, mesh, cache, array_index,
                                          core_id, dependency_,
                                          interpolator_source_vars...);
    } else {
      interpolate<InterpolationTargetTag>(temporal_id, mesh, cache, array_index,
                                          std::nullopt, dependency_,
                                          interpolator_source_vars...);
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*component*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  void pup(PUP::er& p) override {
    Event::pup(p);
    p | dependency_;
  }

 private:
  std::optional<std::string> dependency_;
};

#if defined(SPECTRE_USE_CHARM)
/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... InterpolatorSourceVarTags>
PUP::able::PUP_ID
    Interpolate<VolumeDim, InterpolationTargetTag,
                tmpl::list<InterpolatorSourceVarTags...>>::my_PUP_ID =
        0;  // NOLINT
/// \endcond
#endif  // SPECTRE_USE_CHARM

}  // namespace Events
}  // namespace intrp
