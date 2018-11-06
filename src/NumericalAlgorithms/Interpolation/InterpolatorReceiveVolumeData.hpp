// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Adds volume data from an `Element`.
///
/// Attempts to interpolate if it already has received target points from
/// any InterpolationTargets.
///
/// Uses:
/// - DataBox:
///   - `Tags::NumberOfElements`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::VolumeVarsInfo<Metavariables,VolumeDim>`
///   - `Tags::InterpolatedVarsHolders<Metavariables,VolumeDim>`
struct InterpolatorReceiveVolumeData {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      size_t VolumeDim,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumberOfElements>> =
          nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const typename Metavariables::temporal_id& temporal_id,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&&
          vars) noexcept {
    db::mutate<Tags::VolumeVarsInfo<Metavariables, VolumeDim>>(
        make_not_null(&box),
        [&temporal_id, &element_id, &mesh,
         &vars ](const gsl::not_null<
                 db::item_type<Tags::VolumeVarsInfo<Metavariables, VolumeDim>>*>
                     container) noexcept {
          if (container->find(temporal_id) == container->end()) {
            container->emplace(
                temporal_id,
                std::unordered_map<ElementId<VolumeDim>,
                                   typename Tags::VolumeVarsInfo<
                                       Metavariables, VolumeDim>::Info>{});
          }
          container->at(temporal_id)
              .emplace(std::make_pair(
                  element_id,
                  typename Tags::VolumeVarsInfo<Metavariables, VolumeDim>::Info{
                      mesh, std::move(vars)}));
        });

    // Try to interpolate data for all InterpolationTargets.
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&box, &cache, &temporal_id](auto x) noexcept {
          using tag = typename decltype(x)::type;
          try_to_interpolate<tag, VolumeDim>(
              make_not_null(&box), make_not_null(&cache), temporal_id);
        });
  }
};

}  // namespace Actions
}  // namespace intrp
