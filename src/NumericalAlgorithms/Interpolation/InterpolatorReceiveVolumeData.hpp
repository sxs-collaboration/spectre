// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
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
///   - `Tags::VolumeVarsInfo<Metavariables>`
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
template <typename TemporalId>
struct InterpolatorReceiveVolumeData {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, size_t VolumeDim,
      Requires<tmpl::list_contains_v<DbTags, Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const typename TemporalId::type& temporal_id,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&&
          vars) noexcept {
    db::mutate<Tags::VolumeVarsInfo<Metavariables, TemporalId>>(
        make_not_null(&box),
        [&temporal_id, &element_id, &mesh,
         &vars](const gsl::not_null<
                typename Tags::VolumeVarsInfo<Metavariables, TemporalId>::type*>
                    container) noexcept {
          if (container->find(temporal_id) == container->end()) {
            container->emplace(
                temporal_id,
                std::unordered_map<ElementId<VolumeDim>,
                                   typename Tags::VolumeVarsInfo<
                                       Metavariables, TemporalId>::Info>{});
          }
          container->at(temporal_id)
              .emplace(std::make_pair(
                  element_id, typename Tags::VolumeVarsInfo<Metavariables,
                                                            TemporalId>::Info{
                                  mesh, std::move(vars), {}}));
        });

    // Try to interpolate data for all InterpolationTargets.
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&box, &cache, &temporal_id](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          if constexpr (std::is_same_v<typename tag::temporal_id, TemporalId>) {
            try_to_interpolate<tag>(make_not_null(&box), make_not_null(&cache),
                                    temporal_id);
          }
        });
  }
};

}  // namespace Actions
}  // namespace intrp
