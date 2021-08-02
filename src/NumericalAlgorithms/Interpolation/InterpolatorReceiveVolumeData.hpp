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
struct InterpolatorReceiveVolumeData {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, size_t VolumeDim,
      Requires<tmpl::list_contains_v<DbTags, Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const double time,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&&
          vars) noexcept {
    db::mutate<Tags::VolumeVarsInfo<Metavariables>>(
        make_not_null(&box),
        [&time, &element_id, &mesh,
         &vars](const gsl::not_null<
                typename Tags::VolumeVarsInfo<Metavariables>::type*>
                    container) noexcept {
          if (container->find(time) == container->end()) {
            container->emplace(
                time,
                std::unordered_map<
                    ElementId<VolumeDim>,
                    typename Tags::VolumeVarsInfo<Metavariables>::Info>{});
          }
          container->at(time).emplace(std::make_pair(
              element_id, typename Tags::VolumeVarsInfo<Metavariables>::Info{
                              mesh, std::move(vars), {}}));
        });

    // Try to interpolate data for all InterpolationTargets.
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&box, &cache, &time](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          try_to_interpolate<tag>(make_not_null(&box), make_not_null(&cache),
                                  time);
        });
  }
};

}  // namespace Actions
}  // namespace intrp
