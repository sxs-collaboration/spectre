// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Cce {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Interpolates and sends points to the `CceWorldtubeTarget`
///
/// This is invoked on DgElementArray for the GH system.
/// This action should appear only in the self-start action list, and
/// is an unfortunate hack needed by the contradictory constraints
/// of a locally-stepped CCE system and the events and dense triggers
/// during the self start procedure.
template <typename CceWorltubeTargetTag>
struct InterpolateDuringSelfStart {
  template <typename DbTags, typename Metavariables, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) noexcept {
    auto interpolate_event = intrp::Events::InterpolateWithoutInterpComponent<
        Metavariables::volume_dim, CceWorltubeTargetTag, Metavariables,
        typename CceWorltubeTargetTag::vars_to_interpolate_to_target>{};
    interpolate_event.run(box, cache, array_index, component);
    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace Cce
