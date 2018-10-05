// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Initializes an Interpolator
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `Tags::NumberOfElements`
///   - `Tags::VolumeVarsInfo<Metavariables,VolumeDim>`
///   - `Tags::InterpolatedVarsHolders<Metavariables,VolumeDim>`
/// - Removes: nothing
/// - Modifies: nothing
template <size_t VolumeDim>
struct InitializeInterpolator {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::list<Tags::NumberOfElements,
                 Tags::VolumeVarsInfo<Metavariables, VolumeDim>,
                 Tags::InterpolatedVarsHolders<Metavariables, VolumeDim>>;
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create<db::get_items<return_tag_list<Metavariables>>>(
            0_st,
            db::item_type<Tags::VolumeVarsInfo<Metavariables, VolumeDim>>{},
            db::item_type<
                Tags::InterpolatedVarsHolders<Metavariables, VolumeDim>>{}));
  }
};

}  // namespace Actions
}  // namespace intrp
