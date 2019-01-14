// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace domain {
class BlockId;
}  // namespace domain
template <typename IdType, typename DataType>
class IdPair;
namespace intrp {
namespace Tags {
struct NumberOfElements;
template <typename Metavariables>
struct InterpolatedVarsHolders;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Receives target points from an InterpolationTarget.
///
/// After receiving the points, interpolates volume data onto them
/// if it already has all the volume data.
///
/// Uses:
/// - Databox:
///   - `Tags::NumberOfElements`
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
///   - `Tags::VolumeVarsInfo<Metavariables>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct ReceivePoints {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t VolumeDim,
            Requires<tmpl::list_contains_v<
                DbTags, typename ::intrp::Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const typename Metavariables::temporal_id::type& temporal_id,
      std::vector<IdPair<domain::BlockId, tnsr::I<double, VolumeDim,
                                                  typename ::Frame::Logical>>>&&
          block_logical_coords) noexcept {
    db::mutate<intrp::Tags::InterpolatedVarsHolders<Metavariables>>(
        make_not_null(&box),
        [
          &temporal_id, &block_logical_coords
        ](const gsl::not_null<
            db::item_type<intrp::Tags::InterpolatedVarsHolders<Metavariables>>*>
              vars_holders) noexcept {
          auto& vars_infos =
              get<intrp::Vars::HolderTag<InterpolationTargetTag,
                                         Metavariables>>(*vars_holders)
                  .infos;

          // Add the target interpolation points at this temporal_id.
          vars_infos.emplace(std::make_pair(
              temporal_id,
              intrp::Vars::Info<VolumeDim, typename InterpolationTargetTag::
                                               vars_to_interpolate_to_target>{
                  std::move(block_logical_coords)}));
        });

    try_to_interpolate<InterpolationTargetTag>(
        make_not_null(&box), make_not_null(&cache), temporal_id);
  }
};

}  // namespace Actions
}  // namespace intrp
