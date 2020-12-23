// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Initializes a H5WorldtubeBoundary
 *
 * \details Uses:
 * - initialization tag
 * `Cce::InitializationTags::H5WorldtubeBoundaryDataManager`,
 * - const global cache tag `Cce::Tags::LMax`.
 *
 * Databox changes:
 * - Adds:
 *   - `Cce::Tags::H5WorldtubeBoundaryDataManager`
 *   - `Tags::Variables<typename
 * Metavariables::cce_boundary_communication_tags>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename BoundaryCommunicationTags>
struct InitializeH5WorldtubeBoundary {
  using initialization_tags =
      tmpl::list<Tags::H5WorldtubeBoundaryDataManager>;
  using initialization_tags_to_keep =
      tmpl::list<Tags::H5WorldtubeBoundaryDataManager>;
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::EndTimeFromFile, Tags::StartTimeFromFile>;

  using simple_tags = tmpl::list<::Tags::Variables<BoundaryCommunicationTags>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, Tags::H5WorldtubeBoundaryDataManager>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = db::get<Tags::LMax>(box);
    Variables<typename Metavariables::cce_boundary_communication_tags>
        boundary_variables{
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(boundary_variables));

    return std::make_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTags, Tags::H5WorldtubeBoundaryDataManager>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(std::move(box));
  }
};

/*!
 * \ingroup ActionsGroup
 * \brief Initializes a GhWorldtubeBoundary
 *
 * \details Uses:
 * - initialization tag
 * `Cce::InitializationTags::GhWorldtubeBoundaryDataManager`,
 * - const global cache tags `InitializationTags::LMax`,
 * `InitializationTags::ExtractionRadius`.
 *
 * Databox changes:
 * - Adds:
 *   - `Tags::Variables<typename
 * Metavariables::cce_boundary_communication_tags>`
 *   - `Tags::GhInterfaceManager` (cloned from
 * `InitializationTags::GhInterfaceManager`)
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename BoundaryCommunicationTags>
struct InitializeGhWorldtubeBoundary {
  using initialization_tags = tmpl::list<Tags::GhInterfaceManager>;

  using initialization_tags_to_keep = tmpl::list<Tags::GhInterfaceManager>;

  using const_global_cache_tags =
      tmpl::list<Tags::LMax, InitializationTags::ExtractionRadius,
                 Tags::NoEndTime, Tags::SpecifiedStartTime,
                 Tags::InterfaceManagerInterpolationStrategy>;

  using simple_tags =
      db::AddSimpleTags<::Tags::Variables<BoundaryCommunicationTags>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = db::get<Tags::LMax>(box);
    Variables<typename Metavariables::cce_boundary_communication_tags>
        boundary_variables{
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(boundary_variables));
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
