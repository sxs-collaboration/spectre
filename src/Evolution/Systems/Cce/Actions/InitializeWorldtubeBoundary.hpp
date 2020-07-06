// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
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
 */
struct InitializeH5WorldtubeBoundary {
  using initialization_tags =
      tmpl::list<InitializationTags::H5WorldtubeBoundaryDataManager>;

  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::EndTimeFromFile, Tags::StartTimeFromFile>;

  template <class Metavariables>
  using h5_boundary_manager_simple_tags = db::AddSimpleTags<
      ::Tags::Variables<
          typename Metavariables::cce_boundary_communication_tags>,
      Tags::H5WorldtubeBoundaryDataManager>;

  template <class Metavariables>
  using return_tag_list = h5_boundary_manager_simple_tags<Metavariables>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, InitializationTags::H5WorldtubeBoundaryDataManager>> =
                nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = db::get<Tags::LMax>(box);
    Variables<typename Metavariables::cce_boundary_communication_tags>
        boundary_variables{
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    WorldtubeDataManager boundary_data_manager;
    db::mutate<InitializationTags::H5WorldtubeBoundaryDataManager>(
        make_not_null(&box),
        [&boundary_data_manager](const gsl::not_null<WorldtubeDataManager*>
                                     initialization_data_manager) noexcept {
          boundary_data_manager = std::move(*initialization_data_manager);
        });
    auto initial_box = Initialization::merge_into_databox<
        InitializeH5WorldtubeBoundary,
        h5_boundary_manager_simple_tags<Metavariables>, db::AddComputeTags<>,
        Initialization::MergePolicy::Overwrite>(
        std::move(box), std::move(boundary_variables),
        std::move(boundary_data_manager));

    return std::make_tuple(std::move(initial_box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTags, InitializationTags::H5WorldtubeBoundaryDataManager>> =
                nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
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
 */
struct InitializeGhWorldtubeBoundary {
  using initialization_tags = tmpl::list<Tags::GhInterfaceManager>;

  using initialization_tags_to_keep = tmpl::list<Tags::GhInterfaceManager>;

  using const_global_cache_tags =
      tmpl::list<Tags::LMax, InitializationTags::ExtractionRadius,
                 Tags::NoEndTime, Tags::SpecifiedStartTime>;

  template <class Metavariables>
  using gh_boundary_manager_simple_tags = db::AddSimpleTags<
      ::Tags::Variables<
          typename Metavariables::cce_boundary_communication_tags>,
      Tags::GhInterfaceManager>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTags, Tags::GhInterfaceManager>> =
                nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = db::get<Tags::LMax>(box);
    Variables<typename Metavariables::cce_boundary_communication_tags>
        boundary_variables{
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    auto initial_box = Initialization::merge_into_databox<
        InitializeGhWorldtubeBoundary,
        gh_boundary_manager_simple_tags<Metavariables>, db::AddComputeTags<>,
        Initialization::MergePolicy::Overwrite>(
        std::move(box), std::move(boundary_variables),
        db::get<Tags::GhInterfaceManager>(box).get_clone());

    return std::make_tuple(std::move(initial_box));
  }

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTags, Tags::GhInterfaceManager>> =
          nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Required tag `Tags::GhInterfaceManager` is missing from the DataBox");
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
