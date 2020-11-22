// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
/// \cond
template <typename Metavariables>
struct H5WorldtubeBoundary;
template <typename Metavariables>
struct AnalyticWorldtubeBoundary;
template <typename Metavariables>
struct GhWorldtubeBoundary;
/// \endcond
namespace Actions {

namespace detail {
template <typename Initializer, typename ManagerTag,
          typename BoundaryCommunicationTagsList>
struct InitializeWorldtubeBoundaryBase {
  using initialization_tags = tmpl::list<ManagerTag>;
  using initialization_tags_to_keep = tmpl::list<ManagerTag>;
  using const_global_cache_tags = tmpl::list<Tags::LMax>;

  using simple_tags =
      tmpl::list<::Tags::Variables<BoundaryCommunicationTagsList>>;

  template <typename DataBoxTagsList, typename... InboxTags,
            typename ArrayIndex, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DataBoxTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (tmpl::list_contains_v<DataBoxTagsList, ManagerTag>) {
      const size_t l_max = db::get<Tags::LMax>(box);
      Variables<BoundaryCommunicationTagsList> boundary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(boundary_variables));
      return std::make_tuple(std::move(box));
    } else {
      ERROR(MakeString{} << "Missing required boundary manager tag : "
                         << db::tag_name<ManagerTag>);
    }
  }
};
}  // namespace detail

/*!
 * \ingroup ActionsGroup
 * \brief Generic action for initializing various worldtube boundary components.
 *
 * \details See specializations of this class for initialization details for
 * individual worldtube components.
 */
template <typename WorldtubeComponent>
struct InitializeWorldtubeBoundary;

/*!
 * \ingroup ActionsGroup
 * \brief Initializes a H5WorldtubeBoundary
 *
 * \details Uses:
 * - initialization tag
 * `Cce::Tags::H5WorldtubeBoundaryDataManager`,
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
template <typename Metavariables>
struct InitializeWorldtubeBoundary<H5WorldtubeBoundary<Metavariables>>
    : public detail::InitializeWorldtubeBoundaryBase<
          InitializeWorldtubeBoundary<H5WorldtubeBoundary<Metavariables>>,
          Tags::H5WorldtubeBoundaryDataManager,
          typename Metavariables::cce_boundary_communication_tags> {
  using base_type = detail::InitializeWorldtubeBoundaryBase<
      InitializeWorldtubeBoundary<H5WorldtubeBoundary<Metavariables>>,
      Tags::H5WorldtubeBoundaryDataManager,
      typename Metavariables::cce_boundary_communication_tags>;
  using base_type::apply;
  using typename base_type::simple_tags;
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::EndTimeFromFile, Tags::StartTimeFromFile>;
  using typename base_type::initialization_tags;
  using typename base_type::initialization_tags_to_keep;
};

/*!
 * \ingroup ActionsGroup
 * \brief Initializes a GhWorldtubeBoundary
 *
 * \details Uses:
 * - initialization tags
 * `Cce::Tags::GhWorldtubeBoundaryDataManager`, `Tags::GhInterfaceManager`
 * - const global cache tags `Tags::LMax`, `Tags::ExtractionRadius`.
 *
 * Databox changes:
 * - Adds:
 *   - `Tags::Variables<typename
 * Metavariables::cce_boundary_communication_tags>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename Metavariables>
struct InitializeWorldtubeBoundary<GhWorldtubeBoundary<Metavariables>>
    : public detail::InitializeWorldtubeBoundaryBase<
          InitializeWorldtubeBoundary<GhWorldtubeBoundary<Metavariables>>,
          Tags::GhInterfaceManager,
          typename Metavariables::cce_boundary_communication_tags> {
  using base_type = detail::InitializeWorldtubeBoundaryBase<
      InitializeWorldtubeBoundary<GhWorldtubeBoundary<Metavariables>>,
      Tags::GhInterfaceManager,
      typename Metavariables::cce_boundary_communication_tags>;
  using base_type::apply;
  using typename base_type::simple_tags;

  using const_global_cache_tags =
      tmpl::list<Tags::LMax, InitializationTags::ExtractionRadius,
                 Tags::NoEndTime, Tags::SpecifiedStartTime,
                 Tags::InterfaceManagerInterpolationStrategy>;
  using typename base_type::initialization_tags;
  using typename base_type::initialization_tags_to_keep;
};
}  // namespace Actions
}  // namespace Cce
