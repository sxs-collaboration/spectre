// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {
namespace Actions {
namespace detail {
template <class Tag>
using reduction_data_to_reduction_names = typename Tag::names_tag;
}  // namespace detail
/*!
 * \brief Initializes the DataBox on the observer parallel component
 *
 * Uses:
 * - Metavariables:
 *   - `observed_reduction_data_tags` (see ContributeReductionData)
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <class Metavariables>
struct Initialize {
  using simple_tags = tmpl::append<
      tmpl::list<Tags::ExpectedContributorsForObservations,
                 Tags::ContributorsOfReductionData,
                 Tags::ContributorsOfTensorData, Tags::TensorData>,
      typename Metavariables::observed_reduction_data_tags,
      tmpl::transform<
          typename Metavariables::observed_reduction_data_tags,
          tmpl::bind<detail::reduction_data_to_reduction_names, tmpl::_1>>>;
  using compute_tags = tmpl::list<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    return std::make_tuple(std::move(box));
  }
};

/*!
 * \brief Initializes the DataBox of the observer parallel component that writes
 * to disk.
 *
 * Uses:
 * - Metavariables:
 *   - `observed_reduction_data_tags` (see ContributeReductionData)
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <class Metavariables>
struct InitializeWriter {
  using simple_tags = tmpl::append<
      tmpl::list<Tags::ExpectedContributorsForObservations,
                 Tags::ContributorsOfReductionData, Tags::ReductionDataLock,
                 Tags::ContributorsOfTensorData, Tags::VolumeDataLock,
                 Tags::TensorData, Tags::NodesExpectedToContributeReductions,
                 Tags::NodesThatContributedReductions, Tags::H5FileLock>,
      typename Metavariables::observed_reduction_data_tags,
      tmpl::transform<
          typename Metavariables::observed_reduction_data_tags,
          tmpl::bind<detail::reduction_data_to_reduction_names, tmpl::_1>>>;
  using compute_tags = tmpl::list<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace observers
