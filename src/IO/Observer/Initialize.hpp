// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
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
 */
template <class Metavariables>
struct Initialize {
  using simple_tags = tmpl::append<
      db::AddSimpleTags<Tags::ExpectedContributorsForObservations,
                        Tags::ReductionsContributed,
                        Tags::ContributorsOfTensorData, Tags::TensorData>,
      typename Metavariables::observed_reduction_data_tags,
      tmpl::transform<
          typename Metavariables::observed_reduction_data_tags,
          tmpl::bind<detail::reduction_data_to_reduction_names, tmpl::_1>>>;
  using compute_tags = db::AddComputeTags<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (not tmpl::list_contains_v<DbTagsList, Tags::TensorData>) {
      return helper(typename Metavariables::observed_reduction_data_tags{});
    } else {
      ERROR("You appear to be initializing the Observer twice.");
      return std::make_tuple(std::move(box));
    }
  }

 private:
  template <typename... ReductionTags>
  static auto helper(tmpl::list<ReductionTags...> /*meta*/) noexcept {
    return std::make_tuple(
        db::create<simple_tags>(
            db::item_type<Tags::ExpectedContributorsForObservations>{},
            db::item_type<Tags::ReductionsContributed>{},
            db::item_type<Tags::ContributorsOfTensorData>{},
            db::item_type<Tags::TensorData>{},
            db::item_type<ReductionTags>{}...,
            db::item_type<
                detail::reduction_data_to_reduction_names<ReductionTags>>{}...),
        true);
  }
};

/*!
 * \brief Initializes the DataBox of the observer parallel component that writes
 * to disk.
 *
 * Uses:
 * - Metavariables:
 *   - `observed_reduction_data_tags` (see ContributeReductionData)
 */
template <class Metavariables>
struct InitializeWriter {
  using simple_tags = tmpl::append<
      db::AddSimpleTags<Tags::ExpectedContributorsForObservations,
                        Tags::ReductionsContributed, Tags::ReductionDataLock,
                        Tags::ContributorsOfTensorData, Tags::VolumeDataLock,
                        Tags::TensorData,
                        Tags::NodesExpectedToContributeReductions,
                        Tags::NodesThatContributedReductions, Tags::H5FileLock>,
      typename Metavariables::observed_reduction_data_tags,
      tmpl::transform<
          typename Metavariables::observed_reduction_data_tags,
          tmpl::bind<detail::reduction_data_to_reduction_names, tmpl::_1>>>;
  using compute_tags = db::AddComputeTags<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (not tmpl::list_contains_v<DbTagsList, Tags::TensorData>) {
      return helper(typename Metavariables::observed_reduction_data_tags{});
    } else {
      ERROR("You appear to be initializing the ObserverWriter twice.");
      return std::make_tuple(std::move(box));
    }
  }

 private:
  template <typename... ReductionTags>
  static auto helper(tmpl::list<ReductionTags...> /*meta*/) noexcept {
    return std::make_tuple(
        db::create<simple_tags>(
            db::item_type<Tags::ExpectedContributorsForObservations>{},
            db::item_type<Tags::ReductionsContributed>{}, Parallel::NodeLock{},
            db::item_type<Tags::ContributorsOfTensorData>{},
            Parallel::NodeLock{}, db::item_type<Tags::TensorData>{},
            db::item_type<Tags::NodesExpectedToContributeReductions>{},
            db::item_type<Tags::NodesThatContributedReductions>{},
            Parallel::NodeLock{}, db::item_type<ReductionTags>{}...,
            db::item_type<
                detail::reduction_data_to_reduction_names<ReductionTags>>{}...),
        true);
  }
};
}  // namespace Actions
}  // namespace observers
