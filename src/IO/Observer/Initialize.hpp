// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
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
      db::AddSimpleTags<Tags::NumberOfEvents, Tags::ReductionArrayComponentIds,
                        Tags::VolumeArrayComponentIds, Tags::TensorData,
                        Tags::ReductionObserversContributed>,
      typename Metavariables::observed_reduction_data_tags,
      tmpl::transform<
          typename Metavariables::observed_reduction_data_tags,
          tmpl::bind<detail::reduction_data_to_reduction_names, tmpl::_1>>>;
  using compute_tags = db::AddComputeTags<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename... InboxTags, typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return helper(typename Metavariables::observed_reduction_data_tags{});
  }

 private:
  template <typename... ReductionTags>
  static auto helper(tmpl::list<ReductionTags...> /*meta*/) noexcept {
    return std::make_tuple(db::create<simple_tags>(
        db::item_type<Tags::NumberOfEvents>{},
        db::item_type<Tags::ReductionArrayComponentIds>{},
        db::item_type<Tags::VolumeArrayComponentIds>{},
        db::item_type<Tags::TensorData>{},
        db::item_type<Tags::ReductionObserversContributed>{},
        db::item_type<ReductionTags>{}...,
        db::item_type<
            detail::reduction_data_to_reduction_names<ReductionTags>>{}...));
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
      db::AddSimpleTags<Tags::TensorData, Tags::VolumeObserversRegistered,
                        Tags::VolumeObserversContributed,
                        Tags::ReductionObserversRegistered,
                        Tags::ReductionObserversRegisteredNodes,
                        Tags::ReductionObserversContributed, Tags::H5FileLock>,
      typename Metavariables::observed_reduction_data_tags,
      tmpl::transform<
          typename Metavariables::observed_reduction_data_tags,
          tmpl::bind<detail::reduction_data_to_reduction_names, tmpl::_1>>>;
  using compute_tags = db::AddComputeTags<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename... InboxTags, typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return helper(typename Metavariables::observed_reduction_data_tags{});
  }

 private:
  template <typename... ReductionTags>
  static auto helper(tmpl::list<ReductionTags...> /*meta*/) noexcept {
    return std::make_tuple(db::create<simple_tags>(
        db::item_type<Tags::TensorData>{},
        db::item_type<Tags::VolumeObserversRegistered>{},
        db::item_type<Tags::VolumeObserversContributed>{},
        db::item_type<Tags::ReductionObserversRegistered>{},
        db::item_type<Tags::ReductionObserversRegisteredNodes>{},
        db::item_type<Tags::ReductionObserversContributed>{},
        Parallel::create_lock(), db::item_type<ReductionTags>{}...,
        db::item_type<
            detail::reduction_data_to_reduction_names<ReductionTags>>{}...));
  }
};
}  // namespace Actions
}  // namespace observers
