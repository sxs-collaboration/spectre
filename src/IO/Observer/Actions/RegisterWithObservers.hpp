// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"

/// \cond
template <class... Tags>
class TaggedTuple;
/// \endcond

namespace observers::Actions {
/*!
 * \brief Register an observation ID with the observers.
 *
 * \warning If registering events, you should use RegisterEventsWithObservers
 * instead. If your event is not compatible with RegisterEventsWithObservers,
 * please make it so.
 *
 * The `RegisterHelper` passed as a template parameter must have a static
 * `register_info` function that takes as its first template parameter the
 * `ParallelComponent` and as function arguments a `db::DataBox` and the array
 * component index. The function must return a
 * `std::pair<observers::TypeOfObservation, observers::ObservationId>`
 */
template <typename RegisterHelper>
struct RegisterWithObservers {
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagList>&&> apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    const auto [type_of_observation, observation_id] =
        RegisterHelper::template register_info<ParallelComponent>(box,
                                                                  array_index);

    Parallel::simple_action<observers::Actions::RegisterSenderWithSelf>(
        observer, observation_id,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<std::decay_t<ArrayIndex>>{array_index}),
        type_of_observation);
    return {std::move(box)};
  }
};
}  // namespace observers::Actions
