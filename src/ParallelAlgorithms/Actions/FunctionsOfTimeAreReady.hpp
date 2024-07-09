// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/StdHelpers.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace domain {
namespace detail {
template <typename CacheTag, typename Callback, typename Metavariables,
          typename ArrayIndex, typename Component, typename... Args>
bool functions_of_time_are_ready_impl(
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const Component* /*meta*/, const double time,
    const std::optional<std::unordered_set<std::string>>& functions_to_check,
    Args&&... args) {
  if constexpr (Parallel::is_in_mutable_global_cache<Metavariables, CacheTag>) {
    const auto& proxy =
        ::Parallel::get_parallel_component<Component>(cache)[array_index];
    const Parallel::ArrayComponentId array_component_id =
        Parallel::make_array_component_id<Component>(array_index);

    return Parallel::mutable_cache_item_is_ready<CacheTag>(
        cache, array_component_id,
        [&functions_to_check, &proxy, &time,
         &args...](const std::unordered_map<
                   std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                       functions_of_time) {
          using ::operator<<;
          ASSERT(
              alg::all_of(
                  functions_to_check.value_or(
                      std::unordered_set<std::string>{}),
                  [&functions_of_time](const std::string& function_to_check) {
                    return functions_of_time.count(function_to_check) == 1;
                  }),
              "Not all functions to check ("
                  << functions_to_check.value() << ") are in the global cache ("
                  << keys_of(functions_of_time) << ")");
          for (const auto& [name, f_of_t] : functions_of_time) {
            if (functions_to_check.has_value() and
                functions_to_check->count(name) == 0) {
              continue;
            }
            const double expiration_time = f_of_t->time_bounds()[1];
            if (time > expiration_time) {
              return std::unique_ptr<Parallel::Callback>(
                  new Callback(proxy, std::forward<Args>(args)...));
            }
          }
          return std::unique_ptr<Parallel::Callback>{};
        });
  } else {
    (void)cache;
    (void)array_index;
    (void)time;
    (void)functions_to_check;
    EXPAND_PACK_LEFT_TO_RIGHT((void)args);
    return true;
  }
}
}  // namespace detail

/// \ingroup ComputationalDomainGroup
/// Check that functions of time are up-to-date.
///
/// Check that functions of time in \p CacheTag with names in \p
/// functions_to_check are ready at time \p time.  If  \p functions_to_check is
/// a `std::nullopt`, checks all functions in \p CacheTag.  If any function is
/// not ready, schedules a `Parallel::PerformAlgorithmCallback` with the
/// GlobalCache..
template <typename CacheTag, typename Metavariables, typename ArrayIndex,
          typename Component>
bool functions_of_time_are_ready_algorithm_callback(
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const Component* component_p, const double time,
    const std::optional<std::unordered_set<std::string>>& functions_to_check =
        std::nullopt) {
  using ProxyType =
      std::decay_t<decltype(::Parallel::get_parallel_component<Component>(
          cache)[array_index])>;
  return detail::functions_of_time_are_ready_impl<
      CacheTag, Parallel::PerformAlgorithmCallback<ProxyType>>(
      cache, array_index, component_p, time, functions_to_check);
}

/// \ingroup ComputationalDomainGroup
/// Check that functions of time are up-to-date.
///
/// Check that functions of time in \p CacheTag with names in \p
/// functions_to_check are ready at time \p time.  If  \p functions_to_check is
/// a `std::nullopt`, checks all functions in \p CacheTag.  If any function is
/// not ready, schedules a `Parallel::SimpleActionCallback` with the GlobalCache
/// which calls the simple action passed in as a template parameter. The `Args`
/// are forwareded to the callback.
template <typename CacheTag, typename SimpleAction, typename Metavariables,
          typename ArrayIndex, typename Component, typename... Args>
bool functions_of_time_are_ready_simple_action_callback(
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const Component* component_p, const double time,
    const std::optional<std::unordered_set<std::string>>& functions_to_check,
    Args&&... args) {
  using ProxyType =
      std::decay_t<decltype(::Parallel::get_parallel_component<Component>(
          cache)[array_index])>;
  return detail::functions_of_time_are_ready_impl<
      CacheTag,
      Parallel::SimpleActionCallback<SimpleAction, ProxyType, Args...>>(
      cache, array_index, component_p, time, functions_to_check,
      std::forward<Args>(args)...);
}

/// \ingroup ComputationalDomainGroup
/// Check that functions of time are up-to-date.
///
/// Check that functions of time in \p CacheTag with names in \p
/// functions_to_check are ready at time \p time.  If  \p functions_to_check is
/// a `std::nullopt`, checks all functions in \p CacheTag.  If any function is
/// not ready, schedules a `Parallel::ThreadedActionCallback` with the
/// GlobalCache which calls the threaded action passed in as a template
/// parameter. The `Args` are forwareded to the callback.
template <typename CacheTag, typename ThreadedAction, typename Metavariables,
          typename ArrayIndex, typename Component, typename... Args>
bool functions_of_time_are_ready_threaded_action_callback(
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const Component* component_p, const double time,
    const std::optional<std::unordered_set<std::string>>& functions_to_check,
    Args&&... args) {
  using ProxyType =
      std::decay_t<decltype(::Parallel::get_parallel_component<Component>(
          cache)[array_index])>;
  return detail::functions_of_time_are_ready_impl<
      CacheTag,
      Parallel::ThreadedActionCallback<ThreadedAction, ProxyType, Args...>>(
      cache, array_index, component_p, time, functions_to_check,
      std::forward<Args>(args)...);
}

namespace Actions {
/// \ingroup ComputationalDomainGroup
/// Check that functions of time are up-to-date.
///
/// Wait for all functions of time in `domain::Tags::FunctionsOfTime`
/// to be ready at `::Tags::Time`.  This ensures that the coordinates
/// can be safely accessed in later actions without first verifying
/// the state of the time-dependent maps.
struct CheckFunctionsOfTimeAreReady {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* component) {
    const bool ready = functions_of_time_are_ready_algorithm_callback<
        domain::Tags::FunctionsOfTime>(cache, array_index, component,
                                       db::get<::Tags::Time>(box));
    return {ready ? Parallel::AlgorithmExecution::Continue
                  : Parallel::AlgorithmExecution::Retry,
            std::nullopt};
  }
};
}  // namespace Actions

/// \ingroup ComputationalDomainGroup
/// Dense-output postprocessor to check that functions of time are up-to-date.
///
/// Check that all functions of time in
/// `domain::Tags::FunctionsOfTime` are ready at `::Tags::Time`.  This
/// ensures that the coordinates can be safely accessed in later
/// actions without first verifying the state of the time-dependent
/// maps.  This postprocessor does not actually modify anything.
struct CheckFunctionsOfTimeAreReadyPostprocessor {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  static void apply() {}

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  static bool is_ready(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ParallelComponent* component) {
    return functions_of_time_are_ready_algorithm_callback<
        domain::Tags::FunctionsOfTime>(cache, array_index, component,
                                       db::get<::Tags::Time>(*box));
  }
};
}  // namespace domain
