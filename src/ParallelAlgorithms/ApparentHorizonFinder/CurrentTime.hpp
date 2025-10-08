// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace ah {
template <class Metavariables, typename HorizonMetavars>
struct Component;
template <typename HorizonMetavars>
struct FindApparentHorizon;
}  // namespace ah
/// \endcond

namespace ah {
/*!
 * \brief Determines what the current time should be.
 *
 * \details If there's already a current time, or there are no pending times
 * available, then there's nothing to do. Otherwise, checks if the first pending
 * time is the next to use. If so, sets it as the current time and removes it
 * from pending.
 */
template <typename Fr>
void set_current_time(
    gsl::not_null<std::optional<LinkedMessageId<double>>*> current_time,
    gsl::not_null<std::set<LinkedMessageId<double>>*> pending_times,
    const std::set<LinkedMessageId<double>>& completed_times,
    const std::unordered_map<LinkedMessageId<double>,
                             ah::Storage::SingleTimeStorage<Fr>>& all_storage,
    const ::Verbosity& verbosity, const std::string& name);

/*!
 * \brief Checks if the current time is ready.
 *
 * \details If the current time is after any expiration time, registers a
 * callback for the `ah::FindApparentHorizon` action (but doesn't send the
 * volume variables again because we already did that). Returns if the current
 * time is ready or not.
 */
template <typename HorizonMetavars, typename Metavariables>
bool check_if_current_time_is_ready(
    const LinkedMessageId<double>& current_time,
    Parallel::GlobalCache<Metavariables>& cache,
    const LinkedMessageId<double>& incoming_time,
    const ElementId<3>& incoming_element_id, const ::Mesh<3>& incoming_mesh,
    const std::optional<std::string>& dependency) {
  const auto& domain = get<domain::Tags::Domain<3>>(cache);

  // Now we need to check the functions of time for the current time
  if constexpr (Parallel::is_in_global_cache<Metavariables,
                                             domain::Tags::FunctionsOfTime>) {
    if (domain.is_time_dependent()) {
      auto& this_proxy = Parallel::get_parallel_component<
          Component<Metavariables, HorizonMetavars>>(cache);
      double min_expiration_time = std::numeric_limits<double>::max();
      const Parallel::ArrayComponentId array_component_id =
          Parallel::make_array_component_id<
              Component<Metavariables, HorizonMetavars>>(0);
      // If the functions of time aren't ready, this will set a callback to
      // FindApparentHorizon that will be called by the GlobalCache when
      // domain::Tags::FunctionsOfTime is updated.
      return ::Parallel::mutable_cache_item_is_ready<
          domain::Tags::FunctionsOfTime>(
          cache, array_component_id,
          [&](const std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                  functions_of_time) -> std::unique_ptr<Parallel::Callback> {
            min_expiration_time =
                std::min_element(functions_of_time.begin(),
                                 functions_of_time.end(),
                                 [](const auto& a, const auto& b) {
                                   return a.second->time_bounds()[1] <
                                          b.second->time_bounds()[1];
                                 })
                    ->second->time_bounds()[1];

            // Success: the current time is ok.
            // Failure: the current time is not ok.
            using horizon_frame = typename HorizonMetavars::frame;
            return current_time.id <= min_expiration_time
                       ? std::unique_ptr<Parallel::Callback>{}
                       : std::unique_ptr<Parallel::Callback>(
                             new Parallel::SimpleActionCallback<
                                 FindApparentHorizon<HorizonMetavars>,
                                 decltype(this_proxy), LinkedMessageId<double>,
                                 ElementId<3>, Mesh<3>,
                                 Variables<ah::vars_to_interpolate_to_target<
                                     3, horizon_frame>>,
                                 std::optional<std::string>, bool>(
                                 this_proxy, incoming_time, incoming_element_id,
                                 incoming_mesh,
                                 Variables<ah::vars_to_interpolate_to_target<
                                     3, horizon_frame>>{},
                                 dependency, true));
          });
    }  // if (domain.is_time_dependent())
  } else {
    if (domain.is_time_dependent()) {
      // We error here because the maps are time-dependent, yet
      // the cache does not contain FunctionsOfTime.  It would be
      // nice to make this a compile-time error; however, we want
      // the code to compile for the completely time-independent
      // case where there are no FunctionsOfTime in the cache at
      // all.  Unfortunately, checking whether the maps are
      // time-dependent is currently not constexpr.
      ERROR(
          "There is a time-dependent CoordinateMap in at least one "
          "of the Blocks, but FunctionsOfTime are not in the "
          "GlobalCache.  If you intend to use a time-dependent "
          "CoordinateMap, please add FunctionsOfTime to the GlobalCache.");
    }
  }

  // The current time is ready
  return true;
}
}  // namespace ah
