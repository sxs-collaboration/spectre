// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>
#include <optional>
#include <type_traits>

#include "Parallel/CharmRegistration.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControlReductionHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Main.decl.h"

namespace Parallel {
/// @{
/// Send data from a parallel component to the Main chare for making
/// phase-change decisions.
///
/// This function is distinct from `Parallel::contribute_to_reduction` because
/// this function contributes reduction data to the Main chare via the entry
/// method `phase_change_reduction`. This must be done because the entry method
/// must alter member data specific to the Main chare, and the Main chare cannot
/// execute actions like most SpECTRE parallel components.
/// For all cases other than sending phase-change decision data to the Main
/// chare, you should use `Parallel::contribute_to_reduction`.
template <typename SenderComponent, typename ArrayIndex, typename Metavariables,
          class... Ts>
void contribute_to_phase_change_reduction(
    tuples::TaggedTuple<Ts...> data_for_reduction,
    Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index) {
  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<Metavariables>;
  using reduction_data_type =
      PhaseControl::reduction_data<tmpl::list<Ts...>,
                                   phase_change_tags_and_combines_list>;
  (void)Parallel::charmxx::RegisterReducerFunction<
      reduction_data_type::combine>::registrar;
  CkCallback callback(CProxy_Main<Metavariables>::index_t::
                          template redn_wrapper_phase_change_reduction<
                              PhaseControl::TaggedTupleCombine, Ts...>(nullptr),
                      cache.get_main_proxy().value());
  reduction_data_type reduction_data{data_for_reduction};
  Parallel::local(
      Parallel::get_parallel_component<SenderComponent>(cache)[array_index])
      ->contribute(static_cast<int>(reduction_data.size()),
                   reduction_data.packed().get(),
                   Parallel::charmxx::charm_reducer_functions.at(
                       std::hash<Parallel::charmxx::ReducerFunctions>{}(
                           &reduction_data_type::combine)),
                   callback);
}

template <typename SenderComponent, typename Metavariables, class... Ts>
void contribute_to_phase_change_reduction(
    tuples::TaggedTuple<Ts...> data_for_reduction,
    Parallel::GlobalCache<Metavariables>& cache) {
  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<Metavariables>;
  using reduction_data_type =
      PhaseControl::reduction_data<tmpl::list<Ts...>,
                                   phase_change_tags_and_combines_list>;
  (void)Parallel::charmxx::RegisterReducerFunction<
      reduction_data_type::combine>::registrar;
  CkCallback callback(CProxy_Main<Metavariables>::index_t::
                          template redn_wrapper_phase_change_reduction<
                              PhaseControl::TaggedTupleCombine, Ts...>(nullptr),
                      cache.get_main_proxy().value());
  reduction_data_type reduction_data{data_for_reduction};
  // Note that Singletons could be supported by directly calling the main
  // entry function, but due to this and other peculiarities with
  // Singletons, it is best to discourage their use.
  static_assert(
      not std::is_same_v<typename SenderComponent::chare_type,
                         Parallel::Algorithms::Singleton>,
      "Phase change reduction is not supported for singleton chares. "
      "Consider constructing your chare as a length-1 array chare if you "
      "need to contribute to phase change data");
  Parallel::local_branch(
      Parallel::get_parallel_component<SenderComponent>(cache))
      ->contribute(static_cast<int>(reduction_data.size()),
                   reduction_data.packed().get(),
                   Parallel::charmxx::charm_reducer_functions.at(
                       std::hash<Parallel::charmxx::ReducerFunctions>{}(
                           &reduction_data_type::combine)),
                   callback);
}
/// @}
}  // namespace Parallel
