// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
/// [executable_example_includes]
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP
/// [executable_example_includes]

/// [executable_example_options]
namespace OptionTags {
struct Name {
  using type = std::string;
  static constexpr Options::String help{"A name"};
};
}  // namespace OptionTags

namespace Tags {
struct Name : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::Name>;

  static constexpr bool pass_metavariables = false;
  static std::string create_from_options(const std::string& name) noexcept {
    return name;
  }
};
}  // namespace Tags
/// [executable_example_options]

/// [executable_example_action]
namespace Actions {
struct PrintMessage {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::printf("Hello %s from process %d on node %d!\n",
                     Parallel::get<Tags::Name>(cache), sys::my_proc(),
                     sys::my_node());
  }
};
}  // namespace Actions
/// [executable_example_action]

/// [executable_example_singleton]
template <class Metavariables>
struct HelloWorld {
  using const_global_cache_tags = tmpl::list<Tags::Name>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept;
};

template <class Metavariables>
void HelloWorld<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase /* next_phase */,
    Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept {
  Parallel::simple_action<Actions::PrintMessage>(
      Parallel::get_parallel_component<HelloWorld>(
          *(global_cache.ckLocalBranch())));
}
/// [executable_example_singleton]

/// [executable_example_metavariables]
struct Metavars {
  using component_list = tmpl::list<HelloWorld<Metavars>>;

  static constexpr Options::String help{
      "Say hello from a singleton parallel component."};

  enum class Phase { Initialization, Execute, Exit };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};
/// [executable_example_metavariables]

/// [executable_example_charm_init]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// [executable_example_charm_init]
/// \endcond
