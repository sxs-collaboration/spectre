// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
/// [executable_example_includes]
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
/// [executable_example_includes]

/// [executable_example_options]
namespace Tags {
struct Name : db::SimpleTag {
  static std::string name() noexcept { return "Name"; }
  using type = std::string;
};
}  // namespace Tags
namespace OptionTags {
struct Name {
  using type = std::string;
  static constexpr OptionString help{"A name"};
  using container_tag = Tags::Name;
};
}  // namespace OptionTags
/// [executable_example_options]

/// [executable_example_action]
namespace Actions {
struct PrintMessage {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::printf("Hello %s from process %d on node %d!\n",
                     Parallel::get<OptionTags::Name>(cache),
                     Parallel::my_proc(), Parallel::my_node());
  }
};
}  // namespace Actions
/// [executable_example_action]

/// [executable_example_singleton]
template <class Metavariables>
struct HelloWorld {
  using const_global_cache_tag_list = tmpl::list<OptionTags::Name>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using options = tmpl::list<>;
  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /* global_cache */) noexcept {}
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept;
};

template <class Metavariables>
void HelloWorld<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase /* next_phase */,
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
  Parallel::simple_action<Actions::PrintMessage>(
      Parallel::get_parallel_component<HelloWorld>(
          *(global_cache.ckLocalBranch())));
}
/// [executable_example_singleton]

/// [executable_example_metavariables]
struct Metavars {
  using const_global_cache_tag_list = tmpl::list<>;

  using component_list = tmpl::list<HelloWorld<Metavars>>;

  static constexpr OptionString help{
      "Say hello from a singleton parallel component."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }
};
/// [executable_example_metavariables]

/// [executable_example_charm_init]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// [executable_example_charm_init]
/// \endcond
