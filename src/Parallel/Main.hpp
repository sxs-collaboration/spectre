// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the Charm++ mainchare.

#pragma once

#include <charm++.h>
#include <initializer_list>
#include <type_traits>

#include "Informer/Informer.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Main.decl.h"

namespace Parallel {

/// \ingroup Parallel
/// The main function of a Charm++ executable.
///
/// Metavariables must define the following:
///   - const_global_cache_tag_list  [typelist of tags of constant data]
///   - tentacle_list   [typelist of Tentacles]
///   - phase   [enum class listing phases of the executable]
///   - determine_next_phase   [function that determines the next phase of the
///   executable]
///
/// Each Tentacle in Metavariables::tentacle_list must define the following
/// functions:
///   - initialize
///   - execute_next_global_actions
///
/// The phases in Metavariables::phase must include Initialization (the initial
/// phase) and Exit (the final phase)
template <typename Metavariables>
class Main : public CBase_Main<Metavariables> {
 public:
  using tentacle_list = typename Metavariables::tentacle_list;

  explicit Main(CkArgMsg* msg) noexcept;
  explicit Main(CkMigrateMessage* /*msg*/) {}

  /// Initialize the tentacles.
  void initialize() noexcept;

  /// Determine the next phase of the simulation and execute it.
  void execute_next_phase() noexcept;

 private:
  typename Metavariables::phase current_phase_{
      Metavariables::phase::Initialization};
  CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
};

// ================================================================

template <typename Metavariables>
Main<Metavariables>::Main(CkArgMsg* msg) noexcept {
  Informer::print_startup_info(msg);

  /// \todo detail::register_events_to_trace();

  /// \todo Parse the input file
  /// \todo Create the options
  /// \todo Use the options to fill the ConstGlobalCache
  tuples::TaggedTupleTypelist<
      typename Metavariables::const_global_cache_tag_list>
      const_data_to_be_cached;

  const_global_cache_proxy_ =
      CProxy_ConstGlobalCache<Metavariables>::ckNew(const_data_to_be_cached);

  tuples::TaggedTupleTypelist<tentacle_list> the_tentacles;

  // Construct the group proxies with a dependency on the ConstGlobalCache proxy
  using group_tentacle_list = tmpl::filter<
      tentacle_list,
      tmpl::or_<Parallel::is_group_proxy<tmpl::bind<tmpl::type_from, tmpl::_1>>,
                Parallel::is_node_group_proxy<
                    tmpl::bind<tmpl::type_from, tmpl::_1>>>>;
  CkEntryOptions const_global_cache_dependency;
  const_global_cache_dependency.setGroupDepID(
      const_global_cache_proxy_.ckGetGroupID());
  tmpl::for_each<group_tentacle_list>([
    this, &the_tentacles, &const_global_cache_dependency
  ](auto tentacle) noexcept {
    tuples::get<typename decltype(tentacle)::type>(the_tentacles) =
        decltype(tentacle)::type::type::ckNew(const_global_cache_proxy_,
                                              &const_global_cache_dependency);
  });

  // Construct the proxies for the single chares
  using single_tentacle_list = tmpl::filter<
      tentacle_list,
      Parallel::is_chare_proxy<tmpl::bind<tmpl::type_from, tmpl::_1>>>;
  tmpl::for_each<single_tentacle_list>(
      [ this, &the_tentacles ](auto tentacle) noexcept {
        tuples::get<typename decltype(tentacle)::type>(the_tentacles) =
            decltype(tentacle)::type::type::ckNew(const_global_cache_proxy_);
      });

  // Create proxies for empty array chares (which are created by the initialize
  // functions of the tentacles)
  using array_tentacle_list =
      tmpl::filter<tentacle_list,
                   tmpl::and_<Parallel::is_array_proxy<
                                  tmpl::bind<tmpl::type_from, tmpl::_1>>,
                              tmpl::not_<Parallel::is_bound_array<tmpl::_1>>>>;
  tmpl::for_each<array_tentacle_list>([&the_tentacles](auto tentacle) noexcept {
    tuples::get<typename decltype(tentacle)::type>(the_tentacles) =
        decltype(tentacle)::type::type::ckNew();
  });

  // Create proxies for empty bound array chares
  using bound_array_tentacle_list =
      tmpl::filter<tentacle_list,
                   tmpl::and_<Parallel::is_array_proxy<
                                  tmpl::bind<tmpl::type_from, tmpl::_1>>,
                              Parallel::is_bound_array<tmpl::_1>>>;
  tmpl::for_each<bound_array_tentacle_list>([&the_tentacles](
      auto tentacle) noexcept {
    CkArrayOptions opts;
    opts.bindTo(
        tuples::get<typename decltype(tentacle)::type::bind_to>(the_tentacles));
    tuples::get<typename decltype(tentacle)::type>(the_tentacles) =
        decltype(tentacle)::type::type::ckNew(opts);
  });

  // Send the complete list of tentacles to the ConstGlobalCache on each Charm++
  // node.  After all nodes have finished, the callback is executed.
  CkCallback callback(CkIndex_Main<Metavariables>::initialize(),
                      this->thisProxy);
  const_global_cache_proxy_.set_tentacles(the_tentacles, callback);
}

template <typename Metavariables>
void Main<Metavariables>::initialize() noexcept {
  tmpl::for_each<typename Metavariables::tentacle_list>([this](
      auto tentacle) noexcept {
    decltype(tentacle)::type::initialize(const_global_cache_proxy_);
  });
  CkStartQD(CkCallback(CkIndex_Main<Metavariables>::execute_next_phase(),
                       this->thisProxy));
}

template <typename Metavariables>
void Main<Metavariables>::execute_next_phase() noexcept {
  current_phase_ =
      Metavariables::determine_next_phase(const_global_cache_proxy_);
  if (Metavariables::phase::Exit == current_phase_) {
    Informer::print_exit_info();
    Parallel::exit();
  }
  tmpl::for_each<typename Metavariables::tentacle_list>([this](
      auto tentacle) noexcept {
    decltype(tentacle)::type::execute_next_global_actions(
        current_phase_, const_global_cache_proxy_);
  });
  CkStartQD(CkCallback(CkIndex_Main<Metavariables>::execute_next_phase(),
                       this->thisProxy));
}

}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/Main.def.h"
#undef CK_TEMPLATES_ONLY
