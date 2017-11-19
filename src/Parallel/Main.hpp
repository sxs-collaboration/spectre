// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the Charm++ mainchare.

#pragma once

#include <boost/program_options.hpp>
#include <charm++.h>
#include <initializer_list>
#include <string>
#include <type_traits>

#include "ErrorHandling/Error.hpp"
#include "Informer/Informer.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Main.decl.h"

namespace Parallel {

/// \ingroup ParallelGroup
/// The main function of a Charm++ executable.
///
/// Metavariables must define the following:
///   - help   [c string describing the program]
///   - component_list   [typelist of ParallelComponents]
///   - phase   [enum class listing phases of the executable]
///   - determine_next_phase   [function that determines the next phase of the
///   executable]
/// and may optionally define the following:
///   - input_file   [c string giving default input file name]
///   - ignore_unrecognized_command_line_options   [bool, defaults to false]
///
/// Each ParallelComponent in Metavariables::component_list must define the
/// following functions:
///   - const_global_cache_tag_list  [typelist of tags of constant data]
///   - options  [typelist of option tags for data to be passed to initialize]
///   - initialize
///   - execute_next_global_actions
///
/// The phases in Metavariables::Phase must include Initialization (the initial
/// phase) and Exit (the final phase)
template <typename Metavariables>
class Main : public CBase_Main<Metavariables> {
 public:
  using component_list = typename Metavariables::component_list;
  using const_global_cache_tags =
      typename ConstGlobalCache<Metavariables>::tag_list;

  explicit Main(CkArgMsg* msg) noexcept;
  explicit Main(CkMigrateMessage* /*msg*/)
      : options_("Uninitialized after migration") {}

  /// Initialize the parallel_components.
  void initialize() noexcept;

  /// Determine the next phase of the simulation and execute it.
  void execute_next_phase() noexcept;

 private:
  template <typename ParallelComponent>
  using parallel_component_options = typename ParallelComponent::options;
  using option_list = tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
      const_global_cache_tags,
      tmpl::transform<component_list,
                      tmpl::bind<parallel_component_options, tmpl::_1>>>>>;
  using parallel_component_tag_list = tmpl::transform<
      component_list,
      tmpl::bind<
          tmpl::type_,
          tmpl::bind<Parallel::proxy_from_parallel_component, tmpl::_1>>>;
  typename Metavariables::Phase current_phase_{
      Metavariables::Phase::Initialization};

  CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
  Options<option_list> options_;
};

// ================================================================

template <typename Metavariables>
Main<Metavariables>::Main(CkArgMsg* msg) noexcept
    : options_(Metavariables::help) {
  Informer::print_startup_info(msg);

  /// \todo detail::register_events_to_trace();

  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options;
    command_line_options.add_options()
        ("help,h", "Describe program options")
        ("check-options", "Check input file options")
        ;

    constexpr bool has_options = tmpl::size<option_list>::value > 0;
    // Add input-file option if it makes sense
    make_overloader(
        [&command_line_options](std::true_type /*meta*/, auto mv)
            -> cpp17::void_t<decltype(
                tmpl::type_from<decltype(mv)>::input_file)> {
          // Metavariables has options and default input file name
          command_line_options.add_options()
              ("input-file",
               bpo::value<std::string>()->default_value(
                   tmpl::type_from<decltype(mv)>::input_file),
               "Input file name");
        },
        [&command_line_options](std::true_type /*meta*/, auto /*mv*/,
                                auto... /*unused*/) {
          // Metavariables has options and no default input file name
          command_line_options.add_options()
              ("input-file", bpo::value<std::string>(), "Input file name");
        },
        [](std::false_type /*meta*/, auto mv) -> cpp17::void_t<decltype(
            tmpl::type_from<decltype(mv)>::input_file)> {
          // Metavariables has no options and default input file name

          // always false, but must depend on mv
          static_assert(cpp17::is_same_v<decltype(mv), void>,
                        "Metavariables supplies input file name, "
                        "but there are no options");
          ERROR("This should have failed at compile time");
        },
        [](std::false_type /*meta*/, auto... /*unused*/) {
          // Metavariables has no options and no default input file name
        })(cpp17::bool_constant<has_options>{}, tmpl::type_<Metavariables>{});

    bpo::command_line_parser command_line_parser(msg->argc, msg->argv);
    command_line_parser.options(command_line_options);

    const bool ignore_unrecognized_command_line_options = make_overloader(
        [](auto mv) -> decltype(
            tmpl::type_from<decltype(
                mv)>::ignore_unrecognized_command_line_options) {
          return tmpl::type_from<decltype(
              mv)>::ignore_unrecognized_command_line_options;
        },
        [](auto /*mv*/, auto... /*unused*/) { return false; })(
        tmpl::type_<Metavariables>{});
    if (ignore_unrecognized_command_line_options) {
      // Allow unknown --options
      command_line_parser.allow_unregistered();
    } else {
      // Forbid positional parameters
      command_line_parser.positional({});
    }

    bpo::variables_map parsed_command_line_options;
    bpo::store(command_line_parser.run(), parsed_command_line_options);
    bpo::notify(parsed_command_line_options);

    if (parsed_command_line_options.count("help") != 0) {
      Parallel::printf("%s\n%s", command_line_options, options_.help());
      Parallel::exit();
    }

    std::string input_file;
    if (has_options) {
      if (parsed_command_line_options.count("input-file") == 0) {
        ERROR("No default input file name.  Pass --input-file.");
      }
      input_file = parsed_command_line_options["input-file"].as<std::string>();
      options_.parse_file(input_file);
    } else {
      options_.parse("");
    }

    if (parsed_command_line_options.count("check-options") != 0) {
      // Force all the options to be created.
      options_.template apply<option_list>([](auto... args) {
        (void)std::initializer_list<char>{((void)args, '0')...};
      });
      if (has_options) {
        Parallel::printf("%s parsed successfully!\n", input_file);
      } else {
        // This is still considered successful, since it means the
        // program would have started.
        Parallel::printf("No options to check!\n");
      }
      Parallel::exit();
    }
  } catch (const bpo::error& e) {
    ERROR(e.what());
  }

  const_global_cache_proxy_ =
      options_.template apply<const_global_cache_tags>([](auto... args) {
        return CProxy_ConstGlobalCache<Metavariables>::ckNew(
            tuples::TaggedTupleTypelist<const_global_cache_tags>(
                std::move(args)...));
      });

  tuples::TaggedTupleTypelist<parallel_component_tag_list>
      the_parallel_components;

  // Construct the group proxies with a dependency on the ConstGlobalCache proxy
  using group_component_list = tmpl::filter<
      component_list,
      tmpl::or_<Parallel::is_group_proxy<tmpl::bind<
                    Parallel::proxy_from_parallel_component, tmpl::_1>>,
                Parallel::is_node_group_proxy<tmpl::bind<
                    Parallel::proxy_from_parallel_component, tmpl::_1>>>>;
  CkEntryOptions const_global_cache_dependency;
  const_global_cache_dependency.setGroupDepID(
      const_global_cache_proxy_.ckGetGroupID());

  tmpl::for_each<group_component_list>([
    this, &the_parallel_components, &const_global_cache_dependency
  ](auto parallel_component) noexcept {
    using ParallelComponentProxy = Parallel::proxy_from_parallel_component<
        tmpl::type_from<decltype(parallel_component)>>;
    tuples::get<tmpl::type_<ParallelComponentProxy>>(the_parallel_components) =
        ParallelComponentProxy::ckNew(const_global_cache_proxy_,
                                      &const_global_cache_dependency);
  });

  // Construct the proxies for the single chares
  using singleton_component_list =
      tmpl::filter<component_list,
                   Parallel::is_chare_proxy<tmpl::bind<
                       Parallel::proxy_from_parallel_component, tmpl::_1>>>;
  tmpl::for_each<singleton_component_list>([ this, &the_parallel_components ](
      auto parallel_component) noexcept {
    using ParallelComponentProxy = Parallel::proxy_from_parallel_component<
        tmpl::type_from<decltype(parallel_component)>>;
    tuples::get<tmpl::type_<ParallelComponentProxy>>(the_parallel_components) =
        ParallelComponentProxy::ckNew(const_global_cache_proxy_);
  });

  // Create proxies for empty array chares (which are created by the
  // initialize functions of the parallel_components)
  using array_component_list = tmpl::filter<
      component_list,
      tmpl::and_<Parallel::is_array_proxy<tmpl::bind<
                     Parallel::proxy_from_parallel_component, tmpl::_1>>,
                 tmpl::not_<Parallel::is_bound_array<tmpl::_1>>>>;
  tmpl::for_each<array_component_list>([&the_parallel_components](
      auto parallel_component) noexcept {
    using ParallelComponentProxy = Parallel::proxy_from_parallel_component<
        tmpl::type_from<decltype(parallel_component)>>;
    tuples::get<tmpl::type_<ParallelComponentProxy>>(the_parallel_components) =
        ParallelComponentProxy::ckNew();
  });

  // Create proxies for empty bound array chares
  using bound_array_component_list = tmpl::filter<
      component_list,
      tmpl::and_<Parallel::is_array_proxy<tmpl::bind<
                     Parallel::proxy_from_parallel_component, tmpl::_1>>,
                 Parallel::is_bound_array<tmpl::_1>>>;
  tmpl::for_each<bound_array_component_list>([&the_parallel_components](
      auto parallel_component) noexcept {
    using ParallelComponentProxy = Parallel::proxy_from_parallel_component<
        tmpl::type_from<decltype(parallel_component)>>;
    CkArrayOptions opts;
    opts.bindTo(
        tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
            typename tmpl::type_from<decltype(parallel_component)>::bind_to>>>(
            the_parallel_components));
    tuples::get<tmpl::type_<ParallelComponentProxy>>(the_parallel_components) =
        ParallelComponentProxy::ckNew(opts);
  });

  // Send the complete list of parallel_components to the ConstGlobalCache on
  // each Charm++ node.  After all nodes have finished, the callback is
  // executed.
  CkCallback callback(CkIndex_Main<Metavariables>::initialize(),
                      this->thisProxy);
  const_global_cache_proxy_.set_parallel_components(the_parallel_components,
                                                    callback);
}

template <typename Metavariables>
void Main<Metavariables>::initialize() noexcept {
  tmpl::for_each<component_list>([this](auto parallel_component) noexcept {
    using ParallelComponent = tmpl::type_from<decltype(parallel_component)>;
    options_.template apply<typename ParallelComponent::options>(
        [this](auto... opts) {
          ParallelComponent::initialize(const_global_cache_proxy_,
                                        std::move(opts)...);
        });
  });
  CkStartQD(CkCallback(CkIndex_Main<Metavariables>::execute_next_phase(),
                       this->thisProxy));
}

template <typename Metavariables>
void Main<Metavariables>::execute_next_phase() noexcept {
  current_phase_ = Metavariables::determine_next_phase(
      current_phase_, const_global_cache_proxy_);
  if (Metavariables::Phase::Exit == current_phase_) {
    Informer::print_exit_info();
    Parallel::exit();
  }
  tmpl::for_each<component_list>([this](auto parallel_component) noexcept {
    tmpl::type_from<decltype(parallel_component)>::execute_next_global_actions(
        current_phase_, const_global_cache_proxy_);
  });
  CkStartQD(CkCallback(CkIndex_Main<Metavariables>::execute_next_phase(),
                       this->thisProxy));
}

}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/Main.def.h"
#undef CK_TEMPLATES_ONLY
