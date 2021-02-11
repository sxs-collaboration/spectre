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

#include "Informer/Informer.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Formaline.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/System/Exit.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

#include "Parallel/Main.decl.h"

namespace Parallel {
namespace detail {
CREATE_HAS_TYPE_ALIAS(phase_change_tags_and_combines_list)
CREATE_HAS_TYPE_ALIAS_V(phase_change_tags_and_combines_list)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(phase_change_tags_and_combines_list)
CREATE_HAS_TYPE_ALIAS(initialize_phase_change_decision_data)
CREATE_HAS_TYPE_ALIAS_V(initialize_phase_change_decision_data)
}

/// \ingroup ParallelGroup
/// The main function of a Charm++ executable.
/// See [the Parallelization documentation](group__ParallelGroup.html#details)
/// for an overview of Metavariables, Phases, and parallel components.
template <typename Metavariables>
class Main : public CBase_Main<Metavariables> {
 public:
  using component_list = typename Metavariables::component_list;
  using const_global_cache_tags = get_const_global_cache_tags<Metavariables>;
  using mutable_global_cache_tags =
      get_mutable_global_cache_tags<Metavariables>;

  using phase_change_tags_and_combines_list =
      detail::get_phase_change_tags_and_combines_list_or_default_t<
          Metavariables, tmpl::list<>>;
  /// \cond HIDDEN_SYMBOLS
  /// The constructor used to register the class
  explicit Main(
      const Parallel::charmxx::
          MainChareRegistrationConstructor& /*used_for_reg*/) noexcept {}
  ~Main() noexcept override {
    (void)Parallel::charmxx::RegisterChare<
        Main<Metavariables>, CkIndex_Main<Metavariables>>::registrar;
  }
  Main(const Main&) = default;
  Main& operator=(const Main&) = default;
  Main(Main&&) = default;
  Main& operator=(Main&&) = default;
  /// \endcond

  explicit Main(CkArgMsg* msg) noexcept;
  explicit Main(CkMigrateMessage* /*msg*/) {}

  /// Allocate the initial elements of array components, and then execute the
  /// initialization phase on each component
  void allocate_array_components_and_execute_initialization_phase() noexcept;

  /// Determine the next phase of the simulation and execute it.
  void execute_next_phase() noexcept;

  /// Reduction target for data used in phase change decisions.
  ///
  /// It is required that the `Parallel::ReductionData` holds a single
  /// `tuples::TaggedTuple`.
  template <typename InvokeCombine, typename... Tags>
  void phase_change_reduction(
      ReductionData<ReductionDatum<tuples::TaggedTuple<Tags...>, InvokeCombine,
                                   funcl::Identity, std::index_sequence<>>>
          reduction_data) noexcept;

 private:
  template <typename ParallelComponent>
  using parallel_component_options =
      Parallel::get_option_tags<typename ParallelComponent::initialization_tags,
                                Metavariables>;
  using option_list = tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
      Parallel::get_option_tags<const_global_cache_tags, Metavariables>,
      Parallel::get_option_tags<mutable_global_cache_tags, Metavariables>,
      tmpl::transform<component_list,
                      tmpl::bind<parallel_component_options, tmpl::_1>>>>>;
  using parallel_component_tag_list = tmpl::transform<
      component_list,
      tmpl::bind<
          tmpl::type_,
          tmpl::bind<Parallel::proxy_from_parallel_component, tmpl::_1>>>;
  typename Metavariables::Phase current_phase_{
      Metavariables::Phase::Initialization};

  CProxy_MutableGlobalCache<Metavariables> mutable_global_cache_proxy_;
  CProxy_GlobalCache<Metavariables> global_cache_proxy_;
  // This is only used during startup, and will be cleared after all
  // the chares are created.  It is a member variable because passing
  // local state through charm callbacks is painful.
  tuples::tagged_tuple_from_typelist<option_list> options_{};
  // type to be determined by the collection of available phase changers in the
  // Metavariables
  tuples::tagged_tuple_from_typelist<phase_change_tags_and_combines_list>
      phase_change_decision_data_;
};

// ================================================================

template <typename Metavariables>
Main<Metavariables>::Main(CkArgMsg* msg) noexcept {
  Informer::print_startup_info(msg);

  /// \todo detail::register_events_to_trace();

  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options;
    // disable clang-format because it combines the repeated call operator
    // invocations making the code more difficult to parse.
    // clang-format off
    command_line_options.add_options()
        ("help,h", "Describe program options")
        ("check-options", "Check input file options")
        ("dump-source-tree-as", bpo::value<std::string>(),
         "If specified, then a gzip archive of the source tree is dumped "
         "with the specified name. The archive can be expanded using "
         "'tar -xzf ARCHIVE.tar.gz'")
        ("dump-paths",
         "Dump the PATH, CPATH, LD_LIBRARY_PATH, LIBRARY_PATH, and "
         "CMAKE_PREFIX_PATH at compile time.")
        ("dump-environment",
         "Dump the result of printenv at compile time.")
        ("dump-library-versions",
         "Dump the contents of SpECTRE's LibraryVersions.txt")
        ("dump-only",
         "Exit after dumping requested information.")
        ;
    // clang-format on

    constexpr bool has_options = tmpl::size<option_list>::value > 0;
    // Add input-file option if it makes sense
    make_overloader(
        [&command_line_options](std::true_type /*meta*/, auto mv,
                                int /*gcc_bug*/)
            -> std::void_t<decltype(
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
        [](std::false_type /*meta*/, auto mv, int /*gcc_bug*/)
            -> std::void_t<decltype(
                tmpl::type_from<decltype(mv)>::input_file)> {
          // Metavariables has no options and default input file name

          // always false, but must depend on mv
          static_assert(std::is_same_v<decltype(mv), void>,
                        "Metavariables supplies input file name, "
                        "but there are no options");
          ERROR("This should have failed at compile time");
        },
        [](std::false_type /*meta*/, auto... /*unused*/) {
          // Metavariables has no options and no default input file name
        })(std::bool_constant<has_options>{}, tmpl::type_<Metavariables>{}, 0);

    bpo::command_line_parser command_line_parser(msg->argc, msg->argv);
    command_line_parser.options(command_line_options);

    const bool ignore_unrecognized_command_line_options = make_overloader(
        [](auto mv, int /*gcc_bug*/)
            -> decltype(tmpl::type_from<decltype(
                            mv)>::ignore_unrecognized_command_line_options) {
          return tmpl::type_from<decltype(
              mv)>::ignore_unrecognized_command_line_options;
        },
        [](auto /*mv*/, auto... /*meta*/) { return false; })(
        tmpl::type_<Metavariables>{}, 0);
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

    Options::Parser<option_list> options(Metavariables::help);

    if (parsed_command_line_options.count("help") != 0) {
      Parallel::printf("%s\n%s", command_line_options, options.help());
      sys::exit();
    }

    if (parsed_command_line_options.count("dump-source-tree-as") != 0) {
      formaline::write_to_file(
          parsed_command_line_options["dump-source-tree-as"].as<std::string>());
      Parallel::printf("Dumping archive of source tree at link time.\n");
    }
    if (parsed_command_line_options.count("dump-paths") != 0) {
      Parallel::printf("Paths at link time were:\n%s\n",
                       formaline::get_paths());
    }
    if (parsed_command_line_options.count("dump-environment") != 0) {
      Parallel::printf("Environment variables at link time were:\n%s\n",
                       formaline::get_environment_variables());
    }
    if (parsed_command_line_options.count("dump-library-versions") != 0) {
      Parallel::printf("LibraryVersions.txt at link time was:\n%s\n",
                       formaline::get_library_versions());
    }
    if (parsed_command_line_options.count("dump-only") != 0) {
      sys::exit();
    }

    std::string input_file;
    if (has_options) {
      if (parsed_command_line_options.count("input-file") == 0) {
        ERROR("No default input file name.  Pass --input-file.");
      }
      input_file = parsed_command_line_options["input-file"].as<std::string>();
      options.parse_file(input_file);
    } else {
      options.parse("");
    }

    if (parsed_command_line_options.count("check-options") != 0) {
      // Force all the options to be created.
      options.template apply<option_list, Metavariables>([](auto... args) {
        (void)std::initializer_list<char>{((void)args, '0')...};
      });
      if (has_options) {
        Parallel::printf("\n%s parsed successfully!\n", input_file);
      } else {
        // This is still considered successful, since it means the
        // program would have started.
        Parallel::printf("\nNo options to check!\n");
      }
      sys::exit();
    }

    options_ = options.template apply<option_list, Metavariables>(
        [](auto... args) noexcept {
          return tuples::tagged_tuple_from_typelist<option_list>(
              std::move(args)...);
        });
    Parallel::printf("\nOption parsing completed.\n");
  } catch (const bpo::error& e) {
    ERROR(e.what());
  }

  mutable_global_cache_proxy_ = CProxy_MutableGlobalCache<Metavariables>::ckNew(
      Parallel::create_from_options<Metavariables>(
          options_, mutable_global_cache_tags{}));

  // global_cache_proxy_ depends on mutable_global_cache_proxy_.
  CkEntryOptions mutable_global_cache_dependency;
  mutable_global_cache_dependency.setGroupDepID(
      mutable_global_cache_proxy_.ckGetGroupID());

  global_cache_proxy_ = CProxy_GlobalCache<Metavariables>::ckNew(
      Parallel::create_from_options<Metavariables>(options_,
                                                   const_global_cache_tags{}),
      mutable_global_cache_proxy_, this->thisProxy,
      &mutable_global_cache_dependency);

  tuples::tagged_tuple_from_typelist<parallel_component_tag_list>
      the_parallel_components;

  // Construct the group proxies with a dependency on the GlobalCache proxy
  using group_component_list = tmpl::filter<
      component_list,
      tmpl::or_<Parallel::is_group_proxy<tmpl::bind<
                    Parallel::proxy_from_parallel_component, tmpl::_1>>,
                Parallel::is_node_group_proxy<tmpl::bind<
                    Parallel::proxy_from_parallel_component, tmpl::_1>>>>;
  CkEntryOptions global_cache_dependency;
  global_cache_dependency.setGroupDepID(
      global_cache_proxy_.ckGetGroupID());

  tmpl::for_each<group_component_list>([this, &the_parallel_components,
                                        &global_cache_dependency](
                                           auto parallel_component_v) noexcept {
    using parallel_component = tmpl::type_from<decltype(parallel_component_v)>;
    using ParallelComponentProxy =
        Parallel::proxy_from_parallel_component<parallel_component>;
    tuples::get<tmpl::type_<ParallelComponentProxy>>(the_parallel_components) =
        ParallelComponentProxy::ckNew(
            global_cache_proxy_,
            Parallel::create_from_options<Metavariables>(
                options_, typename parallel_component::initialization_tags{}),
            &global_cache_dependency);
  });

  // Construct the proxies for the single chares
  using singleton_component_list =
      tmpl::filter<component_list,
                   Parallel::is_chare_proxy<tmpl::bind<
                       Parallel::proxy_from_parallel_component, tmpl::_1>>>;
  tmpl::for_each<singleton_component_list>(
      [this, &the_parallel_components](auto parallel_component_v) noexcept {
        using parallel_component =
            tmpl::type_from<decltype(parallel_component_v)>;
        using ParallelComponentProxy =
            Parallel::proxy_from_parallel_component<parallel_component>;
        tuples::get<tmpl::type_<ParallelComponentProxy>>(
            the_parallel_components) =
            ParallelComponentProxy::ckNew(
                global_cache_proxy_,
                Parallel::create_from_options<Metavariables>(
                    options_,
                    typename parallel_component::initialization_tags{}));
      });

  // Create proxies for empty array chares (whose elements will be created by
  // the allocate functions of the array components during
  // execute_initialization_phase)
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

  // Send the complete list of parallel_components to the GlobalCache on
  // each Charm++ node.  After all nodes have finished, the callback is
  // executed.
  CkCallback callback(
      CkIndex_Main<Metavariables>::
          allocate_array_components_and_execute_initialization_phase(),
      this->thisProxy);
  global_cache_proxy_.set_parallel_components(the_parallel_components,
                                                    callback);

  if constexpr (detail::has_initialize_phase_change_decision_data_v<
                Metavariables>) {
    Metavariables::initialize_phase_change_decision_data::apply(
        make_not_null(&phase_change_decision_data_), global_cache_proxy_);
  }
}

template <typename Metavariables>
void Main<Metavariables>::
    allocate_array_components_and_execute_initialization_phase() noexcept {
  ASSERT(current_phase_ == Metavariables::Phase::Initialization,
         "Must be in the Initialization phase.");
  using array_component_list =
      tmpl::filter<component_list,
                   Parallel::is_array_proxy<tmpl::bind<
                       Parallel::proxy_from_parallel_component, tmpl::_1>>>;
  tmpl::for_each<array_component_list>([this](
                                           auto parallel_component_v) noexcept {
    using parallel_component = tmpl::type_from<decltype(parallel_component_v)>;
    parallel_component::allocate_array(
        global_cache_proxy_,
        Parallel::create_from_options<Metavariables>(
            options_, typename parallel_component::initialization_tags{}));
  });

  // Free any resources from the initial option parsing.
  options_ = decltype(options_){};

  tmpl::for_each<component_list>([this](auto parallel_component_v) noexcept {
    using parallel_component = tmpl::type_from<decltype(parallel_component_v)>;
    Parallel::get_parallel_component<parallel_component>(
        *(global_cache_proxy_.ckLocalBranch()))
        .start_phase(current_phase_);
  });
  CkStartQD(CkCallback(CkIndex_Main<Metavariables>::execute_next_phase(),
                       this->thisProxy));
}

template <typename Metavariables>
void Main<Metavariables>::execute_next_phase() noexcept {
  current_phase_ = Metavariables::determine_next_phase(
      make_not_null(&phase_change_decision_data_), current_phase_,
      global_cache_proxy_);
  if (Metavariables::Phase::Exit == current_phase_) {
    Informer::print_exit_info();
    sys::exit();
  }
  tmpl::for_each<component_list>([this](auto parallel_component) noexcept {
    tmpl::type_from<decltype(parallel_component)>::execute_next_phase(
        current_phase_, global_cache_proxy_);
  });
  CkStartQD(CkCallback(CkIndex_Main<Metavariables>::execute_next_phase(),
                       this->thisProxy));
}

template <typename Metavariables>
template <typename InvokeCombine, typename... Tags>
void Main<Metavariables>::phase_change_reduction(
    ReductionData<ReductionDatum<tuples::TaggedTuple<Tags...>, InvokeCombine,
                                 funcl::Identity, std::index_sequence<>>>
        reduction_data) noexcept {
  using tagged_tuple_type = std::decay_t<
      std::tuple_element_t<0, std::decay_t<decltype(reduction_data.data())>>>;
  (void)Parallel::charmxx::RegisterPhaseChangeReduction<
      Metavariables, InvokeCombine, Tags...>::registrar;
  static_assert(tt::is_a_v<tuples::TaggedTuple, tagged_tuple_type>,
                "The main chare expects a tagged tuple in the phase change "
                "reduction target.");
  reduction_data.finalize();
  PhaseControl::TaggedTupleMainCombine::apply(
      make_not_null(&phase_change_decision_data_),
      get<0>(reduction_data.data()));
}

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
    const ArrayIndex& array_index) noexcept {
  if constexpr (detail::has_phase_change_tags_and_combines_list_v<
                    Metavariables>) {
    using reduction_data_type = PhaseControl::reduction_data<
        tmpl::list<Ts...>,
        typename Metavariables::phase_change_tags_and_combines_list>;
    (void)Parallel::charmxx::RegisterReducerFunction<
        reduction_data_type::combine>::registrar;
    CkCallback callback(
        CProxy_Main<Metavariables>::index_t::
            template redn_wrapper_phase_change_reduction<
                PhaseControl::TaggedTupleCombine, Ts...>(nullptr),
        cache.get_main_proxy().value());
    reduction_data_type reduction_data{data_for_reduction};
    Parallel::get_parallel_component<SenderComponent>(cache)[array_index]
        .ckLocal()
        ->contribute(static_cast<int>(reduction_data.size()),
                     reduction_data.packed().get(),
                     Parallel::charmxx::charm_reducer_functions.at(
                         std::hash<Parallel::charmxx::ReducerFunctions>{}(
                             &reduction_data_type::combine)),
                     callback);
  } else {
    ERROR(
        "No phase change tags were found; cannot contribute to phase change "
        "reduction.");
  }
}
template <typename SenderComponent, typename Metavariables, class... Ts>
void contribute_to_phase_change_reduction(
    tuples::TaggedTuple<Ts...> data_for_reduction,
    Parallel::GlobalCache<Metavariables>& cache) noexcept {
  if constexpr (detail::has_phase_change_tags_and_combines_list_v<
                    Metavariables>) {
    using reduction_data_type = PhaseControl::reduction_data<
        tmpl::list<Ts...>,
        typename Metavariables::phase_change_tags_and_combines_list>;
    (void)Parallel::charmxx::RegisterReducerFunction<
        reduction_data_type::combine>::registrar;
    CkCallback callback(
        CProxy_Main<Metavariables>::index_t::
            template redn_wrapper_phase_change_reduction<
                PhaseControl::TaggedTupleCombine, Ts...>(nullptr),
        cache.get_main_proxy().value());
    reduction_data_type reduction_data{data_for_reduction};
    Parallel::get_parallel_component<SenderComponent>(cache)
        .ckLocalBranch()
        ->contribute(static_cast<int>(reduction_data.size()),
                     reduction_data.packed().get(),
                     Parallel::charmxx::charm_reducer_functions.at(
                         std::hash<Parallel::charmxx::ReducerFunctions>{}(
                             &reduction_data_type::combine)),
                     callback);

  } else {
    ERROR(
        "No phase change tags were found; cannot contribute to phase change "
        "reduction.");
  }
}
/// @}
}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/Main.def.h"
#undef CK_TEMPLATES_ONLY
