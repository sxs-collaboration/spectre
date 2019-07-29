// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/variant/variant.hpp>
#include <converse.h>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <ostream>
#include <pup.h>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "ParallelBackend/AlgorithmMetafunctions.hpp"
#include "ParallelBackend/CharmRegistration.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "ParallelBackend/NodeLock.hpp"
#include "ParallelBackend/ParallelComponentHelpers.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"
#include "ParallelBackend/SimpleActionVisitation.hpp"
#include "ParallelBackend/TypeTraits.hpp"
#include "Utilities/BoostHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_include <array>  // for tuple_size

// IWYU pragma: no_include "ParallelBackend/Algorithm.hpp"  // Include... ourself?

/// \cond
namespace Parallel {
namespace Algorithms {
struct Nodegroup;
struct Singleton;
}  // namespace Algorithms
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Parallel {
/// \cond
template <typename ParallelComponent, typename PhaseDepActionList>
class AlgorithmImpl;
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief A distributed object (Charm++ Chare) that executes a series of Actions
 * and is capable of sending and receiving data. Acts as an interface to
 * Charm++.
 *
 * ### Different Types of Algorithms
 * Charm++ chares can be one of four types, which is specified by the type alias
 * `chare_type` inside the `ParallelComponent`. The four available types of
 * Algorithms are:
 * 1. A Parallel::Algorithms::Singleton where there is only one
 * in the entire execution of the program.
 * 2. A Parallel::Algorithms::Array which holds zero or more
 * elements each of which is a distributed object on some core. An array can
 * grow and shrink in size dynamically if need be and can also be bound to
 * another array. That is, the bound array has the same number of elements as
 * the array it is bound to, and elements with the same ID are on the same core.
 * 3. A Parallel::Algorithms::Group, which is an array but there is
 * one element per core and they are not able to be moved around between cores.
 * These are typically useful for gathering data from array elements on their
 * core, and then processing or reducing it.
 * 4. A Parallel::Algorithms::Nodegroup, which is similar to a
 * group except that there is one element per node. For Charm++ SMP (shared
 * memory parallelism) builds a node corresponds to the usual definition of a
 * node on a supercomputer. However, for non-SMP builds nodes and cores are
 * equivalent. An important difference between groups and nodegroups is that
 * entry methods (remote calls to functions) are not threadsafe on nodegroups.
 * It is up to the person writing the Actions that will be executed on the
 * Nodegroup Algorithm to ensure they are threadsafe.
 *
 * ### What is an Algorithm?
 * An Algorithm is a distributed object, a Charm++ chare, that repeatedly
 * executes a series of Actions. An Action is a struct that has a `static` apply
 * function with signature:
 *
 * \code
 * template <typename... DbTags, typename... InboxTags, typename Metavariables,
 * typename ArrayIndex,  typename ActionList>
 * static auto  apply(db::DataBox<tmpl::list<DbTags...>>& box,
 *                    tuples::TaggedTuple<InboxTags...>& inboxes,
 *                    const ConstGlobalCache<Metavariables>& cache,
 *                    const ArrayIndex& array_index,
 *                    const TemporalId& temporal_id, const ActionList meta);
 * \endcode
 *
 * Note that any of the arguments can be const or non-const references except
 * `array_index`, which must be a `const&`.
 *
 * ### Explicit instantiations of entry methods
 * The code in src/ParallelBackend/CharmMain.tpp registers all entry methods,
 * and if one is not properly registered then a static_assert explains how to
 * have it be registered. If there is a bug in the implementation and an entry
 * method isn't being registered or hitting a static_assert then Charm++ will
 * give an error of the following form:
 *
 * \verbatim
 * registration happened after init Entry point: simple_action(), addr:
 * 0x555a3d0e2090
 * ------------- Processor 0 Exiting: Called CmiAbort ------------
 * Reason: Did you forget to instantiate a templated entry method in a .ci file?
 * \endverbatim
 *
 * If you encounter this issue please file a bug report supplying everything
 * necessary to reproduce the issue.
 */
template <typename ParallelComponent, typename... PhaseDepActionListsPack>
class AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>> {
  static_assert(
      sizeof...(PhaseDepActionListsPack) > 0,
      "Must have at least one phase dependent action list "
      "(PhaseActions) in a parallel component. See the first template "
      "parameter of 'AlgorithmImpl' in the error message to see which "
      "component doesn't have any phase dependent action lists.");

 public:
  /// List of Actions in the order that generates the DataBox types
  using all_actions_list = tmpl::flatten<
      tmpl::list<typename PhaseDepActionListsPack::action_list...>>;
  /// The metavariables class passed to the Algorithm
  using metavariables = typename ParallelComponent::metavariables;
  /// List off all the Tags that can be received into the Inbox
  using inbox_tags_list = Parallel::get_inbox_tags<all_actions_list>;
  /// The type of the object used to identify the element of the array, group
  /// or nodegroup spatially. The default should be an `int`.
  using array_index = typename get_array_index<
      typename ParallelComponent::chare_type>::template f<ParallelComponent>;

  using parallel_component = ParallelComponent;
  /// The type of the Chare
  using chare_type = typename parallel_component::chare_type;
  /// The Charm++ proxy object type
  using cproxy_type =
      typename chare_type::template cproxy<parallel_component, array_index>;
  /// The Charm++ base object type
  using cbase_type =
      typename chare_type::template cbase<parallel_component, array_index>;
  /// The type of the phases
  using PhaseType =
      typename tmpl::front<tmpl::list<PhaseDepActionListsPack...>>::phase_type;

  using phase_dependent_action_lists = tmpl::list<PhaseDepActionListsPack...>;

  /// \cond
  // Needed for serialization
  AlgorithmImpl() noexcept;
  /// \endcond

  /// Constructor used by Main to initialize the algorithm
  template <class... OptionsTags>
  AlgorithmImpl(const Parallel::CProxy_ConstGlobalCache<metavariables>&
                    global_cache_proxy,
                tuples::TaggedTuple<OptionsTags...> options) noexcept;

  /// Charm++ migration constructor, used after a chare is migrated
  constexpr explicit AlgorithmImpl(CkMigrateMessage* /*msg*/) noexcept;

  /// \cond
  ~AlgorithmImpl();

  AlgorithmImpl(const AlgorithmImpl& /*unused*/) = delete;
  AlgorithmImpl& operator=(const AlgorithmImpl& /*unused*/) = delete;
  AlgorithmImpl(AlgorithmImpl&& /*unused*/) = delete;
  AlgorithmImpl& operator=(AlgorithmImpl&& /*unused*/) = delete;
  /// \endcond

  /*!
   * \brief Calls the `apply` function `Action` after a reduction has been
   * completed.
   *
   * The `apply` function must take `arg` as its last argument.
   */
  template <typename Action, typename Arg>
  void reduction_action(Arg arg) noexcept;

  /// \brief Explicitly call the action `Action`. If the returned DataBox type
  /// is not one of the types of the algorithm then a compilation error occurs.
  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args) noexcept;

  template <typename Action>
  void simple_action() noexcept;

  // @{
  /// Call an Action on a local nodegroup requiring the Action to handle thread
  /// safety.
  ///
  /// The `CmiNodelock` of the nodegroup is passed to the Action instead of the
  /// `action_list` as a `const gsl::not_null<CmiNodelock*>&`. The node lock can
  /// be locked with the `Parallel::lock()` function, and unlocked with
  /// `Parallel::unlock()`. `Parallel::try_lock()` is also provided in case
  /// something useful can be done if the lock couldn't be acquired.
  template <typename Action, typename... Args,
            Requires<(sizeof...(Args),
                      cpp17::is_same_v<Parallel::Algorithms::Nodegroup,
                                       chare_type>)> = nullptr>
  void threaded_action(std::tuple<Args...> args) noexcept {
    (void)Parallel::charmxx::RegisterThreadedAction<ParallelComponent, Action,
                                                    Args...>::registrar;
    forward_tuple_to_threaded_action<Action>(
        std::move(args), std::make_index_sequence<sizeof...(Args)>{});
  }

  template <typename Action>
  void threaded_action() noexcept {
    // NOLINTNEXTLINE(modernize-redundant-void-arg)
    (void)Parallel::charmxx::RegisterThreadedAction<ParallelComponent,
                                                    Action>::registrar;
    Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
        box_, *const_global_cache_,
        static_cast<const array_index&>(array_index_),
        make_not_null(&node_lock_));
  }
  // @}

  /// \brief Receive data and store it in the Inbox, and try to continue
  /// executing the algorithm
  ///
  /// When an algorithm has terminated it can be restarted by passing
  /// `enable_if_disabled = true`. This allows long-term disabling and
  /// re-enabling of algorithms
  template <typename ReceiveTag, typename ReceiveDataType>
  void receive_data(typename ReceiveTag::temporal_id instance,
                    ReceiveDataType&& t,
                    bool enable_if_disabled = false) noexcept;

  // @{
  /// Start evaluating the algorithm until the is_ready function of an Action
  /// returns false, or an Action returns with `terminate` set to `true`
  ///
  /// In the case where no phase is passed the current phase is assumed.
  constexpr void perform_algorithm() noexcept;

  constexpr void perform_algorithm(const bool restart_if_terminated) noexcept {
    if (restart_if_terminated) {
      set_terminate(false);
    }
    perform_algorithm();
  }
  // @}

  void start_phase(const PhaseType next_phase) noexcept {
    // terminate should be true since we exited a phase previously.
    if (not get_terminate()) {
      ERROR(
          "An algorithm must always be set to terminate at the beginning of a "
          "phase. Since this is not the case the previous phase did not end "
          "correctly. The integer corresponding to the previous phase is: "
          << static_cast<int>(phase_)
          << " and the next phase is: " << static_cast<int>(next_phase));
    }
    // set terminate to true if there are no actions in this PDAL
    set_terminate(number_of_actions_in_phase(next_phase) == 0);
    phase_ = next_phase;
    algorithm_step_ = 0;
    perform_algorithm();
  }

  /// Tell the Algorithm it should no longer execute the algorithm. This does
  /// not mean that the execution of the program is terminated, but only that
  /// the algorithm has terminated. An algorithm can be restarted by pass `true`
  /// as the second argument to the `receive_data` method or by calling
  /// perform_algorithm(true).
  constexpr void set_terminate(const bool t) noexcept { terminate_ = t; }

  /// Check if an algorithm should continue being evaluated
  constexpr bool get_terminate() const noexcept { return terminate_; }

 private:
  static constexpr bool is_singleton =
      cpp17::is_same_v<chare_type, Parallel::Algorithms::Singleton>;

  template <class Dummy = int,
            Requires<(sizeof(Dummy), is_singleton)> = nullptr>
  constexpr void set_array_index() noexcept {}
  template <class Dummy = int,
            Requires<(sizeof(Dummy), not is_singleton)> = nullptr>
  void set_array_index() noexcept {
    // down cast to the algorithm_type, so that the `thisIndex` method can be
    // called, which is defined in the CBase class
    array_index_ = static_cast<typename chare_type::template algorithm_type<
        ParallelComponent, array_index>&>(*this)
                       .thisIndex;
  }

  template <typename PhaseDepActions, size_t... Is>
  constexpr bool iterate_over_actions(
      std::index_sequence<Is...> /*meta*/) noexcept;

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_action(std::tuple<Args...>&& args,
                               std::index_sequence<Is...> /*meta*/) noexcept {
    Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
        box_, *const_global_cache_,
        static_cast<const array_index&>(array_index_),
        std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_threaded_action(
      std::tuple<Args...>&& args,
      std::index_sequence<Is...> /*meta*/) noexcept {
    const gsl::not_null<CmiNodeLock*> node_lock{&node_lock_};
    Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
        box_, *const_global_cache_,
        static_cast<const array_index&>(array_index_), node_lock,
        std::forward<Args>(std::get<Is>(args))...);
  }

  // @{
  /// Since it's not clear how or if it's possible at all to do SFINAE with
  /// Charm's ci files, we use a forward to implementation, where the
  /// implementation is a simple function call that we can use SFINAE with
  template <typename ReceiveTag, typename ReceiveDataType,
            Requires<tt::is_maplike_v<typename ReceiveTag::type::mapped_type>> =
                nullptr>
  void receive_data_impl(typename ReceiveTag::temporal_id& instance,
                         ReceiveDataType&& t);

  template <
      typename ReceiveTag, typename ReceiveDataType,
      Requires<tt::is_a_v<std::unordered_multiset,
                          typename ReceiveTag::type::mapped_type>> = nullptr>
  constexpr void receive_data_impl(typename ReceiveTag::temporal_id& instance,
                                   ReceiveDataType&& t);
  // @}

  size_t number_of_actions_in_phase(const PhaseType phase) const noexcept {
    size_t number_of_actions = 0;
    const auto helper = [&number_of_actions, phase](auto pdal_v) {
      if (pdal_v.phase == phase) {
        number_of_actions = pdal_v.number_of_actions;
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(PhaseDepActionListsPack{}));
    return number_of_actions;
  }

  // Member variables

#ifdef SPECTRE_CHARM_PROJECTIONS
  double non_action_time_start_;
#endif

  Parallel::ConstGlobalCache<metavariables>* const_global_cache_{nullptr};
  bool performing_action_ = false;
  PhaseType phase_{};
  std::size_t algorithm_step_ = 0;
  tmpl::conditional_t<Parallel::is_node_group_proxy<cproxy_type>::value,
                      CmiNodeLock, NoSuchType>
      node_lock_;

  bool terminate_{true};

  using all_cache_tags = ConstGlobalCache_detail::make_tag_list<metavariables>;
  using initial_databox = db::compute_databox_type<tmpl::flatten<tmpl::list<
      Tags::ConstGlobalCacheImpl<metavariables>,
      typename ParallelComponent::add_options_to_databox::simple_tags,
      db::wrap_tags_in<Tags::FromConstGlobalCache, all_cache_tags>>>>;
  // The types held by the boost::variant, box_
  using databox_phase_types = typename Algorithm_detail::build_databox_types<
      tmpl::list<>, phase_dependent_action_lists, initial_databox,
      inbox_tags_list, metavariables, array_index, ParallelComponent>::type;

  template <typename T>
  struct get_databox_types {
    using type = typename T::databox_types;
  };

  using databox_types = tmpl::flatten<
      tmpl::transform<databox_phase_types, get_databox_types<tmpl::_1>>>;
  // Create a boost::variant that can hold any of the DataBox's
  using variant_boxes = tmpl::remove_duplicates<
      tmpl::push_front<databox_types, db::DataBox<tmpl::list<>>>>;
  make_boost_variant_over<variant_boxes> box_;
  tuples::tagged_tuple_from_typelist<inbox_tags_list> inboxes_{};
  array_index array_index_;
};

////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////

/// \cond
template <typename ParallelComponent, typename... PhaseDepActionListsPack>
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    AlgorithmImpl() noexcept {
  make_overloader([](CmiNodeLock& node_lock) { node_lock = create_lock(); },
                  [](NoSuchType /*unused*/) {})(node_lock_);
  set_array_index();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <class... OptionsTags>
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    AlgorithmImpl(const Parallel::CProxy_ConstGlobalCache<metavariables>&
                      global_cache_proxy,
                  tuples::TaggedTuple<OptionsTags...> options) noexcept
    : AlgorithmImpl() {
  (void)options;  // avoid potential compiler warnings if there are no options.
  const_global_cache_ = global_cache_proxy.ckLocalBranch();
  try {
    // Create the DataBox containing items from the ConstGlobalCache, then pass
    // that to the `add_options_to_databox`'s static `apply` function. If the
    // parallel component does not have a `add_options_to_databox` member struct
    // then the resulting operation just returns the DataBox that is passed in.
    box_ = ParallelComponent::add_options_to_databox::apply(
        db::create<db::AddSimpleTags<Tags::ConstGlobalCacheImpl<metavariables>>,
                   db::AddComputeTags<db::wrap_tags_in<
                       Tags::FromConstGlobalCache, all_cache_tags>>>(
            static_cast<const Parallel::ConstGlobalCache<metavariables>*>(
                const_global_cache_)),
        std::move(get<OptionsTags>(options))...);
  } catch (std::exception& e) {
    ERROR(
        "Could not retrieve the DataBox with only the ConstGlobalCache "
        "tags.");
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
constexpr AlgorithmImpl<ParallelComponent,
                        tmpl::list<PhaseDepActionListsPack...>>::
    AlgorithmImpl(CkMigrateMessage* /*msg*/) noexcept
    : AlgorithmImpl() {}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
AlgorithmImpl<ParallelComponent,
              tmpl::list<PhaseDepActionListsPack...>>::~AlgorithmImpl() {
  // We place the registrar in the destructor since every AlgorithmImpl will
  // have a destructor, but we have different constructors so it's not clear
  // which will be instantiated.
  (void)Parallel::charmxx::RegisterParallelComponent<
      ParallelComponent>::registrar;
  make_overloader([](CmiNodeLock& node_lock) { free_lock(&node_lock); },
                  [](NoSuchType /*unused*/) {})(node_lock_);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename Arg>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    reduction_action(Arg arg) noexcept {
  (void)Parallel::charmxx::RegisterReductionAction<
      ParallelComponent, Action, std::decay_t<Arg>>::registrar;
  lock(&node_lock_);
  if (performing_action_) {
    ERROR(
        "Already performing an Action and cannot execute additional Actions "
        "from inside of an Action. This is only possible if the "
        "reduction_action function is not invoked via a proxy, which makes "
        "no sense for a reduction.");
  }
  performing_action_ = true;
  arg.finalize();
  forward_tuple_to_action<Action>(std::move(arg.data()),
                                  std::make_index_sequence<Arg::pack_size()>{});
  performing_action_ = false;
  unlock(&node_lock_);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename... Args>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    simple_action(std::tuple<Args...> args) noexcept {
  (void)Parallel::charmxx::RegisterSimpleAction<ParallelComponent, Action,
                                                Args...>::registrar;
  lock(&node_lock_);
  if (performing_action_) {
    ERROR(
        "Already performing an Action and cannot execute additional Actions "
        "from inside of an Action. This is only possible if the "
        "simple_action function is not invoked via a proxy, which "
        "we do not allow.");
  }
  performing_action_ = true;
  forward_tuple_to_action<Action>(std::move(args),
                                  std::make_index_sequence<sizeof...(Args)>{});
  performing_action_ = false;
  unlock(&node_lock_);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    simple_action() noexcept {
  (void)Parallel::charmxx::RegisterSimpleAction<ParallelComponent,
                                                Action>::registrar;
  lock(&node_lock_);
  if (performing_action_) {
    ERROR(
        "Already performing an Action and cannot execute additional Actions "
        "from inside of an Action. This is only possible if the "
        "simple_action function is not invoked via a proxy, which "
        "we do not allow.");
  }
  performing_action_ = true;
  Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
      box_, *const_global_cache_,
      static_cast<const array_index&>(array_index_));
  performing_action_ = false;
  unlock(&node_lock_);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ReceiveTag, typename ReceiveDataType>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    receive_data(typename ReceiveTag::temporal_id instance, ReceiveDataType&& t,
                 const bool enable_if_disabled) noexcept {
  (void)Parallel::charmxx::RegisterReceiveData<ParallelComponent,
                                               ReceiveTag>::registrar;
  try {
    lock(&node_lock_);
    if (enable_if_disabled) {
      set_terminate(false);
    }
    receive_data_impl<ReceiveTag>(instance, std::forward<ReceiveDataType>(t));
    unlock(&node_lock_);
  } catch (std::exception& e) {
    ERROR("Fatal error: Unexpected exception caught in receive_data: "
          << e.what());
  }
  perform_algorithm();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
constexpr void AlgorithmImpl<
    ParallelComponent,
    tmpl::list<PhaseDepActionListsPack...>>::perform_algorithm() noexcept {
  if (performing_action_ or get_terminate()) {
    return;
  }
#ifdef SPECTRE_CHARM_PROJECTIONS
  non_action_time_start_ = Parallel::wall_time();
#endif
  lock(&node_lock_);
  const auto invoke_for_phase = [this](auto phase_dep_v) noexcept {
    using PhaseDep = decltype(phase_dep_v);
    constexpr PhaseType phase = PhaseDep::phase;
    using actions_list = typename PhaseDep::action_list;
    if (phase_ == phase) {
      while (tmpl::size<actions_list>::value > 0 and not get_terminate() and
             iterate_over_actions<PhaseDep>(
                 std::make_index_sequence<tmpl::size<actions_list>::value>{})) {
      }
    }
  };
  // Loop over all phases, once the current phase is found we perform the
  // algorithm in that phase until we are no longer able to because we are
  // waiting on data to be sent or because the algorithm has been marked as
  // terminated.
  EXPAND_PACK_LEFT_TO_RIGHT(invoke_for_phase(PhaseDepActionListsPack{}));
  unlock(&node_lock_);
#ifdef SPECTRE_CHARM_PROJECTIONS
  traceUserBracketEvent(SPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID,
                        non_action_time_start_, Parallel::wall_time());
#endif
}
/// \endcond

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename PhaseDepActions, size_t... Is>
constexpr bool
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    iterate_over_actions(const std::index_sequence<Is...> /*meta*/) noexcept {
  bool take_next_action = true;
  const auto helper = [ this, &take_next_action ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    if (not(take_next_action and not terminate_ and algorithm_step_ == iter)) {
      return;
    }
    using actions_list = typename PhaseDepActions::action_list;
    using this_action = tmpl::at_c<actions_list, iter>;
    // Invoke the action's static `apply` method. The overloads are for handling
    // the cases where the `apply` method returns:
    // 1. only a DataBox
    // 2. a DataBox and a bool determining whether or not to terminate
    // 3. a DataBox, a bool, and an integer corresponding to which action in the
    //    current phase's algorithm to execute next.
    //
    // The first argument to the invokable is the DataBox to be passed into the
    // action's `apply` method, while the second is:
    // ```
    // typename std::tuple_size<decltype(this_action::apply(
    //                     box, inboxes_, *const_global_cache_,
    //                     cpp17::as_const(array_index_), actions_list{},
    //                     std::add_pointer_t<ParallelComponent>{}))>::type{}
    // ```
    const auto invoke_this_action = make_overloader(
        [this](auto& my_box,
               std::integral_constant<size_t, 1> /*meta*/) noexcept {
          std::tie(box_) =
              this_action::apply(my_box, inboxes_, *const_global_cache_,
                                 cpp17::as_const(array_index_), actions_list{},
                                 std::add_pointer_t<ParallelComponent>{});
        },
        [this](auto& my_box,
               std::integral_constant<size_t, 2> /*meta*/) noexcept {
          std::tie(box_, terminate_) =
              this_action::apply(my_box, inboxes_, *const_global_cache_,
                                 cpp17::as_const(array_index_), actions_list{},
                                 std::add_pointer_t<ParallelComponent>{});
        },
        [this](auto& my_box,
               std::integral_constant<size_t, 3> /*meta*/) noexcept {
          std::tie(box_, terminate_, algorithm_step_) =
              this_action::apply(my_box, inboxes_, *const_global_cache_,
                                 cpp17::as_const(array_index_), actions_list{},
                                 std::add_pointer_t<ParallelComponent>{});
        });

    // `check_if_ready` calls the `is_ready` static method on the action
    // `action` if it has one, otherwise returns true. The first argument is the
    // ```
    // Algorithm_detail::is_is_ready_callable_t<action, databox,
    //         tuples::tagged_tuple_from_typelist<inbox_tags_list>,
    //         Parallel::ConstGlobalCache<metavariables>, array_index>{}
    // ```
    const auto check_if_ready = make_overloader(
        [this](std::true_type /*has_is_ready*/, auto action,
               const auto& check_local_box) noexcept {
          return decltype(action)::is_ready(
              check_local_box, cpp17::as_const(inboxes_), *const_global_cache_,
              cpp17::as_const(array_index_));
        },
        [](std::false_type /*has_is_ready*/, auto /*action*/,
           const auto& /*box*/) noexcept { return true; });

    constexpr size_t phase_index =
        tmpl::index_of<phase_dependent_action_lists, PhaseDepActions>::value;
    using databox_phase_type = tmpl::at_c<databox_phase_types, phase_index>;
    using databox_types_this_phase = typename databox_phase_type::databox_types;

    const auto display_databox_error = [this](
        const size_t line_number) noexcept {
      ERROR(
          "The DataBox type being retrieved at algorithm step: "
          << algorithm_step_ << " in phase " << phase_index
          << " corresponding to action " << pretty_type::get_name<this_action>()
          << " on line " << line_number
          << " is not the correct type but is of variant index " << box_.which()
          << ". If you are using Goto and Label actions then you are using "
             "them incorrectly.");
    };

    // The overload separately handles the first action in the phase from the
    // remaining actions. The reason for this is that the first action can have
    // as its input DataBox either the output of the last action in the phase or
    // the output of the last action in the *previous* phase. This is handled by
    // checking which DataBox is currently in the `boost::variant` (using the
    // call `box_.which()`).
    make_overloader(
        // clang-format off
        [this, &take_next_action, &check_if_ready, &invoke_this_action,
         &display_databox_error](auto current_iter) noexcept
            -> Requires<std::is_same<std::integral_constant<size_t, 0>,
                                     decltype(current_iter)>::value> {
              // clang-format on
              // When `algorithm_step_ == 0` we could be the first DataBox or
              // the last Databox.
              using first_databox = tmpl::at_c<databox_types_this_phase, 0>;
              using last_databox =
                  tmpl::at_c<databox_types_this_phase,
                             tmpl::size<databox_types_this_phase>::value - 1>;
              using local_this_action =
                  tmpl::at_c<actions_list, decltype(current_iter)::value>;
              if (box_.which() ==
                  static_cast<int>(
                      tmpl::index_of<variant_boxes, first_databox>::value)) {
                using this_databox = first_databox;
                auto& box = boost::get<this_databox>(box_);
                if (not check_if_ready(
                        Algorithm_detail::is_is_ready_callable_t<
                            local_this_action, this_databox,
                            tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                            Parallel::ConstGlobalCache<metavariables>,
                            array_index>{},
                        local_this_action{}, box)) {
                  take_next_action = false;
                  return nullptr;
                }
                performing_action_ = true;
                algorithm_step_++;
                invoke_this_action(
                    box,
                    typename std::tuple_size<decltype(local_this_action::apply(
                        box, inboxes_, *const_global_cache_,
                        cpp17::as_const(array_index_), actions_list{},
                        std::add_pointer_t<ParallelComponent>{}))>::type{});
              } else if (box_.which() ==
                         static_cast<int>(
                             tmpl::index_of<variant_boxes,
                                            last_databox>::value)) {
                using this_databox = last_databox;
                auto& box = boost::get<this_databox>(box_);
                if (not check_if_ready(
                        Algorithm_detail::is_is_ready_callable_t<
                            local_this_action, this_databox,
                            tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                            Parallel::ConstGlobalCache<metavariables>,
                            array_index>{},
                        local_this_action{}, box)) {
                  take_next_action = false;
                  return nullptr;
                }
                performing_action_ = true;
                algorithm_step_++;
                invoke_this_action(
                    box,
                    typename std::tuple_size<decltype(local_this_action::apply(
                        box, inboxes_, *const_global_cache_,
                        cpp17::as_const(array_index_), actions_list{},
                        std::add_pointer_t<ParallelComponent>{}))>::type{});
              } else {
                display_databox_error(__LINE__);
              }
              return nullptr;
            },
        // clang-format off
        [
          this, &take_next_action, &check_if_ready, &invoke_this_action, &
          display_databox_error
        ](auto current_iter) noexcept
            -> Requires<not std::is_same<std::integral_constant<size_t, 0>,
                                         decltype(current_iter)>::value> {
              // clang-format on
              // When `algorithm_step_ != 0` we must be the DataBox of the
              // action before this action.
              using this_databox = tmpl::at_c<databox_types_this_phase,
                                              decltype(current_iter)::value>;
              using local_this_action =
                  tmpl::at_c<actions_list, decltype(current_iter)::value>;
              if (box_.which() ==
                  static_cast<int>(
                      tmpl::index_of<variant_boxes, this_databox>::value)) {
                auto& box = boost::get<this_databox>(box_);
                if (not check_if_ready(
                        Algorithm_detail::is_is_ready_callable_t<
                            local_this_action, this_databox,
                            tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                            Parallel::ConstGlobalCache<metavariables>,
                            array_index>{},
                        local_this_action{}, box)) {
                  take_next_action = false;
                  return nullptr;
                }
                performing_action_ = true;
                algorithm_step_++;
                invoke_this_action(
                    box,
                    typename std::tuple_size<decltype(local_this_action::apply(
                        box, inboxes_, *const_global_cache_,
                        cpp17::as_const(array_index_), actions_list{},
                        std::add_pointer_t<ParallelComponent>{}))>::type{});
              } else {
                display_databox_error(__LINE__);
              }
              return nullptr;
            })(std::integral_constant<size_t, iter>{});
    performing_action_ = false;
    // Wrap counter if necessary
    if (algorithm_step_ >= tmpl::size<actions_list>::value) {
      algorithm_step_ = 0;
    }
  };
  // In case of no Actions avoid compiler warning.
  (void)helper;
  // This is a template for loop for Is
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return take_next_action;
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ReceiveTag, typename ReceiveDataType,
          Requires<tt::is_maplike_v<typename ReceiveTag::type::mapped_type>>>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    receive_data_impl(typename ReceiveTag::temporal_id& instance,
                      ReceiveDataType&& t) {
  static_assert(
      cpp17::is_same_v<
          cpp20::remove_cvref_t<typename ReceiveDataType::first_type>,
          typename ReceiveTag::type::mapped_type::key_type> and
          cpp17::is_same_v<
              cpp20::remove_cvref_t<typename ReceiveDataType::second_type>,
              typename ReceiveTag::type::mapped_type::mapped_type>,
      "The type of the data passed to receive_data for a tag that holds a map "
      "must be a std::pair.");
#ifdef SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID
  double start_time = Parallel::wall_time();
#endif
  auto& inbox = tuples::get<ReceiveTag>(inboxes_)[instance];
  ASSERT(0 == inbox.count(t.first),
         "Receiving data from the 'same' source twice. The message id is: "
             << t.first);
  if (not inbox.insert(std::forward<ReceiveDataType>(t)).second) {
    ERROR("Failed to insert data to receive at instance '"
          << instance << "' with tag '" << pretty_type::get_name<ReceiveTag>()
          << "'.\n");
  }
#ifdef SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID
  traceUserBracketEvent(SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID, start_time,
                        Parallel::wall_time());
#endif
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ReceiveTag, typename ReceiveDataType,
          Requires<tt::is_a_v<std::unordered_multiset,
                              typename ReceiveTag::type::mapped_type>>>
constexpr void
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    receive_data_impl(typename ReceiveTag::temporal_id& instance,
                      ReceiveDataType&& t) {
  tuples::get<ReceiveTag>(inboxes_)[instance].insert(
      std::forward<ReceiveDataType>(t));
}
}  // namespace Parallel
