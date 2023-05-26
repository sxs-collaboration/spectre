// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/logical/compl.hpp>
#include <boost/preprocessor/logical/not.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <converse.h>
#include <cstddef>
#include <deque>
#include <exception>
#include <memory>
#include <optional>
#include <pup.h>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace ActionTesting {
/// \cond
struct MockNodeGroupChare;
/// \endcond

namespace detail {

#define ACTION_TESTING_CHECK_MOCK_ACTION_LIST(NAME)                        \
  template <typename Component, typename = std::void_t<>>                  \
  struct get_##NAME##_mocking_list {                                       \
    using replace_these_##NAME = tmpl::list<>;                             \
    using with_these_##NAME = tmpl::list<>;                                \
  };                                                                       \
  template <typename Component>                                            \
  struct get_##NAME##_mocking_list<                                        \
      Component, std::void_t<typename Component::replace_these_##NAME,     \
                             typename Component::with_these_##NAME>> {     \
    using replace_these_##NAME = typename Component::replace_these_##NAME; \
    using with_these_##NAME = typename Component::with_these_##NAME;       \
  };                                                                       \
  template <typename Component>                                            \
  using replace_these_##NAME##_t =                                         \
      typename get_##NAME##_mocking_list<Component>::replace_these_##NAME; \
  template <typename Component>                                            \
  using with_these_##NAME##_t =                                            \
      typename get_##NAME##_mocking_list<Component>::with_these_##NAME

ACTION_TESTING_CHECK_MOCK_ACTION_LIST(simple_actions);
ACTION_TESTING_CHECK_MOCK_ACTION_LIST(threaded_actions);
#undef ACTION_TESTING_CHECK_MOCK_ACTION_LIST

template <typename Component, typename = std::void_t<>>
struct get_simple_tags_from_options_from_component {
  using type = tmpl::list<>;
};

template <typename Component>
struct get_simple_tags_from_options_from_component<
    Component, std::void_t<typename Component::simple_tags_from_options>> {
  using type = typename Component::simple_tags_from_options;
};

// Returns the type of `Tag` (including const and reference-ness as would be
// returned by `db::get<Tag>`) if the tag is in the `DataBox` of type
// `DataBoxType`, otherwise returns `NoSuchType`.
template <typename Tag, typename DataBoxType,
          bool = db::tag_is_retrievable_v<Tag, DataBoxType>>
struct item_type_if_contained;

template <typename Tag, typename DataBoxType>
struct item_type_if_contained<Tag, DataBoxType, true> {
  using type = decltype(db::get<Tag>(DataBoxType{}));
};

template <typename Tag, typename DataBoxType>
struct item_type_if_contained<Tag, DataBoxType, false> {
  using type = NoSuchType;
};

template <typename Tag, typename DataBoxType>
using item_type_if_contained_t =
    typename item_type_if_contained<Tag, DataBoxType>::type;
}  // namespace detail

/// Wraps a size_t representing the node number.  This is so the user
/// can write things like `emplace_array_component(NodeId{3},...)`  instead of
/// `emplace_array_component(3,...)`.
struct NodeId {
  size_t value;
};

inline bool operator==(const NodeId& lhs, const NodeId& rhs) {
  return lhs.value == rhs.value;
}

inline bool operator!=(const NodeId& lhs, const NodeId& rhs) {
  return not(lhs==rhs);
}

/// Wraps a size_t representing the local core number. This is so the
/// user can write things like
/// `emplace_array_component(NodeId{3},LocalCoreId{2},...)`  instead of
/// `emplace_array_component(3,2,...)`.
///
/// The local core number is unique for each core on the same node,
/// but cores on different nodes can have the same local core number.
/// For example, if there are 3 nodes with 2
/// cores each, then the cores on the first node have local core numbers
/// 0 and 1, the cores on the second node also have local core numbers
/// 0 and 1, and so on.
struct LocalCoreId {
  size_t value;
};

inline bool operator==(const LocalCoreId& lhs, const LocalCoreId& rhs) {
  return lhs.value == rhs.value;
}

inline bool operator!=(const LocalCoreId& lhs, const LocalCoreId& rhs) {
  return not(lhs==rhs);
}

/// Wraps a size_t representing the global core number.
///
/// The global core number is unique for each core, even if the cores
/// are on different nodes.  For example, if there are 3 nodes with 2
/// cores each, the global core number goes from 0 through 5 to label
/// each of the 6 cores, and no two cores have the same global core
/// number.
struct GlobalCoreId {
  size_t value;
};

inline bool operator==(const GlobalCoreId& lhs, const GlobalCoreId& rhs) {
  return lhs.value == rhs.value;
}

inline bool operator!=(const GlobalCoreId& lhs, const GlobalCoreId& rhs) {
  return not(lhs==rhs);
}
}  // namespace ActionTesting

namespace std {
template <>
struct hash<ActionTesting::NodeId> {
  size_t operator()(const ActionTesting::NodeId& t) const { return t.value; }
};

template <>
struct hash<ActionTesting::LocalCoreId> {
  size_t operator()(const ActionTesting::LocalCoreId& t) const {
    return t.value;
  }
};

template <>
struct hash<ActionTesting::GlobalCoreId> {
  size_t operator()(const ActionTesting::GlobalCoreId& t) const {
    return t.value;
  }
};
}  // namespace std

namespace ActionTesting {
/// MockDistributedObject mocks the DistributedObject class. It should not be
/// considered as part of the user interface.
///
/// `MockDistributedObject` represents an object on a supercomputer
/// that can have methods invoked on it (possibly) remotely; this is
/// standard nomenclature in the HPC community, based on the idea that
/// such objects get distributed among the cores/nodes on an HPC (even
/// though each object typically lives on only one core).  For
/// example, an element of an array chare in charm++ is a mock
/// distributed object, whereas the entire array chare is a collection
/// of mock distributed objects, each with its own array
/// index.
/// `MockDistributedObject` is a modified implementation of
/// `DistributedObject` and so some of the code is shared between the
/// two. The main difference is that `MockDistributedObject` has
/// support for introspection. For example, it is possible to check
/// how many simple actions are queued, to look at the inboxes,
/// etc. Another key difference is that `MockDistributedObject` runs
/// only one action in the action list at a time. This is done in
/// order to provide opportunity for introspection and checking
/// statements before and after actions are invoked.
template <typename Component>
class MockDistributedObject {
 private:
  class InvokeActionBase {
   public:
    InvokeActionBase() = default;
    InvokeActionBase(const InvokeActionBase&) = default;
    InvokeActionBase& operator=(const InvokeActionBase&) = default;
    InvokeActionBase(InvokeActionBase&&) = default;
    InvokeActionBase& operator=(InvokeActionBase&&) = default;
    virtual ~InvokeActionBase() = default;
    virtual void invoke_action() = 0;
  };

  // Holds the arguments to be passed to the simple action once it is invoked.
  // We delay simple action calls that are made from within an action for
  // several reasons:
  // - This is consistent with what actually happens in the parallel code
  // - This prevents possible stack overflows
  // - Allows better introspection and control over the Actions' behavior
  template <typename Action, typename... Args>
  class InvokeSimpleAction : public InvokeActionBase {
   public:
    InvokeSimpleAction(MockDistributedObject* mock_distributed_object,
                       std::tuple<Args...> args)
        : mock_distributed_object_(mock_distributed_object),
          args_(std::move(args)) {}

    explicit InvokeSimpleAction(MockDistributedObject* mock_distributed_object)
        : mock_distributed_object_(mock_distributed_object) {}

    void invoke_action() override {
      if (not valid_) {
        ERROR(
            "Cannot invoke the exact same simple action twice. This is an "
            "internal bug in the action testing framework. Please file an "
            "issue.");
      }
      valid_ = false;
      invoke_action_impl(std::move(args_));
    }

   private:
    template <typename Arg0, typename... Rest>
    void invoke_action_impl(std::tuple<Arg0, Rest...> args) {
      mock_distributed_object_->simple_action<Action>(std::move(args), true);
    }

    template <typename... LocalArgs,
              Requires<sizeof...(LocalArgs) == 0> = nullptr>
    void invoke_action_impl(std::tuple<LocalArgs...> /*args*/) {
      mock_distributed_object_->simple_action<Action>(true);
    }

    MockDistributedObject* mock_distributed_object_;
    std::tuple<Args...> args_{};
    bool valid_{true};
  };

  // Holds the arguments passed to threaded actions. `InvokeThreadedAction` is
  // analogous to `InvokeSimpleAction`.
  template <typename Action, typename... Args>
  class InvokeThreadedAction : public InvokeActionBase {
   public:
    InvokeThreadedAction(MockDistributedObject* mock_distributed_object,
                         std::tuple<Args...> args)
        : mock_distributed_object_(mock_distributed_object),
          args_(std::move(args)) {}

    explicit InvokeThreadedAction(
        MockDistributedObject* mock_distributed_object)
        : mock_distributed_object_(mock_distributed_object) {}

    void invoke_action() override {
      if (not valid_) {
        ERROR(
            "Cannot invoke the exact same threaded action twice. This is an "
            "internal bug in the action testing framework. Please file an "
            "issue.");
      }
      valid_ = false;
      invoke_action_impl(std::move(args_));
    }

   private:
    template <typename Arg0, typename... Rest>
    void invoke_action_impl(std::tuple<Arg0, Rest...> args) {
      mock_distributed_object_->threaded_action<Action>(std::move(args), true);
    }

    template <typename... LocalArgs,
              Requires<sizeof...(LocalArgs) == 0> = nullptr>
    void invoke_action_impl(std::tuple<LocalArgs...> /*args*/) {
      mock_distributed_object_->threaded_action<Action>(true);
    }

    MockDistributedObject* mock_distributed_object_;
    std::tuple<Args...> args_{};
    bool valid_{true};
  };

 public:
  using phase_dependent_action_lists =
      typename Component::phase_dependent_action_list;
  static_assert(tmpl::size<phase_dependent_action_lists>::value > 0,
                "Must have at least one phase dependent action list "
                "(PhaseActions) in a parallel component.");

  using all_actions_list = tmpl::flatten<tmpl::transform<
      phase_dependent_action_lists,
      Parallel::get_action_list_from_phase_dep_action_list<tmpl::_1>>>;

  using metavariables = typename Component::metavariables;

  using inbox_tags_list =
      Parallel::get_inbox_tags<tmpl::push_front<all_actions_list, Component>>;

  using array_index = typename Parallel::get_array_index<
      typename Component::chare_type>::template f<Component>;

  using parallel_component = Component;

  using all_cache_tags = Parallel::get_const_global_cache_tags<metavariables>;
  using simple_tags_from_options =
      typename detail::get_simple_tags_from_options_from_component<
          Component>::type;
  using initial_tags = tmpl::flatten<tmpl::list<
      Parallel::Tags::MetavariablesImpl<metavariables>,
      Parallel::Tags::ArrayIndexImpl<array_index>,
      Parallel::Tags::GlobalCacheImpl<metavariables>, simple_tags_from_options,
      db::wrap_tags_in<Parallel::Tags::FromGlobalCache, all_cache_tags>,
      Parallel::Algorithm_detail::action_list_simple_tags<Component>,
      Parallel::Algorithm_detail::action_list_compute_tags<Component>>>;
  using databox_type = db::compute_databox_type<initial_tags>;

  MockDistributedObject() = default;

  template <typename... Options>
  MockDistributedObject(
      const NodeId node_id, const LocalCoreId local_core_id,
      std::unordered_map<NodeId, std::unordered_map<LocalCoreId, GlobalCoreId>>
          mock_global_cores,
      std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
          mock_nodes_and_local_cores,
      const array_index& index,
      Parallel::GlobalCache<typename Component::metavariables>* cache,
      tuples::tagged_tuple_from_typelist<inbox_tags_list>* inboxes,
      Options&&... opts)
      : mock_node_(node_id.value),
        mock_local_core_(local_core_id.value),
        mock_global_cores_(std::move(mock_global_cores)),
        mock_nodes_and_local_cores_(std::move(mock_nodes_and_local_cores)),
        array_index_(index),
        global_cache_(cache),
        inboxes_(inboxes) {
    ::Initialization::mutate_assign<
        tmpl::push_front<simple_tags_from_options, Parallel::Tags::ArrayIndex,
                         Parallel::Tags::GlobalCacheImpl<metavariables>>>(
        make_not_null(&box_), array_index_, global_cache_,
        std::forward<Options>(opts)...);
  }

  void set_phase(Parallel::Phase phase) {
    phase_ = phase;
    algorithm_step_ = 0;
    terminate_ = number_of_actions_in_phase(phase) == 0;
    halt_algorithm_until_next_phase_ = false;
  }
  Parallel::Phase get_phase() const { return phase_; }

  void set_terminate(bool t) { terminate_ = t; }
  bool get_terminate() const { return terminate_; }

  // Actions may call this, but since tests step through actions manually it has
  // no effect.
  void perform_algorithm() {}

  size_t number_of_actions_in_phase(const Parallel::Phase phase) const {
    size_t number_of_actions = 0;
    tmpl::for_each<phase_dependent_action_lists>(
        [&number_of_actions, phase](auto pdal_v) {
          const auto pdal = tmpl::type_from<decltype(pdal_v)>{};
          if (pdal.phase == phase) {
            number_of_actions = pdal.number_of_actions;
          }
        });
    return number_of_actions;
  }

  /// @{
  /// Returns the DataBox.
  databox_type& get_databox() { return box_; }

  const databox_type& get_databox() const { return box_; }
  /// @}

  template <typename Tag>
  const auto& get_databox_tag() const {
    return db::get<Tag>(box_);
  }

  /// Returns true if Tag was used to insert an item into the DataBox
  template <typename Tag>
  bool box_contains() const {
    return tmpl::list_contains_v<typename databox_type::tags_list, Tag>;
  }

  /// \brief Returns true if Tag can be retrieved (via db::get) from the DataBox
  ///
  /// \note compute and reference tags can be retrieved via the simple tag they
  /// are derived from.
  template <typename Tag>
  bool tag_is_retrievable() const {
    return db::tag_is_retrievable_v<Tag, databox_type>;
  }

  /// Force the next action invoked to be the `next_action_id`th action in the
  /// current phase.
  void force_next_action_to_be(const size_t next_action_id) {
    algorithm_step_ = next_action_id;
  }

  /// Returns which action (by integer) will be invoked next in the current
  /// phase.
  size_t get_next_action_index() const { return algorithm_step_; }

  /// Invoke the next action in the action list for the current phase,
  /// failing if it was not ready.
  void next_action();

  /// Invoke the next action in the action list for the current phase,
  /// returning whether the action was ready.
  bool next_action_if_ready();

  /// Defines the methods used for invoking threaded and simple actions. Since
  /// the two cases are so similar we use a macro to reduce the amount of
  /// redundant code.
#define SIMPLE_AND_THREADED_ACTIONS(USE_SIMPLE_ACTION, NAME)                  \
  template <typename Action, typename... Args,                                \
            Requires<not tmpl::list_contains_v<                               \
                detail::replace_these_##NAME##s_t<Component>, Action>> =      \
                nullptr>                                                      \
  void NAME(std::tuple<Args...> args,                                         \
            const bool direct_from_action_runner = false) {                   \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      forward_tuple_to_##NAME<Action>(                                        \
          std::move(args), std::make_index_sequence<sizeof...(Args)>{});      \
      performing_action_ = false;                                             \
    } else {                                                                  \
      NAME##_queue_.push_back(                                                \
          std::make_unique<BOOST_PP_IF(USE_SIMPLE_ACTION, InvokeSimpleAction, \
                                       InvokeThreadedAction) < Action,        \
                           Args...> > (this, std::move(args)));               \
    }                                                                         \
  }                                                                           \
  template <typename Action, typename... Args,                                \
            Requires<tmpl::list_contains_v<                                   \
                detail::replace_these_##NAME##s_t<Component>, Action>> =      \
                nullptr>                                                      \
  void NAME(std::tuple<Args...> args,                                         \
            const bool direct_from_action_runner = false) {                   \
    using index_of_action =                                                   \
        tmpl::index_of<detail::replace_these_##NAME##s_t<Component>, Action>; \
    using new_action = tmpl::at_c<detail::with_these_##NAME##s_t<Component>,  \
                                  index_of_action::value>;                    \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      forward_tuple_to_##NAME<new_action>(                                    \
          std::move(args), std::make_index_sequence<sizeof...(Args)>{});      \
      performing_action_ = false;                                             \
    } else {                                                                  \
      NAME##_queue_.push_back(                                                \
          std::make_unique<BOOST_PP_IF(USE_SIMPLE_ACTION, InvokeSimpleAction, \
                                       InvokeThreadedAction) < new_action,    \
                           Args...> > (this, std::move(args)));               \
    }                                                                         \
  }                                                                           \
  template <typename Action,                                                  \
            Requires<not tmpl::list_contains_v<                               \
                detail::replace_these_##NAME##s_t<Component>, Action>> =      \
                nullptr>                                                      \
  void NAME(const bool direct_from_action_runner = false) {                   \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      Action::template apply<Component>(                                      \
          box_, *global_cache_,                                               \
          std::as_const(array_index_)                                         \
              BOOST_PP_COMMA_IF(BOOST_PP_NOT(USE_SIMPLE_ACTION)) BOOST_PP_IF( \
                  USE_SIMPLE_ACTION, , make_not_null(&node_lock_)));          \
      performing_action_ = false;                                             \
    } else {                                                                  \
      NAME##_queue_.push_back(                                                \
          std::make_unique<BOOST_PP_IF(USE_SIMPLE_ACTION, InvokeSimpleAction, \
                                       InvokeThreadedAction) < Action> >      \
          (this));                                                            \
    }                                                                         \
  }                                                                           \
  template <typename Action,                                                  \
            Requires<tmpl::list_contains_v<                                   \
                detail::replace_these_##NAME##s_t<Component>, Action>> =      \
                nullptr>                                                      \
  void NAME(const bool direct_from_action_runner = false) {                   \
    using index_of_action =                                                   \
        tmpl::index_of<detail::replace_these_##NAME##s_t<Component>, Action>; \
    using new_action = tmpl::at_c<detail::with_these_##NAME##s_t<Component>,  \
                                  index_of_action::value>;                    \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      new_action::template apply<Component>(                                  \
          box_, *global_cache_,                                               \
          std::as_const(array_index_)                                         \
              BOOST_PP_COMMA_IF(BOOST_PP_NOT(USE_SIMPLE_ACTION)) BOOST_PP_IF( \
                  USE_SIMPLE_ACTION, , make_not_null(&node_lock_)));          \
      performing_action_ = false;                                             \
    } else {                                                                  \
      simple_action_queue_.push_back(                                         \
          std::make_unique<BOOST_PP_IF(USE_SIMPLE_ACTION, InvokeSimpleAction, \
                                       InvokeThreadedAction) < new_action> >  \
          (this));                                                            \
    }                                                                         \
  }

  SIMPLE_AND_THREADED_ACTIONS(1, simple_action)
  SIMPLE_AND_THREADED_ACTIONS(0, threaded_action)
#undef SIMPLE_AND_THREADED_ACTIONS

  // local synchronous actions are performed as direct function calls
  // regardless, so their 'mocking' implementation is no different from their
  // original implementation
  template <typename Action, typename... Args>
  typename Action::return_type local_synchronous_action(Args&&... args) {
    static_assert(std::is_same_v<typename Component::chare_type,
                                 ActionTesting::MockNodeGroupChare>,
                  "Cannot call a local synchronous action on a chare that is "
                  "not a NodeGroup");
    return Action::template apply<Component>(box_, make_not_null(&node_lock_),
                                             std::forward<Args>(args)...);
  }

  bool is_simple_action_queue_empty() const {
    return simple_action_queue_.empty();
  }

  size_t simple_action_queue_size() const {
    return simple_action_queue_.size();
  }

  void invoke_queued_simple_action() {
    if (simple_action_queue_.empty()) {
      ERROR(
          "There are no queued simple actions to invoke. Are you sure a "
          "previous action invoked a simple action on this component?");
    }
    simple_action_queue_.front()->invoke_action();
    simple_action_queue_.pop_front();
  }

  bool is_threaded_action_queue_empty() const {
    return threaded_action_queue_.empty();
  }

  size_t threaded_action_queue_size() const {
    return threaded_action_queue_.size();
  }

  void invoke_queued_threaded_action() {
    if (threaded_action_queue_.empty()) {
      ERROR(
          "There are no queued threaded actions to invoke. Are you sure a "
          "previous action invoked a threaded action on this component?");
    }
    threaded_action_queue_.front()->invoke_action();
    threaded_action_queue_.pop_front();
  }

  template <typename InboxTag, typename Data>
  void receive_data(const typename InboxTag::temporal_id& id, Data&& data,
                    const bool enable_if_disabled = false) {
    // The variable `enable_if_disabled` might be useful in the future but is
    // not needed now. However, it is required by the interface to be compliant
    // with the Algorithm invocations.
    (void)enable_if_disabled;
    InboxTag::insert_into_inbox(
        make_not_null(&tuples::get<InboxTag>(*inboxes_)), id,
        std::forward<Data>(data));
  }

  template <typename InboxTag, typename MessageType>
  void receive_data(MessageType* message) {
    InboxTag::insert_into_inbox(
        make_not_null(&tuples::get<InboxTag>(*inboxes_)), message);
  }

  Parallel::GlobalCache<typename Component::metavariables>& cache() {
    return *global_cache_;
  }

  /// @{
  /// Wrappers for charm++ informational functions.

  /// Number of processing elements
  int number_of_procs() const;

  /// Global %Index of my processing element.
  int my_proc() const;

  /// Number of nodes.
  int number_of_nodes() const;

  /// %Index of my node.
  int my_node() const;

  /// Number of processing elements on the given node.
  int procs_on_node(int node_index) const;

  /// The local index of my processing element on my node.
  /// This is in the interval 0, ..., procs_on_node(my_node()) - 1.
  int my_local_rank() const;

  /// %Index of first processing element on the given node.
  int first_proc_on_node(int node_index) const;

  /// %Index of the node for the given processing element.
  int node_of(int proc_index) const;

  /// The local index for the given processing element on its node.
  int local_rank_of(int proc_index) const;
  /// @}

 public:
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_simple_action(std::tuple<Args...>&& args,
                                      std::index_sequence<Is...> /*meta*/) {
    Action::template apply<Component>(
        box_, *global_cache_, std::as_const(array_index_),
        std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_threaded_action(std::tuple<Args...>&& args,
                                        std::index_sequence<Is...> /*meta*/) {
    Action::template apply<Component>(
        box_, *global_cache_, std::as_const(array_index_),
        make_not_null(&node_lock_), std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename ThisAction, typename ActionList, typename DbTags>
  bool invoke_iterable_action(db::DataBox<DbTags>& my_box) {
    Parallel::AlgorithmExecution requested_execution{};
    std::optional<std::size_t> next_action_step{};
    std::tie(requested_execution, next_action_step) = ThisAction::apply(
        my_box, *inboxes_, *global_cache_, std::as_const(array_index_),
        ActionList{}, std::add_pointer_t<Component>{});

    if (next_action_step.has_value()) {
      ASSERT(
          Parallel::AlgorithmExecution::Retry != requested_execution,
          "Switching actions on Retry doesn't make sense. Specify std::nullopt "
          "as the second argument of the iterable action return type");
      algorithm_step_ = next_action_step.value();
    }

    switch (requested_execution) {
      case Parallel::AlgorithmExecution::Continue:
        return true;
      case Parallel::AlgorithmExecution::Retry:
        return false;
      case Parallel::AlgorithmExecution::Pause:
        terminate_ = true;
        return true;
      case Parallel::AlgorithmExecution::Halt:
        halt_algorithm_until_next_phase_ = true;
        terminate_ = true;
        return true;
      default:  // LCOV_EXCL_LINE
        // LCOV_EXCL_START
        ERROR("No case for a Parallel::AlgorithmExecution with integral value "
              << static_cast<
                     std::underlying_type_t<Parallel::AlgorithmExecution>>(
                     requested_execution)
              << "\n");
        // LCOV_EXCL_STOP
    }
  }

  template <typename PhaseDepActions, size_t... Is>
  bool next_action_impl(std::index_sequence<Is...> /*meta*/);

  bool terminate_{false};
  bool halt_algorithm_until_next_phase_{false};
  databox_type box_;
  // The next action we should execute.
  size_t algorithm_step_ = 0;
  bool performing_action_ = false;
  Parallel::Phase phase_{Parallel::Phase::Initialization};

  size_t mock_node_{0};
  size_t mock_local_core_{0};
  // mock_global_cores[node][local_core] is the global_core.
  std::unordered_map<NodeId, std::unordered_map<LocalCoreId, GlobalCoreId>>
      mock_global_cores_{};
  // mock_nodes_and_local_cores_[global_core] is the pair node,local_core.
  std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
      mock_nodes_and_local_cores_{};

  typename Component::array_index array_index_{};
  Parallel::GlobalCache<typename Component::metavariables>* global_cache_{
      nullptr};
  tuples::tagged_tuple_from_typelist<inbox_tags_list>* inboxes_{nullptr};
  std::deque<std::unique_ptr<InvokeActionBase>> simple_action_queue_;
  std::deque<std::unique_ptr<InvokeActionBase>> threaded_action_queue_;
  Parallel::NodeLock node_lock_;
};

template <typename Component>
void MockDistributedObject<Component>::pup(PUP::er& p) {
  if (not p.isSizing()) {
    ERROR(
        "MockDistributedObject is not serializable. This pup member can only "
        "be used for sizing.\n");
  }
  p | box_;
}

template <typename Component>
void MockDistributedObject<Component>::next_action() {
  if (not next_action_if_ready()) {
    ERROR("Attempted to run an action, but it returned "
          "AlgorithmExecution::Retry.  Actions that are expected to retry "
          "can be tested with next_action_if_ready().");
  }
}

template <typename Component>
bool MockDistributedObject<Component>::next_action_if_ready() {
  bool found_matching_phase = false;
  bool was_ready = false;
  const auto invoke_for_phase =
      [this, &found_matching_phase, &was_ready](auto phase_dep_v) {
        using PhaseDep = typename decltype(phase_dep_v)::type;
        constexpr Parallel::Phase phase = PhaseDep::phase;
        using actions_list = typename PhaseDep::action_list;
        if (phase_ == phase) {
          found_matching_phase = true;
          was_ready = this->template next_action_impl<PhaseDep>(
              std::make_index_sequence<tmpl::size<actions_list>::value>{});
        }
      };
  tmpl::for_each<phase_dependent_action_lists>(invoke_for_phase);
  if (not found_matching_phase) {
    ERROR("Could not find any actions in the current phase ("
          << phase_ << ") for the component '" << pretty_type::name<Component>()
          << "'.");
  }
  return was_ready;
}

template <typename Component>
template <typename PhaseDepActions, size_t... Is>
bool MockDistributedObject<Component>::next_action_impl(
    std::index_sequence<Is...> /*meta*/) {
  if (UNLIKELY(performing_action_)) {
    ERROR(
        "Cannot call an Action while already calling an Action on the same "
        "MockDistributedObject (an element of a parallel component array, or a "
        "parallel component singleton).");
  }
  if (UNLIKELY(halt_algorithm_until_next_phase_)) {
    ERROR(
        "The algorithm has been halted pending a phase change. No iterable "
        "action can be executed until after the next change of phase.");
  }
  // Keep track of if we already evaluated an action since we want `next_action`
  // to only evaluate one per call.
  bool already_did_an_action = false;
  bool was_ready = true;
  const auto helper = [this, &already_did_an_action,
                       &was_ready](auto iteration) {
    constexpr size_t iter = decltype(iteration)::value;
    if (already_did_an_action or algorithm_step_ != iter) {
      return;
    }

    using actions_list = typename PhaseDepActions::action_list;
    using this_action = tmpl::at_c<actions_list, iter>;

    performing_action_ = true;
    ++algorithm_step_;
    if (not invoke_iterable_action<this_action, actions_list>(box_)) {
      was_ready = false;
      --algorithm_step_;
    }
    performing_action_ = false;
    already_did_an_action = true;
    // Wrap counter if necessary
    if (algorithm_step_ >= tmpl::size<actions_list>::value) {
      algorithm_step_ = 0;
    }
  };
  // Silence compiler warning when there are no Actions.
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return was_ready;
}

template <typename Component>
int MockDistributedObject<Component>::number_of_procs() const {
  return static_cast<int>(mock_nodes_and_local_cores_.size());
}

template <typename Component>
int MockDistributedObject<Component>::my_proc() const {
  return static_cast<int>(mock_global_cores_.at(NodeId{mock_node_})
                              .at(LocalCoreId{mock_local_core_})
                              .value);
}

template <typename Component>
int MockDistributedObject<Component>::number_of_nodes() const {
  return static_cast<int>(mock_global_cores_.size());
}

template <typename Component>
int MockDistributedObject<Component>::my_node() const {
  return static_cast<int>(mock_node_);
}

template <typename Component>
int MockDistributedObject<Component>::procs_on_node(
    const int node_index) const {
  return static_cast<int>(
      mock_global_cores_.at(NodeId{static_cast<size_t>(node_index)}).size());
}

template <typename Component>
int MockDistributedObject<Component>::my_local_rank() const {
  return static_cast<int>(mock_local_core_);
}

template <typename Component>
int MockDistributedObject<Component>::first_proc_on_node(
    const int node_index) const {
  return static_cast<int>(
      mock_global_cores_.at(NodeId{static_cast<size_t>(node_index)})
          .at(LocalCoreId{0})
          .value);
}

template <typename Component>
int MockDistributedObject<Component>::node_of(const int proc_index) const {
  return static_cast<int>(mock_nodes_and_local_cores_
                              .at(GlobalCoreId{static_cast<size_t>(proc_index)})
                              .first.value);
}

template <typename Component>
int MockDistributedObject<Component>::local_rank_of(
    const int proc_index) const {
  return static_cast<int>(mock_nodes_and_local_cores_
                              .at(GlobalCoreId{static_cast<size_t>(proc_index)})
                              .second.value);
}

}  // namespace ActionTesting
