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
#include <tuple>
#include <utility>

#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/SimpleActionVisitation.hpp"
#include "Utilities/BoostHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace ActionTesting {

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
struct get_initialization_tags_from_component {
  using type = tmpl::list<>;
};

template <typename Component>
struct get_initialization_tags_from_component<
    Component, std::void_t<typename Component::initialization_tags>> {
  using type = typename Component::initialization_tags;
};

// Given the tags `SimpleTags`, forwards them into the `DataBox`.
template <typename SimpleTagsList>
struct ForwardAllOptionsToDataBox;

template <typename... SimpleTags>
struct ForwardAllOptionsToDataBox<tmpl::list<SimpleTags...>> {
  using simple_tags = tmpl::list<SimpleTags...>;

  template <typename DbTagsList, typename... Args>
  static auto apply(db::DataBox<DbTagsList>&& box, Args&&... args) noexcept {
    static_assert(
        sizeof...(SimpleTags) == sizeof...(Args),
        "The number of arguments passed to ForwardAllOptionsToDataBox must "
        "match the number of SimpleTags passed.");
    return db::create_from<db::RemoveTags<>, simple_tags>(
        std::move(box), std::forward<Args>(args)...);
  }
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

/// Wraps a size_t representing the local core number. This is so the
/// user can write things like
/// `emplace_array_component(NodeId{3},LocalCoreId{2},...)`  instead of
/// `emplace_array_component(3,2,...)`.
struct LocalCoreId {
  size_t value;
};

/// MockDistributedObject mocks the AlgorithmImpl class. It should not be
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
/// `AlgorithmImpl` and so some of the code is shared between the
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
    virtual void invoke_action() noexcept = 0;
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

    void invoke_action() noexcept override {
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
    void invoke_action_impl(std::tuple<Arg0, Rest...> args) noexcept {
      mock_distributed_object_->simple_action<Action>(std::move(args), true);
    }

    template <typename... LocalArgs,
              Requires<sizeof...(LocalArgs) == 0> = nullptr>
    void invoke_action_impl(std::tuple<LocalArgs...> /*args*/) noexcept {
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

    void invoke_action() noexcept override {
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
    void invoke_action_impl(std::tuple<Arg0, Rest...> args) noexcept {
      mock_distributed_object_->threaded_action<Action>(std::move(args), true);
    }

    template <typename... LocalArgs,
              Requires<sizeof...(LocalArgs) == 0> = nullptr>
    void invoke_action_impl(std::tuple<LocalArgs...> /*args*/) noexcept {
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

  using PhaseType =
      typename tmpl::front<phase_dependent_action_lists>::phase_type;

  using all_cache_tags = Parallel::get_const_global_cache_tags<metavariables>;
  using initialization_tags =
      typename detail::get_initialization_tags_from_component<Component>::type;
  using initial_tags = tmpl::flatten<tmpl::list<
      Parallel::Tags::GlobalCacheImpl<metavariables>, initialization_tags,
      db::wrap_tags_in<Parallel::Tags::FromGlobalCache, all_cache_tags>>>;
  using initial_databox = db::compute_databox_type<initial_tags>;

  // The types held by the boost::variant, box_
  using databox_phase_types =
      typename Parallel::Algorithm_detail::build_databox_types<
          tmpl::list<>, phase_dependent_action_lists, initial_databox,
          inbox_tags_list, metavariables, typename Component::array_index,
          Component>::type;
  template <typename T>
  struct get_databox_types {
    using type = typename T::databox_types;
  };

  using databox_types = tmpl::flatten<
      tmpl::transform<databox_phase_types, get_databox_types<tmpl::_1>>>;
  using variant_boxes = tmpl::remove_duplicates<
      tmpl::push_front<databox_types, db::DataBox<tmpl::list<>>>>;

  MockDistributedObject() = default;

  template <typename... Options>
  MockDistributedObject(
      const NodeId node_id, const LocalCoreId local_core_id,
      std::vector<std::vector<size_t>> mock_global_cores,
      std::vector<std::pair<size_t, size_t>> mock_nodes_and_local_cores,
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
    box_ = detail::ForwardAllOptionsToDataBox<initialization_tags>::apply(
        db::create<
            db::AddSimpleTags<Parallel::Tags::GlobalCacheImpl<metavariables>>,
            db::AddComputeTags<db::wrap_tags_in<Parallel::Tags::FromGlobalCache,
                                                all_cache_tags>>>(
            global_cache_),
        std::forward<Options>(opts)...);
  }

  void set_phase(PhaseType phase) noexcept {
    phase_ = phase;
    algorithm_step_ = 0;
    terminate_ = number_of_actions_in_phase(phase) == 0;
  }
  PhaseType get_phase() const noexcept { return phase_; }

  void set_terminate(bool t) noexcept { terminate_ = t; }
  bool get_terminate() const noexcept { return terminate_; }

  // Actions may call this, but since tests step through actions manually it has
  // no effect.
  void perform_algorithm() noexcept {}

  size_t number_of_actions_in_phase(const PhaseType phase) const noexcept {
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

  // @{
  /// Returns the DataBox with the tags set from the GlobalCache and the
  /// tags in `AdditionalTagsList`. If the DataBox type is incorrect
  /// `std::terminate` is called.
  template <typename AdditionalTagsList>
  auto& get_databox() noexcept {
    using box_type = db::compute_databox_type<
        tmpl::flatten<tmpl::list<initial_tags, AdditionalTagsList>>>;
    return boost::get<box_type>(box_);
  }

  template <typename AdditionalTagsList>
  const auto& get_databox() const noexcept {
    using box_type = db::compute_databox_type<
        tmpl::flatten<tmpl::list<initial_tags, AdditionalTagsList>>>;
    return boost::get<box_type>(box_);
  }
  // @}

  /// Walks through the variant of DataBoxes and retrieves the tag from the
  /// current one, if the current DataBox has the tag. If the current DataBox
  /// does not have the requested tag it is an error.
  template <typename Tag>
  const auto& get_databox_tag() const noexcept {
    return get_databox_tag_visitation<Tag>(box_);
  }

  template <typename Tag>
  bool box_contains() const noexcept {
    return box_contains_visitation<Tag>(box_);
  }

  template <typename Tag>
  bool tag_is_retrievable() const noexcept {
    return tag_is_retrievable_visitation<Tag>(box_);
  }

  // @{
  /// Returns the `boost::variant` of DataBoxes.
  auto& get_variant_box() noexcept { return box_; }

  const auto& get_variant_box() const noexcept { return box_; }
  // @}

  /// Force the next action invoked to be the `next_action_id`th action in the
  /// current phase.
  void force_next_action_to_be(const size_t next_action_id) noexcept {
    algorithm_step_ = next_action_id;
  }

  /// Returns which action (by integer) will be invoked next in the current
  /// phase.
  size_t get_next_action_index() const noexcept { return algorithm_step_; }

  /// Invoke the next action in the action list for the current phase.
  void next_action() noexcept;

  /// Evaluates the `is_ready` method on the next action and returns the result.
  bool is_ready() noexcept;

  /// Defines the methods used for invoking threaded and simple actions. Since
  /// the two cases are so similar we use a macro to reduce the amount of
  /// redundant code.
#define SIMPLE_AND_THREADED_ACTIONS(USE_SIMPLE_ACTION, NAME)                  \
  template <typename Action, typename... Args,                                \
            Requires<not tmpl::list_contains_v<                               \
                detail::replace_these_##NAME##s_t<Component>, Action>> =      \
                nullptr>                                                      \
  void NAME(std::tuple<Args...> args,                                         \
            const bool direct_from_action_runner = false) noexcept {          \
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
            const bool direct_from_action_runner = false) noexcept {          \
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
  void NAME(const bool direct_from_action_runner = false) noexcept {          \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      Parallel::Algorithm_detail::simple_action_visitor<Action, Component>(   \
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
  void NAME(const bool direct_from_action_runner = false) noexcept {          \
    using index_of_action =                                                   \
        tmpl::index_of<detail::replace_these_##NAME##s_t<Component>, Action>; \
    using new_action = tmpl::at_c<detail::with_these_##NAME##s_t<Component>,  \
                                  index_of_action::value>;                    \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      Parallel::Algorithm_detail::simple_action_visitor<new_action,           \
                                                        Component>(           \
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

  bool is_simple_action_queue_empty() const noexcept {
    return simple_action_queue_.empty();
  }

  void invoke_queued_simple_action() noexcept {
    if (simple_action_queue_.empty()) {
      ERROR(
          "There are no queued simple actions to invoke. Are you sure a "
          "previous action invoked a simple action on this component?");
    }
    simple_action_queue_.front()->invoke_action();
    simple_action_queue_.pop_front();
  }

  bool is_threaded_action_queue_empty() const noexcept {
    return threaded_action_queue_.empty();
  }

  void invoke_queued_threaded_action() noexcept {
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

  // {@
  /// Wrappers for charm++ informational functions.

  /// Number of processing elements
  int number_of_procs() const noexcept;

  /// Global %Index of my processing element.
  int my_proc() const noexcept;

  /// Number of nodes.
  int number_of_nodes() const noexcept;

  /// %Index of my node.
  int my_node() const noexcept;

  /// Number of processing elements on the given node.
  int procs_on_node(int node_index) const noexcept;

  /// The local index of my processing element on my node.
  /// This is in the interval 0, ..., procs_on_node(my_node()) - 1.
  int my_local_rank() const noexcept;

  /// %Index of first processing element on the given node.
  int first_proc_on_node(int node_index) const noexcept;

  /// %Index of the node for the given processing element.
  int node_of(int proc_index) const noexcept;

  /// The local index for the given processing element on its node.
  int local_rank_of(int proc_index) const noexcept;
  // @}

 private:
  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_simple_action(
      std::tuple<Args...>&& args,
      std::index_sequence<Is...> /*meta*/) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<Action, Component>(
        box_, *global_cache_, std::as_const(array_index_),
        std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_threaded_action(
      std::tuple<Args...>&& args,
      std::index_sequence<Is...> /*meta*/) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<Action, Component>(
        box_, *global_cache_, std::as_const(array_index_),
        make_not_null(&node_lock_), std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename PhaseDepActions, size_t... Is>
  void next_action_impl(std::index_sequence<Is...> /*meta*/) noexcept;

  template <typename PhaseDepActions, size_t... Is>
  bool is_ready_impl(std::index_sequence<Is...> /*meta*/) noexcept;

  template <typename Tag, typename ThisVariantBox, typename Type,
            typename... Variants,
            Requires<tmpl::size<tmpl::filter<
                         typename ThisVariantBox::tags_list,
                         std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>::value !=
                     0> = nullptr>
  void get_databox_tag_visitation_impl(
      const Type** result, const gsl::not_null<int*> iter,
      const gsl::not_null<bool*> already_visited,
      const boost::variant<Variants...>& box) const noexcept {
    if (box.which() == *iter and not *already_visited) {
      *result = &db::get<Tag>(boost::get<ThisVariantBox>(box));
      (void)result;
      *already_visited = true;
    }
    (*iter)++;
  }
  template <typename Tag, typename ThisVariantBox, typename Type,
            typename... Variants,
            Requires<tmpl::size<tmpl::filter<
                         typename ThisVariantBox::tags_list,
                         std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>::value ==
                     0> = nullptr>
  void get_databox_tag_visitation_impl(
      const Type** /*result*/, const gsl::not_null<int*> iter,
      const gsl::not_null<bool*> already_visited,
      const boost::variant<Variants...>& box) const noexcept {
    if (box.which() == *iter and not *already_visited) {
      ERROR("Cannot retrieve tag: "
            << db::tag_name<Tag>()
            << " from the current DataBox because it is not in it.");
    }
    (*iter)++;
  }

  template <typename Tag, typename... Variants>
  const auto& get_databox_tag_visitation(
      const boost::variant<Variants...>& box) const noexcept {
    using item_types = tmpl::remove_duplicates<tmpl::remove_if<
        tmpl::list<cpp20::remove_cvref_t<
            detail::item_type_if_contained_t<Tag, Variants>>...>,
        std::is_same<NoSuchType, tmpl::_1>>>;
    static_assert(tmpl::size<item_types>::value != 0,
                  "Could not find the tag or the tag as a base tag in any "
                  "DataBox in the get_databox_tag function.");
    static_assert(
        tmpl::size<item_types>::value < 2,
        "Found the tag in or the tag as a base tag in more than one DataBox in "
        "the get_databox_tag function. This means you need to explicitly "
        "retrieve the DataBox type to retrieve the tag or file an issue "
        "requesting a get_databox_tag function that can also take a type "
        "explicitly. We have not yet encountered a need for this functionality "
        "but it could be added.");
    const tmpl::front<item_types>* result = nullptr;
    int iter = 0;
    bool already_visited = false;
    EXPAND_PACK_LEFT_TO_RIGHT(get_databox_tag_visitation_impl<Tag, Variants>(
        &result, &iter, &already_visited, box));
    if (result == nullptr) {
      ERROR("The result pointer is nullptr, which it should never be.\n");
    }
    return *result;
  }

  template <typename Tag, typename ThisVariantBox, typename... Variants,
            Requires<tmpl::list_contains_v<typename ThisVariantBox::tags_list,
                                           Tag>> = nullptr>
  void box_contains_visitation_impl(
      bool* const contains_tag, const gsl::not_null<int*> iter,
      const boost::variant<Variants...>& box) const noexcept {
    if (box.which() == *iter) {
      *contains_tag =
          tmpl::list_contains_v<typename ThisVariantBox::tags_list, Tag>;
    }
    (*iter)++;
  }
  template <typename Tag, typename ThisVariantBox, typename... Variants,
            Requires<not tmpl::list_contains_v<
                typename ThisVariantBox::tags_list, Tag>> = nullptr>
  void box_contains_visitation_impl(
      bool* const /*contains_tag*/, const gsl::not_null<int*> iter,
      const boost::variant<Variants...>& /*box*/) const noexcept {
    (*iter)++;
  }

  template <typename Tag, typename... Variants>
  bool box_contains_visitation(
      const boost::variant<Variants...>& box) const noexcept {
    bool contains_tag = false;
    int iter = 0;
    EXPAND_PACK_LEFT_TO_RIGHT(
        box_contains_visitation_impl<Tag, Variants>(&contains_tag, &iter, box));
    return contains_tag;
  }

  template <typename Tag, typename... Variants>
  bool tag_is_retrievable_visitation(
      const boost::variant<Variants...>& box) const noexcept {
    bool is_retrievable = false;
    const auto helper = [&box, &is_retrievable](auto box_type) noexcept {
      using DataBoxType = typename decltype(box_type)::type;
      if (static_cast<int>(
              tmpl::index_of<tmpl::list<Variants...>, DataBoxType>::value) ==
          box.which()) {
        is_retrievable = db::tag_is_retrievable_v<Tag, DataBoxType>;
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(tmpl::type_<Variants>{}));
    return is_retrievable;
  }

  bool terminate_{false};
  make_boost_variant_over<variant_boxes> box_ = db::DataBox<tmpl::list<>>{};
  // The next action we should execute.
  size_t algorithm_step_ = 0;
  bool performing_action_ = false;
  PhaseType phase_{};

  size_t mock_node_{0};
  size_t mock_local_core_{0};
  // mock_global_cores[node][local_core] is the global_core.
  std::vector<std::vector<size_t>> mock_global_cores_{};
  // mock_nodes_and_local_cores_[global_core] is the pair node,local_core.
  std::vector<std::pair<size_t, size_t>> mock_nodes_and_local_cores_{};

  typename Component::array_index array_index_{};
  Parallel::GlobalCache<typename Component::metavariables>* global_cache_{
      nullptr};
  tuples::tagged_tuple_from_typelist<inbox_tags_list>* inboxes_{nullptr};
  std::deque<std::unique_ptr<InvokeActionBase>> simple_action_queue_;
  std::deque<std::unique_ptr<InvokeActionBase>> threaded_action_queue_;
  Parallel::NodeLock node_lock_;
};

template <typename Component>
void MockDistributedObject<Component>::next_action() noexcept {
  bool found_matching_phase = false;
  const auto invoke_for_phase =
      [this, &found_matching_phase](auto phase_dep_v) noexcept {
        using PhaseDep = typename decltype(phase_dep_v)::type;
        constexpr PhaseType phase = PhaseDep::phase;
        using actions_list = typename PhaseDep::action_list;
        if (phase_ == phase) {
          found_matching_phase = true;
          this->template next_action_impl<PhaseDep>(
              std::make_index_sequence<tmpl::size<actions_list>::value>{});
        }
      };
  tmpl::for_each<phase_dependent_action_lists>(invoke_for_phase);
  if (not found_matching_phase) {
    ERROR("Could not find any actions in the current phase for the component '"
          << pretty_type::short_name<Component>() << "'.");
  }
}

template <typename Component>
template <typename PhaseDepActions, size_t... Is>
void MockDistributedObject<Component>::next_action_impl(
    std::index_sequence<Is...> /*meta*/) noexcept {
  if (UNLIKELY(performing_action_)) {
    ERROR(
        "Cannot call an Action while already calling an Action on the same "
        "MockDistributedObject (an element of a parallel component array, or a "
        "parallel component singleton).");
  }
  // Keep track of if we already evaluated an action since we want `next_action`
  // to only evaluate one per call.
  bool already_did_an_action = false;
  const auto helper = [this, &already_did_an_action](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    if (already_did_an_action or algorithm_step_ != iter) {
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
    //                     box, inboxes_, *global_cache_,
    //                     std::as_const(array_index_), actions_list{},
    //                     std::add_pointer_t<ParallelComponent>{}))>::type{}
    // ```
    const auto invoke_this_action = make_overloader(
        [this](auto& my_box,
               std::integral_constant<size_t, 1> /*meta*/) noexcept {
          std::tie(box_) = this_action::apply(
              my_box, *inboxes_, *global_cache_, std::as_const(array_index_),
              actions_list{}, std::add_pointer_t<Component>{});
        },
        [this](auto& my_box,
               std::integral_constant<size_t, 2> /*meta*/) noexcept {
          std::tie(box_, terminate_) = this_action::apply(
              my_box, *inboxes_, *global_cache_, std::as_const(array_index_),
              actions_list{}, std::add_pointer_t<Component>{});
        },
        [this](auto& my_box,
               std::integral_constant<size_t, 3> /*meta*/) noexcept {
          std::tie(box_, terminate_, algorithm_step_) = this_action::apply(
              my_box, *inboxes_, *global_cache_, std::as_const(array_index_),
              actions_list{}, std::add_pointer_t<Component>{});
        });

    // `check_if_ready` calls the `is_ready` static method on the action
    // `action` if it has one, otherwise returns true. The first argument is the
    // ```
    // Algorithm_detail::is_is_ready_callable_t<action, databox,
    //         tuples::tagged_tuple_from_typelist<inbox_tags_list>,
    //         Parallel::GlobalCache<metavariables>, array_index>{}
    // ```
    const auto check_if_ready = make_overloader(
        [this](std::true_type /*has_is_ready*/, auto action,
               const auto& check_local_box) noexcept {
          return decltype(action)::is_ready(
              check_local_box, std::as_const(*inboxes_), *global_cache_,
              std::as_const(array_index_));
        },
        [](std::false_type /*has_is_ready*/, auto /*action*/,
           const auto& /*box*/) noexcept { return true; });

    constexpr size_t phase_index =
        tmpl::index_of<phase_dependent_action_lists, PhaseDepActions>::value;
    using databox_phase_type = tmpl::at_c<databox_phase_types, phase_index>;
    using databox_types_this_phase = typename databox_phase_type::databox_types;

    const auto display_databox_error = [this]() noexcept {
      ERROR(
          "The DataBox type being retrieved at algorithm step: "
          << algorithm_step_ << " in phase " << phase_index
          << " corresponding to action " << pretty_type::get_name<this_action>()
          << " is not the correct type but is of variant index " << box_.which()
          << ". The type of the current box is: " << type_of_current_state(box_)
          << "\nIf you are using Goto and Label actions then you are using "
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
        [ this, &check_if_ready, &invoke_this_action, &
          display_databox_error ](auto current_iter) noexcept
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
                    Parallel::Algorithm_detail::is_is_ready_callable_t<
                        local_this_action, this_databox,
                        tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                        Parallel::GlobalCache<metavariables>, array_index>{},
                    local_this_action{}, box)) {
              ERROR("Tried to invoke the action '"
                    << pretty_type::get_name<local_this_action>()
                    << "' but have not received all the "
                       "necessary data.");
            }
            performing_action_ = true;
            algorithm_step_++;
            invoke_this_action(
                box,
                typename std::tuple_size<decltype(local_this_action::apply(
                    box, *inboxes_, *global_cache_, std::as_const(array_index_),
                    actions_list{}, std::add_pointer_t<Component>{}))>::type{});
          } else if (box_.which() ==
                     static_cast<int>(
                         tmpl::index_of<variant_boxes, last_databox>::value)) {
            using this_databox = last_databox;
            auto& box = boost::get<this_databox>(box_);
            if (not check_if_ready(
                    Parallel::Algorithm_detail::is_is_ready_callable_t<
                        local_this_action, this_databox,
                        tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                        Parallel::GlobalCache<metavariables>, array_index>{},
                    local_this_action{}, box)) {
              ERROR("Tried to invoke the action '"
                    << pretty_type::get_name<local_this_action>()
                    << "' but have not received all the "
                       "necessary data.");
            }
            performing_action_ = true;
            algorithm_step_++;
            invoke_this_action(
                box,
                typename std::tuple_size<decltype(local_this_action::apply(
                    box, *inboxes_, *global_cache_, std::as_const(array_index_),
                    actions_list{}, std::add_pointer_t<Component>{}))>::type{});
          } else {
            display_databox_error();
          }
          return nullptr;
        },
        // clang-format off
        [ this, &check_if_ready, &invoke_this_action, &
          display_databox_error ](auto current_iter) noexcept
            -> Requires<not std::is_same<std::integral_constant<size_t, 0>,
                                         decltype(current_iter)>::value> {
          // clang-format on
          // When `algorithm_step_ != 0` we must be the DataBox of before us
          using this_databox = tmpl::at_c<databox_types_this_phase,
                                          decltype(current_iter)::value>;
          using local_this_action =
              tmpl::at_c<actions_list, decltype(current_iter)::value>;
          if (box_.which() ==
              static_cast<int>(
                  tmpl::index_of<variant_boxes, this_databox>::value)) {
            auto& box = boost::get<this_databox>(box_);
            if (not check_if_ready(
                    Parallel::Algorithm_detail::is_is_ready_callable_t<
                        local_this_action, this_databox,
                        tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                        Parallel::GlobalCache<metavariables>, array_index>{},
                    local_this_action{}, box)) {
              ERROR("Tried to invoke the action '"
                    << pretty_type::get_name<local_this_action>()
                    << "' but have not received all the "
                       "necessary data.");
            }
            performing_action_ = true;
            algorithm_step_++;
            invoke_this_action(
                box,
                typename std::tuple_size<decltype(local_this_action::apply(
                    box, *inboxes_, *global_cache_, std::as_const(array_index_),
                    actions_list{}, std::add_pointer_t<Component>{}))>::type{});
          } else {
            display_databox_error();
          }
          return nullptr;
        })(std::integral_constant<size_t, iter>{});

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
}

template <typename Component>
bool MockDistributedObject<Component>::is_ready() noexcept {
  bool action_is_ready = false;
  bool found_matching_phase = false;
  const auto invoke_for_phase = [this, &action_is_ready, &found_matching_phase](
                                    auto phase_dep_v) noexcept {
    using PhaseDep = typename decltype(phase_dep_v)::type;
    constexpr PhaseType phase = PhaseDep::phase;
    using actions_list = typename PhaseDep::action_list;
    if (phase_ == phase) {
      found_matching_phase = true;
      action_is_ready = this->template is_ready_impl<PhaseDep>(
          std::make_index_sequence<tmpl::size<actions_list>::value>{});
    }
  };
  tmpl::for_each<phase_dependent_action_lists>(invoke_for_phase);
  if (not found_matching_phase) {
    ERROR("Could not find any actions in the current phase for the component '"
          << pretty_type::short_name<Component>() << "'.");
  }
  return action_is_ready;
}

template <typename Component>
template <typename PhaseDepActions, size_t... Is>
bool MockDistributedObject<Component>::is_ready_impl(
    std::index_sequence<Is...> /*meta*/) noexcept {
  bool next_action_is_ready = false;
  const auto helper = [this, &array_index = array_index_, &inboxes = *inboxes_,
                       &global_cache = global_cache_,
                       &next_action_is_ready](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    using actions_list = typename PhaseDepActions::action_list;
    using this_action = tmpl::at_c<actions_list, iter>;

    constexpr size_t phase_index =
        tmpl::index_of<phase_dependent_action_lists, PhaseDepActions>::value;
    using databox_phase_type = tmpl::at_c<databox_phase_types, phase_index>;
    using databox_types_this_phase = typename databox_phase_type::databox_types;
    using this_databox =
        tmpl::at_c<databox_types_this_phase,
                   iter == 0 ? tmpl::size<databox_types_this_phase>::value - 1
                             : iter>;
    if (iter != algorithm_step_) {
      return;
    }

    this_databox* box_ptr{};
    try {
      box_ptr = &boost::get<this_databox>(box_);
    } catch (std::exception& e) {
      ERROR(
          "\nFailed to retrieve Databox in take_next_action:\nCaught "
          "exception: '"
          << e.what() << "'\nDataBox type: '"
          << pretty_type::get_name<this_databox>() << "'\nIteration: " << iter
          << "\nAction: '" << pretty_type::get_name<this_action>()
          << "'\nBoost::Variant id: " << box_.which()
          << "\nBoost::Variant type is: '" << type_of_current_state(box_)
          << "'\n\n");
    }
    this_databox& box = *box_ptr;

    // `check_if_ready` calls the `is_ready` static method on the action
    // `action` if it has one, otherwise returns true. The first argument is the
    // ```
    // Algorithm_detail::is_is_ready_callable_t<action, databox,
    //         tuples::tagged_tuple_from_typelist<inbox_tags_list>,
    //         Parallel::GlobalCache<metavariables>, array_index>{}
    // ```
    const auto check_if_ready = make_overloader(
        [&box, &array_index, &global_cache, &inboxes](
            std::true_type /*has_is_ready*/, auto t) {
          return decltype(t)::is_ready(std::as_const(box),
                                       std::as_const(inboxes), *global_cache,
                                       std::as_const(array_index));
        },
        [](std::false_type /*has_is_ready*/, auto /*meta*/) { return true; });

    next_action_is_ready =
        check_if_ready(Parallel::Algorithm_detail::is_is_ready_callable_t<
                           this_action, this_databox,
                           tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                           Parallel::GlobalCache<metavariables>,
                           typename Component::array_index>{},
                       this_action{});
  };
  // Silence compiler warning when there are no Actions.
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return next_action_is_ready;
}

template <typename Component>
int MockDistributedObject<Component>::number_of_procs() const noexcept {
  return static_cast<int>(mock_nodes_and_local_cores_.size());
}

template <typename Component>
int MockDistributedObject<Component>::my_proc() const noexcept {
  return static_cast<int>(
      mock_global_cores_.at(mock_node_).at(mock_local_core_));
}

template <typename Component>
int MockDistributedObject<Component>::number_of_nodes() const noexcept {
  return static_cast<int>(mock_global_cores_.size());
}

template <typename Component>
int MockDistributedObject<Component>::my_node() const noexcept {
  return static_cast<int>(mock_node_);
}

template <typename Component>
int MockDistributedObject<Component>::procs_on_node(
    const int node_index) const noexcept {
  return static_cast<int>(
      mock_global_cores_.at(static_cast<size_t>(node_index)).size());
}

template <typename Component>
int MockDistributedObject<Component>::my_local_rank() const noexcept {
  return static_cast<int>(mock_local_core_);
}

template <typename Component>
int MockDistributedObject<Component>::first_proc_on_node(
    const int node_index) const noexcept {
  return static_cast<int>(
      mock_global_cores_.at(static_cast<size_t>(node_index)).front());
}

template <typename Component>
int MockDistributedObject<Component>::node_of(
    const int proc_index) const noexcept {
  return static_cast<int>(
      mock_nodes_and_local_cores_.at(static_cast<size_t>(proc_index)).first);
}

template <typename Component>
int MockDistributedObject<Component>::local_rank_of(
    const int proc_index) const noexcept {
  return static_cast<int>(
      mock_nodes_and_local_cores_.at(static_cast<size_t>(proc_index)).second);
}

}  // namespace ActionTesting
