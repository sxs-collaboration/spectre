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
#include <ostream>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/SimpleActionVisitation.hpp"
#include "Utilities/BoostHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <class ChareType>
struct get_array_index;
}  // namespace Parallel
/// \endcond

namespace ActionTesting {
namespace detail {
#define ACTION_TESTING_CHECK_MOCK_ACTION_LIST(NAME)                        \
  template <typename Component, typename = cpp17::void_t<>>                \
  struct get_##NAME##_mocking_list {                                       \
    using replace_these_##NAME = tmpl::list<>;                             \
    using with_these_##NAME = tmpl::list<>;                                \
  };                                                                       \
  template <typename Component>                                            \
  struct get_##NAME##_mocking_list<                                        \
      Component, cpp17::void_t<typename Component::replace_these_##NAME,   \
                               typename Component::with_these_##NAME>> {   \
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
}  // namespace detail

// MockDistributedObject mocks the AlgorithmImpl class.
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
    InvokeSimpleAction(MockDistributedObject* local_alg,
                       std::tuple<Args...> args)
        : local_algorithm_(local_alg), args_(std::move(args)) {}

    explicit InvokeSimpleAction(MockDistributedObject* local_alg)
        : local_algorithm_(local_alg) {}

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
      local_algorithm_->simple_action<Action>(std::move(args), true);
    }

    template <typename... LocalArgs,
              Requires<sizeof...(LocalArgs) == 0> = nullptr>
    void invoke_action_impl(std::tuple<LocalArgs...> /*args*/) noexcept {
      local_algorithm_->simple_action<Action>(true);
    }

    MockDistributedObject* local_algorithm_;
    std::tuple<Args...> args_{};
    bool valid_{true};
  };

  template <typename Action, typename... Args>
  class InvokeThreadedAction : public InvokeActionBase {
   public:
    InvokeThreadedAction(MockDistributedObject* local_alg,
                         std::tuple<Args...> args)
        : local_algorithm_(local_alg), args_(std::move(args)) {}

    explicit InvokeThreadedAction(MockDistributedObject* local_alg)
        : local_algorithm_(local_alg) {}

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
      local_algorithm_->threaded_action<Action>(std::move(args), true);
    }

    template <typename... LocalArgs,
              Requires<sizeof...(LocalArgs) == 0> = nullptr>
    void invoke_action_impl(std::tuple<LocalArgs...> /*args*/) noexcept {
      local_algorithm_->threaded_action<Action>(true);
    }

    MockDistributedObject* local_algorithm_;
    std::tuple<Args...> args_{};
    bool valid_{true};
  };

 public:
  using actions_list = typename Component::action_list;

  using inbox_tags_list = Parallel::get_inbox_tags<actions_list>;

  using metavariables = typename Component::metavariables;

 private:
  template <typename ActionsList>
  struct compute_databox_type;

  template <typename... ActionsPack>
  struct compute_databox_type<tmpl::list<ActionsPack...>> {
    using type = Parallel::Algorithm_detail::build_action_return_typelist<
        typename Component::initial_databox,
        tmpl::list<
            tuples::tagged_tuple_from_typelist<inbox_tags_list>,
            Parallel::ConstGlobalCache<typename Component::metavariables>,
            typename Component::array_index, actions_list,
            std::add_pointer_t<Component>>,
        ActionsPack...>;
  };

 public:
  using databox_types = typename compute_databox_type<actions_list>::type;

  MockDistributedObject() = default;

  explicit MockDistributedObject(
      typename Component::initial_databox initial_box)
      : box_(std::move(initial_box)) {}

  void set_index(typename Component::array_index index) noexcept {
    array_index_ = std::move(index);
  }

  void set_cache(Parallel::ConstGlobalCache<typename Component::metavariables>*
                     cache_ptr) noexcept {
    const_global_cache_ = cache_ptr;
  }

  void set_inboxes(tuples::tagged_tuple_from_typelist<
                   Parallel::get_inbox_tags<typename Component::action_list>>*
                       inboxes_ptr) noexcept {
    inboxes_ = inboxes_ptr;
  }

  void set_terminate(bool t) { terminate_ = t; }
  bool get_terminate() { return terminate_; }

  // Actions may call this, but since tests step through actions manually it has
  // no effect.
  void perform_algorithm() noexcept {}

  template <typename BoxType>
  BoxType& get_databox() noexcept {
    return boost::get<BoxType>(box_);
  }

  template <typename BoxType>
  const BoxType& get_databox() const noexcept {
    return boost::get<BoxType>(box_);
  }

  auto& get_variant_box() noexcept { return box_; }

  const auto& get_variant_box() const noexcept { return box_; }

  void force_next_action_to_be(const size_t next_action_id) noexcept {
    algorithm_step_ = next_action_id;
  }

  size_t get_next_action_index() const noexcept { return algorithm_step_; }

  template <size_t... Is>
  void next_action(std::index_sequence<Is...> /*meta*/) noexcept;

  template <size_t... Is>
  bool is_ready(std::index_sequence<Is...> /*meta*/) noexcept;

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
                           Args...>> (this, std::move(args)));                \
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
                           Args...>> (this, std::move(args)));                \
    }                                                                         \
  }                                                                           \
  template <typename Action,                                                  \
            Requires<not tmpl::list_contains_v<                               \
                detail::replace_these_##NAME##s_t<Component>, Action>> =      \
                nullptr>                                                      \
  void NAME(const bool direct_from_action_runner = false) noexcept {          \
    if (direct_from_action_runner) {                                          \
      performing_action_ = true;                                              \
      Parallel::Algorithm_detail::simple_action_visitor<                      \
          Action, typename Component::initial_databox>(                       \
          box_, *inboxes_, *const_global_cache_,                              \
          cpp17::as_const(array_index_), actions_list{},                      \
          std::add_pointer_t<Component>{nullptr});                            \
      performing_action_ = false;                                             \
    } else {                                                                  \
      NAME##_queue_.push_back(                                                \
          std::make_unique<BOOST_PP_IF(USE_SIMPLE_ACTION, InvokeSimpleAction, \
                                       InvokeThreadedAction) < Action>>       \
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
      Parallel::Algorithm_detail::simple_action_visitor<                      \
          new_action, typename Component::initial_databox>(                   \
          box_, *inboxes_, *const_global_cache_,                              \
          cpp17::as_const(array_index_), actions_list{},                      \
          std::add_pointer_t<Component>{nullptr} BOOST_PP_COMMA_IF(           \
              BOOST_PP_NOT(USE_SIMPLE_ACTION))                                \
              BOOST_PP_IF(USE_SIMPLE_ACTION, , make_not_null(&node_lock_)));  \
      performing_action_ = false;                                             \
    } else {                                                                  \
      simple_action_queue_.push_back(                                         \
          std::make_unique<BOOST_PP_IF(USE_SIMPLE_ACTION, InvokeSimpleAction, \
                                       InvokeThreadedAction) < new_action>>   \
          (this));                                                            \
    }                                                                         \
  }

  SIMPLE_AND_THREADED_ACTIONS(1, simple_action)
  SIMPLE_AND_THREADED_ACTIONS(0, threaded_action)
#undef SIMPLE_AND_THREADED_ACTIONS

  bool is_simple_action_queue_empty() noexcept {
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

  bool is_threaded_action_queue_empty() noexcept {
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
  void receive_data(const typename InboxTag::temporal_id& id, const Data& data,
                    const bool enable_if_disabled = false) {
    // The variable `enable_if_disabled` might be useful in the future but is
    // not needed now. However, it is required by the interface to be compliant
    // with the Algorithm invocations.
    (void)enable_if_disabled;
    tuples::get<InboxTag>(*inboxes_)[id].emplace(data);
  }

 private:
  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_simple_action(
      std::tuple<Args...>&& args,
      std::index_sequence<Is...> /*meta*/) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<
        Action, typename Component::initial_databox>(
        box_, *inboxes_, *const_global_cache_, cpp17::as_const(array_index_),
        actions_list{}, std::add_pointer_t<Component>{nullptr},
        std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_threaded_action(
      std::tuple<Args...>&& args,
      std::index_sequence<Is...> /*meta*/) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<
        Action, typename Component::initial_databox>(
        box_, *inboxes_, *const_global_cache_, cpp17::as_const(array_index_),
        actions_list{}, std::add_pointer_t<Component>{nullptr},
        make_not_null(&node_lock_), std::forward<Args>(std::get<Is>(args))...);
  }

  bool terminate_{false};
  make_boost_variant_over<
      tmpl::push_front<databox_types, db::DataBox<tmpl::list<>>>>
      box_ = db::DataBox<tmpl::list<>>{};
  // The next action we should execute.
  size_t algorithm_step_ = 0;
  bool performing_action_ = false;

  typename Component::array_index array_index_{};
  Parallel::ConstGlobalCache<typename Component::metavariables>*
      const_global_cache_{nullptr};
  tuples::tagged_tuple_from_typelist<
      Parallel::get_inbox_tags<typename Component::action_list>>* inboxes_{
      nullptr};
  std::deque<std::unique_ptr<InvokeActionBase>> simple_action_queue_;
  std::deque<std::unique_ptr<InvokeActionBase>> threaded_action_queue_;
  CmiNodeLock node_lock_ = Parallel::create_lock();
};

template <typename Component>
template <size_t... Is>
void MockDistributedObject<Component>::next_action(
    std::index_sequence<Is...> /*meta*/) noexcept {
  auto& const_global_cache = *const_global_cache_;
  if (UNLIKELY(performing_action_)) {
    ERROR(
        "Cannot call an Action while already calling an Action on the same "
        "MockDistributedObject (an element of a parallel component array, or a "
        "parallel component singleton).");
  }
  // Keep track of if we already evaluated an action since we want `next_action`
  // to only evaluate one per call.
  bool already_did_an_action = false;
  const auto helper = [
    this, &array_index = array_index_, &inboxes = *inboxes_,
    &const_global_cache, &already_did_an_action
  ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    using this_action = tmpl::at_c<actions_list, iter>;
    using this_databox =
        tmpl::at_c<databox_types,
                   iter == 0 ? tmpl::size<databox_types>::value - 1 : iter>;
    if (already_did_an_action or algorithm_step_ != iter) {
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

    const auto check_if_ready = make_overloader(
        [&box, &array_index, &const_global_cache, &inboxes](
            std::true_type /*has_is_ready*/, auto t) {
          return decltype(t)::is_ready(
              cpp17::as_const(box), cpp17::as_const(inboxes),
              const_global_cache, cpp17::as_const(array_index));
        },
        [](std::false_type /*has_is_ready*/, auto) { return true; });
    if (not check_if_ready(
            Parallel::Algorithm_detail::is_is_ready_callable_t<
                this_action, this_databox,
                tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                Parallel::ConstGlobalCache<metavariables>,
                typename Component::array_index>{},
            this_action{})) {
      ERROR("Tried to invoke the action '"
            << pretty_type::get_name<this_action>()
            << "' but have not received all the "
               "necessary data.");
    }
    performing_action_ = true;
    algorithm_step_++;
    constexpr Component const* const component_ptr = nullptr;
    make_overloader(
        [ this, &array_index, component_ptr, &const_global_cache, &inboxes ](
            auto& my_box, std::integral_constant<size_t, 1> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_) = this_action::apply(
                  my_box, inboxes, const_global_cache,
                  cpp17::as_const(array_index), actions_list{}, component_ptr);
            },
        [ this, &array_index, component_ptr, &const_global_cache, &inboxes ](
            auto& my_box, std::integral_constant<size_t, 2> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_, terminate_) = this_action::apply(
                  my_box, inboxes, const_global_cache,
                  cpp17::as_const(array_index), actions_list{}, component_ptr);
            },
        [ this, &array_index, component_ptr, &const_global_cache, &inboxes ](
            auto& my_box, std::integral_constant<size_t, 3> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_, terminate_, algorithm_step_) = this_action::apply(
                  my_box, inboxes, const_global_cache,
                  cpp17::as_const(array_index), actions_list{}, component_ptr);
            })(
        box, typename std::tuple_size<decltype(this_action::apply(
                 box, inboxes, const_global_cache, cpp17::as_const(array_index),
                 actions_list{}, component_ptr))>::type{});
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
template <size_t... Is>
bool MockDistributedObject<Component>::is_ready(
    std::index_sequence<Is...> /*meta*/) noexcept {
  bool next_action_is_ready = false;
  const auto helper = [
    this, &array_index = array_index_, &inboxes = *inboxes_,
    &const_global_cache = const_global_cache_, &next_action_is_ready
  ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    using this_action = tmpl::at_c<actions_list, iter>;
    using this_databox =
        tmpl::at_c<databox_types,
                   iter == 0 ? tmpl::size<databox_types>::value - 1 : iter>;
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

    const auto check_if_ready = make_overloader(
        [&box, &array_index, &const_global_cache, &inboxes](
            std::true_type /*has_is_ready*/, auto t) {
          return decltype(t)::is_ready(
              cpp17::as_const(box), cpp17::as_const(inboxes),
              *const_global_cache, cpp17::as_const(array_index));
        },
        [](std::false_type /*has_is_ready*/, auto) { return true; });

    next_action_is_ready =
        check_if_ready(Parallel::Algorithm_detail::is_is_ready_callable_t<
                           this_action, this_databox,
                           tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                           Parallel::ConstGlobalCache<metavariables>,
                           typename Component::array_index>{},
                       this_action{});
  };
  // Silence compiler warning when there are no Actions.
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return next_action_is_ready;
}

namespace ActionTesting_detail {
// A mock class for the Charm++ generated CProxyElement_AlgorithmArray (we use
// an array for everything, so no need to mock groups, nodegroups, singletons).
template <typename Component, typename InboxTagList>
class MockArrayElementProxy {
 public:
  using Inbox = tuples::tagged_tuple_from_typelist<InboxTagList>;

  MockArrayElementProxy(MockDistributedObject<Component>& local_algorithm,
                        Inbox& inbox)
      : local_algorithm_(local_algorithm), inbox_(inbox) {}

  template <typename InboxTag, typename Data>
  void receive_data(const typename InboxTag::temporal_id& id, const Data& data,
                    const bool enable_if_disabled = false) {
    // The variable `enable_if_disabled` might be useful in the future but is
    // not needed now. However, it is required by the interface to be compliant
    // with the Algorithm invocations.
    (void)enable_if_disabled;
    tuples::get<InboxTag>(inbox_)[id].emplace(data);
  }

  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args) noexcept {
    local_algorithm_.template simple_action<Action>(std::move(args));
  }

  template <typename Action>
  void simple_action() noexcept {
    local_algorithm_.template simple_action<Action>();
  }

  template <typename Action, typename... Args>
  void threaded_action(std::tuple<Args...> args) noexcept {
    local_algorithm_.template threaded_action<Action>(std::move(args));
  }

  template <typename Action>
  void threaded_action() noexcept {
    local_algorithm_.template threaded_action<Action>();
  }

  void set_terminate(bool t) noexcept { local_algorithm_.set_terminate(t); }

  // Actions may call this, but since tests step through actions manually it has
  // no effect.
  void perform_algorithm() noexcept {}

  MockDistributedObject<Component>* ckLocal() { return &local_algorithm_; }

 private:
  MockDistributedObject<Component>& local_algorithm_;
  Inbox& inbox_;
};

// A mock class for the Charm++ generated CProxy_AlgorithmArray (we use an array
// for everything, so no need to mock groups, nodegroups, singletons).
template <typename Component, typename Index, typename InboxTagList>
class MockProxy {
 public:
  using Inboxes =
      std::unordered_map<Index,
                         tuples::tagged_tuple_from_typelist<InboxTagList>>;
  using TupleOfMockDistributedObjects =
      std::unordered_map<Index, MockDistributedObject<Component>>;

  MockProxy() : inboxes_(nullptr) {}

  void set_data(TupleOfMockDistributedObjects* local_algorithms,
                Inboxes* inboxes) {
    local_algorithms_ = local_algorithms;
    inboxes_ = inboxes;
  }

  MockArrayElementProxy<Component, InboxTagList> operator[](
      const Index& index) {
    ASSERT(local_algorithms_->count(index) == 1,
           "Should have exactly one local algorithm with key '"
               << index << "' but found " << local_algorithms_->count(index)
               << ". The known keys are " << keys_of(*local_algorithms_)
               << ". Did you forget to add a local algorithm when constructing "
                  "the MockRuntimeSystem?");
    return MockArrayElementProxy<Component, InboxTagList>(
        local_algorithms_->at(index), inboxes_->operator[](index));
  }

  MockDistributedObject<Component>* ckLocalBranch() noexcept {
    ASSERT(
        local_algorithms_->size() == 1,
        "Can only have one algorithm when getting the ckLocalBranch, but have "
            << local_algorithms_->size());
    // We always retrieve the 0th local branch because we are assuming running
    // on a single core.
    return std::addressof(local_algorithms_->at(0));
  }

  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args) noexcept {
    std::for_each(
        local_algorithms_->begin(), local_algorithms_->end(),
        [&args](auto& index_and_local_algorithm) noexcept {
          index_and_local_algorithm.second.template simple_action<Action>(args);
        });
  }

  template <typename Action>
  void simple_action() noexcept {
    std::for_each(
        local_algorithms_->begin(),
        local_algorithms_->end(), [](auto& index_and_local_algorithm) noexcept {
          index_and_local_algorithm.second.template simple_action<Action>();
        });
  }

  template <typename Action, typename... Args>
  void threaded_action(std::tuple<Args...> args) noexcept {
    if (local_algorithms_->size() != 1) {
      ERROR("NodeGroups must have exactly one element during testing, but have "
            << local_algorithms_->size());
    }
    local_algorithms_->begin()->second.template threaded_action<Action>(
        std::move(args));
  }

  template <typename Action>
  void threaded_action() noexcept {
    if (local_algorithms_->size() != 1) {
      ERROR("NodeGroups must have exactly one element during testing, but have "
            << local_algorithms_->size());
    }
    local_algorithms_->begin()->second.template threaded_action<Action>();
  }

  // clang-tidy: no non-const references
  void pup(PUP::er& /*p*/) noexcept {  // NOLINT
    ERROR(
        "Should not try to serialize the mock proxy. If you encountered this "
        "error you are using the mocking framework in a way that it was not "
        "intended to be used. It may be possible to extend it to more use "
        "cases but it is recommended you file an issue to discuss before "
        "modifying the mocking framework.");
  }

 private:
  TupleOfMockDistributedObjects* local_algorithms_;
  Inboxes* inboxes_;
};
}  // namespace ActionTesting_detail

/// A mock class for the CMake-generated `Parallel::Algorithms::Array`
struct MockArrayChare {
  template <typename Component, typename Index>
  using cproxy = ActionTesting_detail::MockProxy<
      Component, Index,
      Parallel::get_inbox_tags<typename Component::action_list>>;
};
}  // namespace ActionTesting

/// \cond HIDDEN_SYMBOLS
namespace Parallel {
template <>
struct get_array_index<ActionTesting::MockArrayChare> {
  template <typename Component>
  using f = typename Component::array_index;
};
}  // namespace Parallel
/// \endcond

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Structures used for mocking the parallel components framework in order
 * to test actions.
 */
namespace ActionTesting {
/// \ingroup TestingFrameworkGroup
/// A class that mocks the infrastructure needed to run actions.  It
/// simulates message passing using the inbox infrastructure and
/// handles most of the arguments to the apply and is_ready action
/// methods.
template <typename Metavariables>
class MockRuntimeSystem {
 public:
  // No moving, since MockProxy holds a pointer to us.
  MockRuntimeSystem(const MockRuntimeSystem&) = delete;
  MockRuntimeSystem(MockRuntimeSystem&&) = delete;
  MockRuntimeSystem& operator=(const MockRuntimeSystem&) = delete;
  MockRuntimeSystem& operator=(MockRuntimeSystem&&) = delete;
  ~MockRuntimeSystem() = default;

  template <typename Component>
  struct InboxesTag {
    using type = std::unordered_map<
        typename Component::array_index,
        tuples::tagged_tuple_from_typelist<
            Parallel::get_inbox_tags<typename Component::action_list>>>;
  };

  template <typename Component>
  struct MockDistributedObjectsTag {
    using type = std::unordered_map<typename Component::array_index,
                                    MockDistributedObject<Component>>;
  };

  using GlobalCache = Parallel::ConstGlobalCache<Metavariables>;
  using CacheTuple =
      tuples::tagged_tuple_from_typelist<typename GlobalCache::tag_list>;
  using TupleOfMockDistributedObjects = tuples::tagged_tuple_from_typelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<MockDistributedObjectsTag, tmpl::_1>>>;
  using Inboxes = tuples::tagged_tuple_from_typelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<InboxesTag, tmpl::_1>>>;

  /// Construct from the tuple of ConstGlobalCache objects.
  explicit MockRuntimeSystem(CacheTuple cache_contents,
                             TupleOfMockDistributedObjects local_algorithms)
      : cache_(std::move(cache_contents)),
        local_algorithms_(std::move(local_algorithms)) {
    tmpl::for_each<typename Metavariables::component_list>([this](
                                                               auto component) {
      using Component = tmpl::type_from<decltype(component)>;
      Parallel::get_parallel_component<Component>(cache_).set_data(
          &tuples::get<MockDistributedObjectsTag<Component>>(local_algorithms_),
          &tuples::get<InboxesTag<Component>>(inboxes_));

      for (auto& local_alg_pair : this->template algorithms<Component>()) {
        const auto& index = local_alg_pair.first;
        auto& local_alg = local_alg_pair.second;
        local_alg.set_index(index);
        local_alg.set_cache(&cache_);
        local_alg.set_inboxes(
            &(tuples::get<InboxesTag<Component>>(inboxes_)[index]));
      }
    });
  }

  // @{
  /// Invoke the simple action `Action` on the `Component` labeled by
  /// `array_index` immediately.
  template <typename Component, typename Action, typename Arg0,
            typename... Args>
  void simple_action(const typename Component::array_index& array_index,
                     Arg0&& arg0, Args&&... args) noexcept {
    algorithms<Component>()
        .at(array_index)
        .template simple_action<Action>(
            std::make_tuple(std::forward<Arg0>(arg0),
                            std::forward<Args>(args)...),
            true);
  }

  template <typename Component, typename Action>
  void simple_action(
      const typename Component::array_index& array_index) noexcept {
    algorithms<Component>()
        .at(array_index)
        .template simple_action<Action>(true);
  }
  // @}

  /// Return true if there are no queued simple actions on the
  /// `Component` labeled by `array_index`.
  template <typename Component>
  bool is_simple_action_queue_empty(
      const typename Component::array_index& array_index) noexcept {
    return algorithms<Component>()
        .at(array_index)
        .is_simple_action_queue_empty();
  }

  /// Invoke the next queued simple action on the `Component` labeled by
  /// `array_index`.
  template <typename Component>
  void invoke_queued_simple_action(
      const typename Component::array_index& array_index) noexcept {
    algorithms<Component>().at(array_index).invoke_queued_simple_action();
  }

  /// Return true if there are no queued threaded actions on the
  /// `Component` labeled by `array_index`.
  template <typename Component>
  bool is_threaded_action_queue_empty(
      const typename Component::array_index& array_index) noexcept {
    return algorithms<Component>()
        .at(array_index)
        .is_threaded_action_queue_empty();
  }

  /// Invoke the next queued threaded action on the `Component` labeled by
  /// `array_index`.
  template <typename Component>
  void invoke_queued_threaded_action(
      const typename Component::array_index& array_index) noexcept {
    algorithms<Component>().at(array_index).invoke_queued_threaded_action();
  }

  /// Instead of the next call to `next_action` applying the next action in
  /// the action list, force the next action to be `Action`
  template <typename Component, typename Action>
  void force_next_action_to_be(
      const typename Component::array_index& array_index) noexcept {
    static_assert(
        tmpl::list_contains_v<typename Component::action_list, Action>,
        "Cannot force a next action that is not in the action list of the "
        "parallel component. See the first template parameter of "
        "'force_next_action_to_be' for the component and the second template "
        "parameter for the action.");
    algorithms<Component>()
        .at(array_index)
        .force_next_action_to_be(
            tmpl::index_of<typename Component::action_list, Action>::value);
  }

  /// Obtain the index into the action list of the next action.
  template <typename Component>
  size_t get_next_action_index(
      const typename Component::array_index& array_index) noexcept {
    return algorithms<Component>().at(array_index).get_next_action_index();
  }

  /// Invoke the next action in the ActionList on the parallel component
  /// `Component` on the component labeled by `array_index`.
  template <typename Component>
  void next_action(
      const typename Component::array_index& array_index) noexcept {
    algorithms<Component>()
        .at(array_index)
        .next_action(
            std::make_index_sequence<tmpl::size<typename MockDistributedObject<
                Component>::actions_list>::value>{});
  }

  /// Call is_ready on the next action in the action list as if on the portion
  /// of Component labeled by array_index.
  template <typename Component>
  bool is_ready(const typename Component::array_index& array_index) noexcept {
    return algorithms<Component>()
        .at(array_index)
        .is_ready(
            std::make_index_sequence<tmpl::size<typename MockDistributedObject<
                Component>::actions_list>::value>{});
  }

  /// Access the inboxes for a given component.
  template <typename Component>
  std::unordered_map<
      typename Component::array_index,
      tuples::tagged_tuple_from_typelist<
          Parallel::get_inbox_tags<typename Component::action_list>>>&
  inboxes() noexcept {
    return tuples::get<InboxesTag<Component>>(inboxes_);
  }

  /// Find the set of array indices on Component where the specified
  /// inbox is not empty.
  template <typename Component, typename InboxTag>
  std::unordered_set<typename Component::array_index>
  nonempty_inboxes() noexcept {
    std::unordered_set<typename Component::array_index> result;
    for (const auto& element_box : inboxes<Component>()) {
      if (not tuples::get<InboxTag>(element_box.second).empty()) {
        result.insert(element_box.first);
      }
    }
    return result;
  }

  /// Access the mocked algorithms for a component, indexed by array index.
  template <typename Component>
  auto& algorithms() noexcept {
    return tuples::get<MockDistributedObjectsTag<Component>>(local_algorithms_);
  }

  GlobalCache& cache() noexcept { return cache_; }

 private:
  GlobalCache cache_;
  Inboxes inboxes_;
  TupleOfMockDistributedObjects local_algorithms_;
};

/// Returns a vector of all the indices of the Components
/// in the ComponentList that have queued actions.
template <typename ComponentList, typename MockRuntimeSystem,
          typename ArrayIndex>
std::vector<size_t> indices_of_components_with_queued_actions(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const ArrayIndex& array_index) noexcept {
  std::vector<size_t> result{};
  size_t i = 0;
  tmpl::for_each<ComponentList>([&](auto tag) noexcept {
    using Tag = typename decltype(tag)::type;
    if (not runner->template is_simple_action_queue_empty<Tag>(array_index)) {
      result.push_back(i);
    }
    ++i;
  });
  return result;
}

/// \cond
namespace detail {
// Helper function used in invoke_queued_simple_action.
template <typename ComponentList, typename MockRuntimeSystem,
          typename ArrayIndex, size_t... Is>
void invoke_queued_action(const gsl::not_null<MockRuntimeSystem*> runner,
                          const size_t component_to_invoke,
                          const ArrayIndex& array_index,
                          std::index_sequence<Is...> /*meta*/) noexcept {
  const auto helper =
      [ component_to_invoke, &runner, &array_index ](auto I) noexcept {
    if (I.value == component_to_invoke) {
      runner->template invoke_queued_simple_action<
          tmpl::at_c<ComponentList, I.value>>(array_index);
    }
  };
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
}
}  // namespace detail
/// \endcond

/// Invokes the next queued action on a random Component.
/// `index_map` is the thing returned by
/// `indices_of_components_with_queued_actions`
template <typename ComponentList, typename MockRuntimeSystem,
          typename Generator, typename ArrayIndex>
void invoke_random_queued_action(const gsl::not_null<MockRuntimeSystem*> runner,
                                 const gsl::not_null<Generator*> generator,
                                 const std::vector<size_t>& index_map,
                                 const ArrayIndex& array_index) noexcept {
  std::uniform_int_distribution<size_t> ran(0, index_map.size() - 1);
  const size_t component_to_invoke = index_map.at(ran(*generator));
  detail::invoke_queued_action<ComponentList>(
      runner, component_to_invoke, array_index,
      std::make_index_sequence<tmpl::size<ComponentList>::value>{});
}
}  // namespace ActionTesting
