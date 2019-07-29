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
#include "ParallelBackend/AlgorithmMetafunctions.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "ParallelBackend/NodeLock.hpp"
#include "ParallelBackend/ParallelComponentHelpers.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"
#include "ParallelBackend/Serialize.hpp"
#include "ParallelBackend/SimpleActionVisitation.hpp"
#include "Utilities/Algorithm.hpp"
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

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Structures used for mocking the parallel components framework in order
 * to test actions.
 */
namespace ActionTesting {}

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

/// \cond
template <typename SimpleTagsList, typename ComputeTagsList = tmpl::list<>>
struct InitializeDataBox;
/// \endcond

namespace detail {
// `get_initialization` computes the type of the TaggedTuple that holds the
// initial simple tags for the DataBox, as well as the
// `initialize_databox_action` so that the user interface to the Action Testing
// does not require the user to redundantly specify this information.
// `get_initialization` also performs various sanity checks for the user.
template <typename Component>
struct get_initialization {
  using phase = typename Component::metavariables::Phase;

  template <typename T>
  struct is_initialization_phase {
    using type =
        std::integral_constant<bool, T::phase == phase::Initialization>;
  };

  using initialization_pdal_list =
      tmpl::filter<typename Component::phase_dependent_action_list,
                   is_initialization_phase<tmpl::_1>>;
  static_assert(
      tmpl::size<initialization_pdal_list>::value == 1,
      "Must have exactly one Initialization PhaseDependentActionList");

  using initialization_pdal = typename tmpl::front<initialization_pdal_list>;

  static_assert(
      tt::is_a_v<ActionTesting::InitializeDataBox,
                 tmpl::front<typename initialization_pdal::action_list>>,
      "The first action in the initialization phase must be "
      "ActionTesting::InitializeDataBox, even if the simple and "
      "compute tags are empty.");

  using initialize_databox_action =
      tmpl::front<typename initialization_pdal::action_list>;

  using InitialValues = typename initialize_databox_action::InitialValues;
};

// Checks whether or not the `Metavariables` has a `Phase::Initialization`.
template <typename Metavariables, typename = cpp17::void_t<>>
struct has_initialization_phase : std::false_type {};

template <typename Metavariables>
struct has_initialization_phase<
    Metavariables,
    cpp17::void_t<decltype(Metavariables::Phase::Initialization)>>
    : std::true_type {};

template <typename Metavariables>
constexpr bool has_initialization_phase_v =
    has_initialization_phase<Metavariables>::value;
}  // namespace detail

/// \cond

// Initializes the DataBox values not set via the ConstGlobalCache. This is
// done as part of an `Initialization` phase and is triggered using the
// `emplace_component_and_initialize` function.
template <typename... SimpleTags, typename ComputeTagsList>
struct InitializeDataBox<tmpl::list<SimpleTags...>, ComputeTagsList> {
  using simple_tags = tmpl::list<SimpleTags...>;
  using compute_tags = ComputeTagsList;
  using InitialValues = tuples::TaggedTuple<SimpleTags...>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            typename ArrayIndex,
            Requires<not tmpl2::flat_all_v<
                tmpl::list_contains_v<DbTagsList, SimpleTags>...>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (not initial_values_valid_) {
      ERROR(
          "The values being used to construct the initial DataBox have not "
          "been set.");
    }
    initial_values_valid_ = false;
    return std::make_tuple(
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<SimpleTags...>,
                        ComputeTagsList>(
            std::move(box),
            std::move(tuples::get<SimpleTags>(initial_values_))...));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            typename ArrayIndex,
            Requires<tmpl2::flat_all_v<
                tmpl::list_contains_v<DbTagsList, SimpleTags>...>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }

  /// Sets the initial values of simple tags in the DataBox.
  static void set_initial_values(const InitialValues& t) noexcept {
    initial_values_ =
        deserialize<InitialValues>(serialize<InitialValues>(t).data());
    initial_values_valid_ = true;
  }

 private:
  static bool initial_values_valid_;
  static InitialValues initial_values_;
};

/// \cond
template <typename... SimpleTags, typename ComputeTagsList>
tuples::TaggedTuple<SimpleTags...> InitializeDataBox<
    tmpl::list<SimpleTags...>, ComputeTagsList>::initial_values_ = {};
template <typename... SimpleTags, typename ComputeTagsList>
bool InitializeDataBox<tmpl::list<SimpleTags...>,
                       ComputeTagsList>::initial_values_valid_ = false;
/// \endcond

/// MockDistributedObject mocks the AlgorithmImpl class. It should not be
/// considered as part of the user interface.
///
/// The `MockDistributedObject` represents a single chare or distributed object
/// on a supercomputer which can have methods invoked on it. This class is a
/// modified implementation of `AlgorithmImpl` and so some of the code is shared
/// between the two. The main difference is that `MockDistributedObject` has
/// support for introspection. For example, it is possible to check how many
/// simple actions are queued, to look at the inboxes, etc. Another key
/// difference is that `MockDistributedObject` runs only one action in the
/// action list at a time. This is done in order to provide opportunity for
/// introspection and checking statements before and after actions are invoked.
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

  // Holds the arguments passed to threaded actions. `InvokeThreadedAction` is
  // analogous to `InvokeSimpleAction`.
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
  using phase_dependent_action_lists =
      typename Component::phase_dependent_action_list;
  static_assert(tmpl::size<phase_dependent_action_lists>::value > 0,
                "Must have at least one phase dependent action list "
                "(PhaseActions) in a parallel component.");

  using all_actions_list = tmpl::flatten<tmpl::transform<
      phase_dependent_action_lists,
      Parallel::get_action_list_from_phase_dep_action_list<tmpl::_1>>>;

  using metavariables = typename Component::metavariables;

  using inbox_tags_list = Parallel::get_inbox_tags<all_actions_list>;

  using array_index = typename Parallel::get_array_index<
      typename Component::chare_type>::template f<Component>;

  using parallel_component = Component;

  using PhaseType =
      typename tmpl::front<phase_dependent_action_lists>::phase_type;

  using all_cache_tags =
      Parallel::ConstGlobalCache_detail::make_tag_list<metavariables>;
  using initial_tags = tmpl::flatten<tmpl::list<
      Parallel::Tags::ConstGlobalCacheImpl<metavariables>,
      typename Component::add_options_to_databox::simple_tags,
      db::wrap_tags_in<Parallel::Tags::FromConstGlobalCache, all_cache_tags>>>;
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
      const array_index& index,
      Parallel::ConstGlobalCache<typename Component::metavariables>* cache,
      tuples::tagged_tuple_from_typelist<inbox_tags_list>* inboxes,
      Options&&... opts)
      : array_index_(index), const_global_cache_(cache), inboxes_(inboxes) {
    box_ = Component::add_options_to_databox::apply(
        db::create<db::AddSimpleTags<
                       Parallel::Tags::ConstGlobalCacheImpl<metavariables>>,
                   db::AddComputeTags<db::wrap_tags_in<
                       Parallel::Tags::FromConstGlobalCache, all_cache_tags>>>(
            static_cast<const Parallel::ConstGlobalCache<metavariables>*>(
                const_global_cache_)),
        std::forward<Options>(opts)...);
  }

  void set_phase(PhaseType phase) noexcept { phase_ = phase; }
  PhaseType get_phase() const noexcept { return phase_; }

  void set_terminate(bool t) noexcept { terminate_ = t; }
  bool get_terminate() const noexcept { return terminate_; }

  // Actions may call this, but since tests step through actions manually it has
  // no effect.
  void perform_algorithm() noexcept {}

  // @{
  /// Returns the DataBox with the tags set from the ConstGlobalCache and the
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
      Parallel::Algorithm_detail::simple_action_visitor<Action, Component>(   \
          box_, *const_global_cache_,                                         \
          cpp17::as_const(array_index_)                                       \
              BOOST_PP_COMMA_IF(BOOST_PP_NOT(USE_SIMPLE_ACTION)) BOOST_PP_IF( \
                  USE_SIMPLE_ACTION, , make_not_null(&node_lock_)));          \
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
      Parallel::Algorithm_detail::simple_action_visitor<new_action,           \
                                                        Component>(           \
          box_, *const_global_cache_,                                         \
          cpp17::as_const(array_index_)                                       \
              BOOST_PP_COMMA_IF(BOOST_PP_NOT(USE_SIMPLE_ACTION)) BOOST_PP_IF( \
                  USE_SIMPLE_ACTION, , make_not_null(&node_lock_)));          \
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
    Parallel::Algorithm_detail::simple_action_visitor<Action, Component>(
        box_, *const_global_cache_, cpp17::as_const(array_index_),
        std::forward<Args>(std::get<Is>(args))...);
  }

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_threaded_action(
      std::tuple<Args...>&& args,
      std::index_sequence<Is...> /*meta*/) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<Action, Component>(
        box_, *const_global_cache_, cpp17::as_const(array_index_),
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
    if (box.which() == *iter and not*already_visited) {
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
    if (box.which() == *iter and not*already_visited) {
      ERROR("Cannot retrieve tag: "
            << db::tag_name<Tag>()
            << " from the current DataBox because it is not in it.");
    }
    (*iter)++;
  }

  template <typename Tag, typename... Variants>
  const auto& get_databox_tag_visitation(
      const boost::variant<Variants...>& box) const noexcept {
    using item_types = tmpl::remove_duplicates<
        tmpl::remove_if<tmpl::list<cpp20::remove_cvref_t<
                            db::item_type_if_contained_t<Tag, Variants>>...>,
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
  bool box_contains_visitation(const boost::variant<Variants...>& box) const
      noexcept {
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
    const auto helper = [&box, &is_retrievable ](auto box_type) noexcept {
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

  typename Component::array_index array_index_{};
  Parallel::ConstGlobalCache<typename Component::metavariables>*
      const_global_cache_{nullptr};
  tuples::tagged_tuple_from_typelist<inbox_tags_list>* inboxes_{nullptr};
  std::deque<std::unique_ptr<InvokeActionBase>> simple_action_queue_;
  std::deque<std::unique_ptr<InvokeActionBase>> threaded_action_queue_;
  CmiNodeLock node_lock_ = Parallel::create_lock();
};

template <typename Component>
void MockDistributedObject<Component>::next_action() noexcept {
  bool found_matching_phase = false;
  const auto invoke_for_phase =
      [ this, &found_matching_phase ](auto phase_dep_v) noexcept {
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
  const auto helper =
      [ this, &already_did_an_action ](auto iteration) noexcept {
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
    //                     box, inboxes_, *const_global_cache_,
    //                     cpp17::as_const(array_index_), actions_list{},
    //                     std::add_pointer_t<ParallelComponent>{}))>::type{}
    // ```
    const auto invoke_this_action = make_overloader(
        [this](auto& my_box, std::integral_constant<size_t, 1> /*meta*/)
            noexcept {
              std::tie(box_) = this_action::apply(
                  my_box, *inboxes_, *const_global_cache_,
                  cpp17::as_const(array_index_), actions_list{},
                  std::add_pointer_t<Component>{});
            },
        [this](auto& my_box, std::integral_constant<size_t, 2> /*meta*/)
            noexcept {
              std::tie(box_, terminate_) = this_action::apply(
                  my_box, *inboxes_, *const_global_cache_,
                  cpp17::as_const(array_index_), actions_list{},
                  std::add_pointer_t<Component>{});
            },
        [this](auto& my_box, std::integral_constant<size_t, 3> /*meta*/)
            noexcept {
              std::tie(box_, terminate_, algorithm_step_) = this_action::apply(
                  my_box, *inboxes_, *const_global_cache_,
                  cpp17::as_const(array_index_), actions_list{},
                  std::add_pointer_t<Component>{});
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
              check_local_box, cpp17::as_const(*inboxes_), *const_global_cache_,
              cpp17::as_const(array_index_));
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
                            Parallel::ConstGlobalCache<metavariables>,
                            array_index>{},
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
                        box, *inboxes_, *const_global_cache_,
                        cpp17::as_const(array_index_), actions_list{},
                        std::add_pointer_t<Component>{}))>::type{});
              } else if (box_.which() ==
                         static_cast<int>(
                             tmpl::index_of<variant_boxes,
                                            last_databox>::value)) {
                using this_databox = last_databox;
                auto& box = boost::get<this_databox>(box_);
                if (not check_if_ready(
                        Parallel::Algorithm_detail::is_is_ready_callable_t<
                            local_this_action, this_databox,
                            tuples::tagged_tuple_from_typelist<inbox_tags_list>,
                            Parallel::ConstGlobalCache<metavariables>,
                            array_index>{},
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
                        box, *inboxes_, *const_global_cache_,
                        cpp17::as_const(array_index_), actions_list{},
                        std::add_pointer_t<Component>{}))>::type{});
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
                            Parallel::ConstGlobalCache<metavariables>,
                            array_index>{},
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
                        box, *inboxes_, *const_global_cache_,
                        cpp17::as_const(array_index_), actions_list{},
                        std::add_pointer_t<Component>{}))>::type{});
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
  const auto invoke_for_phase =
      [ this, &action_is_ready, &
        found_matching_phase ](auto phase_dep_v) noexcept {
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
  const auto helper = [
    this, &array_index = array_index_, &inboxes = *inboxes_,
    &const_global_cache = const_global_cache_, &next_action_is_ready
  ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    using actions_list = typename PhaseDepActions::action_list;
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

    // `check_if_ready` calls the `is_ready` static method on the action
    // `action` if it has one, otherwise returns true. The first argument is the
    // ```
    // Algorithm_detail::is_is_ready_callable_t<action, databox,
    //         tuples::tagged_tuple_from_typelist<inbox_tags_list>,
    //         Parallel::ConstGlobalCache<metavariables>, array_index>{}
    // ```
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
  void perform_algorithm(const bool /*restart_if_terminated*/) noexcept {}

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
    alg::for_each(
        *local_algorithms_, [&args](auto& index_and_local_algorithm) noexcept {
          index_and_local_algorithm.second.template simple_action<Action>(args);
        });
  }

  template <typename Action>
  void simple_action() noexcept {
    alg::for_each(
        *local_algorithms_, [](auto& index_and_local_algorithm) noexcept {
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
      typename MockDistributedObject<Component>::inbox_tags_list>;
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

namespace ActionTesting {
/// \ingroup TestingFrameworkGroup
/// A class that mocks the infrastructure needed to run actions.  It simulates
/// message passing using the inbox infrastructure and handles most of the
/// arguments to the apply and is_ready action methods. This mocks the Charm++
/// runtime system as well as the layer built on top of it as part of SpECTRE.
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
            typename MockDistributedObject<Component>::inbox_tags_list>>;
  };

  template <typename Component>
  struct MockDistributedObjectsTag {
    using type = std::unordered_map<typename Component::array_index,
                                    MockDistributedObject<Component>>;
  };

  using GlobalCache = Parallel::ConstGlobalCache<Metavariables>;
  using CacheTuple = tuples::tagged_tuple_from_typelist<
      Parallel::ConstGlobalCache_detail::make_tag_list<Metavariables>>;

  using mock_objects_tags =
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<MockDistributedObjectsTag, tmpl::_1>>;
  using TupleOfMockDistributedObjects =
      tuples::tagged_tuple_from_typelist<mock_objects_tags>;
  using Inboxes = tuples::tagged_tuple_from_typelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<InboxesTag, tmpl::_1>>>;

  /// Construct from the tuple of ConstGlobalCache objects.
  explicit MockRuntimeSystem(CacheTuple cache_contents)
      : cache_(std::move(cache_contents)) {
    tmpl::for_each<typename Metavariables::component_list>([this](
                                                               auto component) {
      using Component = tmpl::type_from<decltype(component)>;
      Parallel::get_parallel_component<Component>(cache_).set_data(
          &tuples::get<MockDistributedObjectsTag<Component>>(local_algorithms_),
          &tuples::get<InboxesTag<Component>>(inboxes_));
    });
  }

  /// Emplace a component that does not need to be initialized.
  template <typename Component, typename... Options>
  void emplace_component(const typename Component::array_index& array_index,
                         Options&&... opts) noexcept {
    algorithms<Component>().emplace(
        array_index,
        MockDistributedObject<Component>(
            array_index, &cache_,
            &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
            std::forward<Options>(opts)...));
  }

  /// Emplace a component that needs to be initialized.
  template <typename Component, typename... Options,
            typename Metavars = Metavariables,
            Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
  void emplace_component_and_initialize(
      const typename Component::array_index& array_index,
      const typename detail::get_initialization<Component>::InitialValues&
          initial_values,
      Options&&... opts) noexcept {
    detail::get_initialization<Component>::initialize_databox_action::
        set_initial_values(initial_values);
    auto iterator_bool = algorithms<Component>().emplace(
        array_index,
        MockDistributedObject<Component>(
            array_index, &cache_,
            &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
            std::forward<Options>(opts)...));
    if (not iterator_bool.second) {
      ERROR("Failed to insert parallel component '"
            << pretty_type::get_name<Component>() << "' with index "
            << array_index);
    }
    iterator_bool.first->second.set_phase(Metavariables::Phase::Initialization);
    iterator_bool.first->second.next_action();
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

  // @{
  /// Invoke the threaded action `Action` on the `Component` labeled by
  /// `array_index` immediately.
  template <typename Component, typename Action, typename Arg0,
            typename... Args>
  void threaded_action(const typename Component::array_index& array_index,
                       Arg0&& arg0, Args&&... args) noexcept {
    algorithms<Component>()
        .at(array_index)
        .template threaded_action<Action>(
            std::make_tuple(std::forward<Arg0>(arg0),
                            std::forward<Args>(args)...),
            true);
  }

  template <typename Component, typename Action>
  void threaded_action(
      const typename Component::array_index& array_index) noexcept {
    algorithms<Component>()
        .at(array_index)
        .template threaded_action<Action>(true);
  }
  // @}

  /// Return true if there are no queued simple actions on the
  /// `Component` labeled by `array_index`.
  template <typename Component>
  bool is_simple_action_queue_empty(
      const typename Component::array_index& array_index) const noexcept {
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
      const typename Component::array_index& array_index) const noexcept {
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
        tmpl::list_contains_v<
            typename MockDistributedObject<Component>::all_actions_list,
            Action>,
        "Cannot force a next action that is not in the action list of the "
        "parallel component. See the first template parameter of "
        "'force_next_action_to_be' for the component and the second template "
        "parameter for the action.");
    bool found_matching_phase = false;
    const auto invoke_for_phase = [ this, &array_index, &found_matching_phase ](
        auto phase_dep_v) noexcept {
      using PhaseDep = decltype(phase_dep_v);
      constexpr typename Metavariables::Phase phase = PhaseDep::type::phase;
      using actions_list = typename PhaseDep::type::action_list;
      auto& distributed_object = this->algorithms<Component>().at(array_index);
      if (distributed_object.get_phase() == phase) {
        found_matching_phase = true;
        distributed_object.force_next_action_to_be(
            tmpl::conditional_t<
                std::is_same<tmpl::no_such_type_,
                             tmpl::index_of<actions_list, Action>>::value,
                std::integral_constant<size_t, 0>,
                tmpl::index_of<actions_list, Action>>::value);
      }
    };
    tmpl::for_each<typename MockDistributedObject<
        Component>::phase_dependent_action_lists>(invoke_for_phase);
    if (not found_matching_phase) {
      ERROR(
          "Could not find any actions in the current phase for the component '"
          << pretty_type::short_name<Component>()
          << "'. Maybe you are not in the phase you expected to be in? The "
             "integer value corresponding to the current phase is "
          << static_cast<int>(
                 this->algorithms<Component>().at(array_index).get_phase()));
    }
  }

  /// Obtain the index into the action list of the next action.
  template <typename Component>
  size_t get_next_action_index(
      const typename Component::array_index& array_index) const noexcept {
    return algorithms<Component>().at(array_index).get_next_action_index();
  }

  /// Invoke the next action in the ActionList on the parallel component
  /// `Component` on the component labeled by `array_index`.
  template <typename Component>
  void next_action(
      const typename Component::array_index& array_index) noexcept {
    algorithms<Component>().at(array_index).next_action();
  }

  /// Call is_ready on the next action in the action list as if on the portion
  /// of Component labeled by array_index.
  template <typename Component>
  bool is_ready(const typename Component::array_index& array_index) noexcept {
    return algorithms<Component>().at(array_index).is_ready();
  }

  /// Access the inboxes for a given component.
  template <typename Component>
  auto inboxes() noexcept -> std::unordered_map<
      typename Component::array_index,
      tuples::tagged_tuple_from_typelist<Parallel::get_inbox_tags<
          typename MockDistributedObject<Component>::all_actions_list>>>& {
    return tuples::get<InboxesTag<Component>>(inboxes_);
  }

  /// Find the set of array indices on Component where the specified
  /// inbox is not empty.
  template <typename Component, typename InboxTag>
  auto nonempty_inboxes() noexcept
      -> std::unordered_set<typename Component::array_index> {
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

  template <typename Component>
  const auto& algorithms() const noexcept {
    return tuples::get<MockDistributedObjectsTag<Component>>(local_algorithms_);
  }

  GlobalCache& cache() noexcept { return cache_; }

  /// Set the phase of all parallel components to `next_phase`
  void set_phase(const typename Metavariables::Phase next_phase) noexcept {
    tmpl::for_each<mock_objects_tags>(
        [ this, &next_phase ](auto component_v) noexcept {
          for (auto& object : tuples::get<typename decltype(component_v)::type>(
                   local_algorithms_)) {
            object.second.set_phase(next_phase);
          }
        });
  }

 private:
  GlobalCache cache_;
  Inboxes inboxes_;
  TupleOfMockDistributedObjects local_algorithms_;
};

/// Emplaces a distributed object with index `array_index` into the parallel
/// component `Component`. The options `opts` are forwarded to be used in a call
/// to `add_options_to_databox::apply`.
template <typename Component, typename... Options>
void emplace_component(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const typename Component::array_index& array_index,
    Options&&... opts) noexcept {
  runner->template emplace_component<Component>(array_index,
                                                std::forward<Options>(opts)...);
}

/// Emplaces a distributed object with index `array_index` into the parallel
/// component `Component`. The options `opts` are forwarded to be used in a call
/// to `add_options_to_databox::apply`. Additionally, the simple tags in the
/// DataBox are initialized from the values set in `initial_values`.
template <typename Component, typename... Options,
          typename Metavars = typename Component::metavariables,
          Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
void emplace_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const typename Component::array_index& array_index,
    const typename detail::get_initialization<Component>::InitialValues&
        initial_values,
    Options&&... opts) noexcept {
  runner->template emplace_component_and_initialize<Component>(
      array_index, initial_values, std::forward<Options>(opts)...);
}

// @{
/// Retrieves the DataBox with tags `TagsList` (omitting the `ConstGlobalCache`
/// and `add_from_options` tags) from the parallel component `Component` with
/// index `array_index`.
template <typename Component, typename TagsList, typename Metavariables>
const auto& get_databox(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template algorithms<Component>()
      .at(array_index)
      .template get_databox<TagsList>();
}

template <typename Component, typename TagsList, typename Metavariables>
auto& get_databox(const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
                  const typename Component::array_index& array_index) noexcept {
  return runner->template algorithms<Component>()
      .at(array_index)
      .template get_databox<TagsList>();
}
// @}

/// Get the index in the action list of the next action.
template <typename Component, typename Metavariables>
size_t get_next_action_index(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template get_next_action_index<Component>(array_index);
}

/// Returns the `Tag` from the `DataBox` of the parallel component `Component`
/// with array index `array_index`. If the component's current `DataBox` type
/// does not contain `Tag` then an error is emitted.
template <typename Component, typename Tag, typename Metavariables>
const auto& get_databox_tag(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template algorithms<Component>()
      .at(array_index)
      .template get_databox_tag<Tag>();
}

/// Returns `true` if the current DataBox of `Component` with index
/// `array_index` contains the tag `Tag`. If the tag is not contained, returns
/// `false`.
template <typename Component, typename Tag, typename Metavariables>
bool box_contains(const MockRuntimeSystem<Metavariables>& runner,
                  const typename Component::array_index& array_index) noexcept {
  return runner.template algorithms<Component>()
      .at(array_index)
      .template box_contains<Tag>();
}

/// Returns `true` if the tag `Tag` can be retrieved from the current DataBox
/// of `Component` with index `array_index`.
template <typename Component, typename Tag, typename Metavariables>
bool tag_is_retrievable(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template algorithms<Component>()
      .at(array_index)
      .template tag_is_retrievable<Tag>();
}

/// Runs the next action in the current phase on the `array_index`th element
/// of the parallel component `Component`.
template <typename Component, typename Metavariables>
void next_action(const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
                 const typename Component::array_index& array_index) noexcept {
  runner->template next_action<Component>(array_index);
}

/// Runs the `is_ready` function and returns the result for the next action in
/// the current phase on the `array_index`th element of the parallel component
/// `Component`.
template <typename Component, typename Metavariables>
bool is_ready(MockRuntimeSystem<Metavariables>& runner,
              const typename Component::array_index& array_index) noexcept {
  return runner.template is_ready<Component>(array_index);
}

/// Runs the simple action `Action` on the `array_index`th element of the
/// parallel component `Component`.
template <typename Component, typename Action, typename Metavariables,
          typename... Args>
void simple_action(
    const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
    const typename Component::array_index& array_index,
    Args&&... args) noexcept {
  runner->template simple_action<Component, Action>(
      array_index, std::forward<Args>(args)...);
}

/// Runs the simple action `Action` on the `array_index`th element of the
/// parallel component `Component`.
template <typename Component, typename Action, typename Metavariables,
          typename... Args>
void threaded_action(
    const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
    const typename Component::array_index& array_index,
    Args&&... args) noexcept {
  runner->template threaded_action<Component, Action>(
      array_index, std::forward<Args>(args)...);
}

/// Runs the next queued simple action on the `array_index`th element of
/// the parallel component `Component`.
template <typename Component, typename Metavariables>
void invoke_queued_simple_action(
    const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
    const typename Component::array_index& array_index) noexcept {
  runner->template invoke_queued_simple_action<Component>(array_index);
}

/// Returns `true` if there are no simple actions in the queue.
template <typename Component, typename Metavariables>
bool is_simple_action_queue_empty(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template is_simple_action_queue_empty<Component>(array_index);
}

/// Runs the next queued threaded action on the `array_index`th element of
/// the parallel component `Component`.
template <typename Component, typename Metavariables>
void invoke_queued_threaded_action(
    const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
    const typename Component::array_index& array_index) noexcept {
  runner->template invoke_queued_threaded_action<Component>(array_index);
}

/// Returns `true` if there are no threaded actions in the queue.
template <typename Component, typename Metavariables>
bool is_threaded_action_queue_empty(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template is_threaded_action_queue_empty<Component>(array_index);
}

/// Returns whether or not the `Component` with index `array_index` has been
/// terminated.
template <typename Component, typename Metavariables>
bool get_terminate(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template algorithms<Component>()
      .at(array_index)
      .get_terminate();
}

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
