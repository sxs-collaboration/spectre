// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Framework/MockDistributedObject.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
 *
 * The ActionTesting framework is designed to mock the parallel components so
 * that actions and sequences of actions can be tested in a controlled
 * environment that behaves effectively identical to the actual parallel
 * environment.
 *
 * ### The basics
 *
 * The action testing framework (ATF) works essentially identically to the
 * parallel infrastructure. A metavariables must be supplied which must at least
 * list the components used (`using component_list = tmpl::list<>`) and the
 * phases (`enum class Phase {}`). As a simple example, let's look at the test
 * for the `Parallel::Actions::TerminatePhase` action.
 *
 * The `Metavariables` is given by:
 *
 * \snippet Test_TerminatePhase.cpp metavariables
 *
 * The component list in this case just contains a single component that is
 * described below. The `Phase` enum contains the standard `Initialization` and
 * `Exit` phases. A `Testing` phase is also added, but the name of the phase has
 * no special meaning. Multiple phases with different names could instead be
 * used.
 *
 * The component is templated on the metavariables, which, while not always
 * necessary, eliminates some compilation issues that may arise otherwise. The
 * component for the `TerminatePhase` test is given by:
 *
 * \snippet Test_TerminatePhase.cpp component
 *
 * Just like with the standard parallel code, a `metavariables` type alias must
 * be present. The chare type should be `ActionTesting::MockArrayChare`,
 * `ActionTesting::MockGroupChare`, `ActionTesting::MockNodeGroupChare`, or
 * `ActionTesting::MockSingletonChare`. Currently many groups, nodegroups, and
 * singletons are treated during action testing as a one-element array
 * component, but that usage is deprecated and will eventually be removed.
 * The `index_type` must be whatever the actions will use to index
 * the array. In the case of a singleton `int` is recommended. Finally, the
 * `phase_dependent_action_list` must be specified just as in the cases for
 * parallel executables. In this case only the `Testing` phase is used and only
 * has the `TerminatePhase` action listed.
 *
 * The `SPECTRE_TEST_CASE` is
 *
 * \snippet Test_TerminatePhase.cpp test case
 *
 * The type alias `component = Component<Metavariables>` is just used to reduce
 * the amount of typing needed. The ATF provides a
 * `ActionTesting::MockRuntimeSystem` class, which is the code that takes the
 * place of the parallel runtime system (RTS) (e.g. Charm++) that manages the
 * actions and components. The constructor of
 * `ActionTesting::MockRuntimeSystem` takes a `std::vector<size_t>` whose
 * length is the number of (mocked) nodes and whose values are the number of
 * (mocked) cores on each node; the RTS runs only on a single core, but it
 * keeps track of which components are on which (mocked) nodes and cores
 * so that one can test some of the functionality of multiple cores/nodes.
 * Components are added to the RTS using the functions
 * `ActionTesting::emplace_array_component()`,
 * `ActionTesting::emplace_singleton_component()`,
 * `ActionTesting::emplace_group_component()`, and
 * `ActionTesting::emplace_nodegroup_component()`.  Currently there is a
 * deprecated `ActionTesting::emplace_component()` function that is the same as
 * `ActionTesting::emplace_array_component()` and can be used for the
 * deprecated usage of treating singletons, groups, and nodegroups as
 * single-element arrays.  The emplace functions take the runner by
 * `not_null`.  The array emplace function take the array index of the
 * component being inserted (`0` in the above example, but this is arbitrary),
 * and the array and singleton emplace functions take the (mocked) node and
 * core on which the component lives.  The emplace function for groups places
 * its object on all (mocked) cores and the array index is the same as the
 * global core index; the emplace function for
 * nodegroups places its object on a single (mocked) core on each
 * (mocked) node, and the array index is the same as the node index.
 * All emplace functions optionally take a parameter pack of
 * additional arguments that are forwarded to the constructor of the component.
 * These additional arguments are how input options (set in the
 * `initialization_tags` type alias of the parallel component) get passed to
 * parallel components.
 *
 * With the ATF, everything is controlled explicitly, providing the ability to
 * perform full introspection into the state of the RTS at virtually any point
 * during the execution. This means that the phase is not automatically advanced
 * like it is in a parallel executable, but instead must be advanced by calling
 * the `ActionTesting::set_phase` function. While testing the `TerminatePhase`
 * action we are only interesting in checking whether the algorithm has set the
 * `terminate` flag. This is done using the `ActionTesting::get_terminate()`
 * function and is done for each distributed object (i.e. per component). The
 * next action in the action list for the current phase is invoked by using the
 * `ActionTesting::next_action()` function, which takes as arguments the runner
 * by `not_null` and the array index of the distributed object to invoke the
 * next action on.
 *
 * ### InitializeDataBox and action introspection
 *
 * Having covered the very basics of ATF let us look at the test for the
 * `Actions::Goto` action. We will introduce the functionality
 * `InitializeDataBox`,`force_next_action_to_be`, `get_next_action_index`, and
 * `get_databox_tag`. The `Goto<the_label_class>` action changes the next action
 * in the algorithm to be the `Actions::Label<the_label_class>`. For this test
 * the Metavariables is:
 *
 * \snippet Test_Goto.cpp metavariables
 *
 * You can see that this time there are two test phases, `TestGoto` and
 * `TestRepeatUntil`.
 *
 * The component for this case is quite a bit more complicated so we will go
 * through it in detail. It is given by
 *
 * \snippet Test_Goto.cpp component
 *
 * Just as before there are `metavariables`, `chare_type`, and `array_index`
 * type aliases. The `repeat_until_phase_action_list` is the list of iterable
 * actions that are called during the `TestRepeatUntil` phase. The
 * `Initialization` phase action list now has the action
 * `ActionTesting::InitializeDataBox`, which takes simple tags and compute tags
 * that will be added to the DataBox in the initial phase. How the values for
 * the simple tags are set will be made clear below when we discuss the
 * `ActionTesting::emplace_component_and_initialize()` functions. The action
 * lists for the `TestGoto` and `TestRepeatUntil` phases are fairly simple and
 * not the focus of this discussion.
 *
 * Having discussed the metavariables and component, let us now look at the test
 * case.
 *
 * \snippet Test_Goto.cpp test case
 *
 * Just as for the `TerminatePhase` test we have some type aliases to reduce the
 * amount of typing needed and to make the test easier to read. The runner is
 * again created with the default constructor. However, the component is now
 * inserted using the functions
 * `ActionTesting::emplace_array_component_and_initialize()`,
 * `ActionTesting::emplace_singleton_component_and_initialize()`,
 * `ActionTesting::emplace_group_component_and_initialize()`,
 * and `ActionTesting::emplace_nodegroup_component_and_initialize()`.
 * There is a deprecated function
 * `ActionTesting::emplace_component_and_initialize()` that is used when one
 * treats groups, nodegroups, and singletons as arrays. The third argument to
 * the `ActionTesting::emplace_component_and_initialize()` functions is a
 * `tuples::TaggedTuple` of the simple tags, which is to be populated with the
 * initial values for the component. Note that there is no call to
 * `ActionTesting::set_phase (&runner, Metavariables::Phase::Initialization)`:
 * this and the required action invocation is handled internally by the
 * `ActionTesting::emplace_component_and_initialize()` functions.
 *
 * Once the phase is set the next action to be executed is set to be
 * `Actions::Label<Label1>` by calling the
 * `MockRuntimeSystem::force_next_action_to_be()`  member function of the
 * runner. The argument to the function is the array index of the component for
 * which to set the next action. After the `Label<Label1>` action is invoked we
 * check that the next action is the fourth (index `3` because we use zero-based
 * indexing) by calling the `MockRuntimeSystem::get_next_action_index()` member
 * function. For clarity, the indices of the actions in the `Phase::TestGoto`
 * phase are:
 * - 0: `Actions::Goto<Label1>`
 * - 1: `Actions::Label<Label2>`
 * - 2: `Actions::Label<Label1>`
 * - 3: `Actions::Goto<Label2>`
 *
 * ### DataBox introspection
 *
 * Since the exact DataBox type of any component at some point in the action
 * list may not be known, the `ActionTesting::get_databox_tag()` function is
 * provided. An example usage of this function can be seen in the last line of
 * the test snippet above. The component and tag are passed as template
 * parameters while the runner and array index are passed as arguments. It is
 * also possible to retrieve DataBox directly using the
 * `ActionTesting::get_databox()` function if the DataBox type is known:
 *
 * \snippet Test_ActionTesting.cpp get databox
 *
 * There is also a `gsl::not_null` overload of `ActionTesting::get_databox()`
 * that can be used to mutate tags in the DataBox. It is also possible to check
 * if an item can be retrieved from the DataBox using the
 * `ActionTesting::tag_is_retrievable()` function as follows:
 *
 * \snippet Test_ActionTesting.cpp tag is retrievable
 *
 * ### Stub actions and invoking simple and threaded actions
 *
 * A simple action can be invoked on a distributed object or component using
 * the `ActionTesting::simple_action()` function:
 *
 * \snippet Test_ActionTesting.cpp invoke simple action
 *
 * A threaded action can be invoked on a distributed object or component using
 * the `ActionTesting::threaded_action()` function:
 *
 * \snippet Test_ActionTesting.cpp invoke threaded action
 *
 * Sometimes an individual action calls other actions but we still want to be
 * able to test the action and its effects in isolation. To this end the ATF
 * supports replacing calls to actions with calls to other actions. For example,
 * instead of calling `simple_action_a` we want to call `simple_action_a_mock`
 * which just verifies the data received is correct and sets a flag in the
 * DataBox that it was invoked. This can be done by setting the type aliases
 * `replace_these_simple_actions` and `with_these_simple_actions` in the
 * parallel component definition as follows:
 *
 * \snippet Test_ActionTesting.cpp simple action replace
 *
 * The length of the two type lists must be the same and the Nth action in
 * `replace_these_simple_actions` gets replaced by the Nth action in
 * `with_these_simple_actions`. Note that `simple_action_a_mock` will get
 * invoked in any context that `simple_action_a` is called to be invoked. The
 * same feature also exists for threaded actions:
 *
 * \snippet Test_ActionTesting.cpp threaded action replace
 *
 * Furthermore, simple actions invoked from an action are not run immediately.
 * Instead, they are queued so that order randomization and introspection may
 * occur. The simplest form of introspection is checking whether the simple
 * action queue is empty:
 *
 * \snippet Test_ActionTesting.cpp simple action queue empty
 *
 * The `ActionTesting::invoke_queued_simple_action()` invokes the next queued
 * simple action:
 *
 * \snippet Test_ActionTesting.cpp invoke queued simple action
 *
 * Note that the same functionality exists for threaded actions. The functions
 * are `ActionTesting::is_threaded_action_queue_empty()`, and
 * `ActionTesting::invoke_queued_threaded_action()`.
 *
 * ### Mocking or replacing components with stubs
 *
 * An action can invoke an action on another parallel component. In this case we
 * need to be able to tell the mocking framework to replace the component the
 * action is trying to invoke the other action on and instead use a different
 * component that we have set up to mock the original component. For example,
 * the action below invokes an action on `ComponentB`
 *
 * \snippet Test_ActionTesting.cpp action call component b
 *
 * Let us assume we cannot use the component `ComponentB` in our test and that
 * we need to mock it. We do so using the type alias `component_being_mocked` in
 * the mock component:
 *
 * \snippet Test_ActionTesting.cpp mock component b
 *
 * When creating the runner `ComponentBMock` is emplaced:
 *
 * \snippet Test_ActionTesting.cpp initialize component b
 *
 * Any checks and function calls into the ATF also use `ComponentBMock`. For
 * example:
 *
 * \snippet Test_ActionTesting.cpp component b mock checks
 *
 * ### Const global cache tags
 *
 * Actions sometimes need tags/items to be placed into the
 * `Parallel::GlobalCache`. Once the list of tags for the global cache has
 * been assembled, the associated objects need to be inserted. This is done
 * using the constructor of the `ActionTesting::MockRuntimeSystem`. For example,
 * consider the tags:
 *
 * \snippet Test_ActionTesting.cpp tags for const global cache
 *
 * These are added into the global cache by, for example, the metavariables:
 *
 * \snippet Test_ActionTesting.cpp const global cache metavars
 *
 * A constructor of `ActionTesting::MockRuntimeSystem` takes a
 * `tuples::TaggedTuple` of the const global cache tags. If you know the order
 * of the tags in the `tuples::TaggedTuple` you can use the constructor without
 * explicitly specifying the type of the `tuples::TaggedTuple` as follows:
 *
 * \snippet Test_ActionTesting.cpp constructor const global cache tags known
 *
 * If you do not know the order of the tags but know all the tags that are
 * present you can use another constructor where you explicitly specify the
 * `tuples::TaggedTuple` type:
 *
 * \snippet Test_ActionTesting.cpp constructor const global cache tags unknown
 *
 * ### Inbox tags introspection
 *
 * The inbox tags can also be retrieved from a component by using the
 * `ActionTesting::get_inbox_tag` function. Both a const version:
 *
 * \snippet Test_ActionTesting.cpp const get inbox tags
 *
 * and a non-const version exist:
 *
 * \snippet Test_ActionTesting.cpp get inbox tags
 *
 * The non-const version can be used like in the above example to clear or
 * otherwise manipulate the inbox tags.
 *
 * ###
 */
namespace ActionTesting {}

namespace ActionTesting {

/// \cond
struct MockArrayChare;
struct MockGroupChare;
struct MockNodeGroupChare;
struct MockSingletonChare;
/// \endcond

// Initializes the DataBox values not set via the GlobalCache. This is
// done as part of an `Initialization` phase and is triggered using the
// `emplace_component_and_initialize` function.
template <typename... SimpleTags, typename ComputeTagsList>
struct InitializeDataBox<tmpl::list<SimpleTags...>, ComputeTagsList> {
  // these type aliases must be different from those used by `SetupDataBox` to
  // avoid a clash.
  using action_testing_simple_tags = tmpl::list<SimpleTags...>;
  using action_testing_compute_tags = ComputeTagsList;
  using InitialValues = tuples::TaggedTuple<SimpleTags...>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            typename ArrayIndex,
            Requires<not tmpl2::flat_any_v<
                tmpl::list_contains_v<DbTagsList, SimpleTags>...>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
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
            Requires<tmpl2::flat_any_v<
                tmpl::list_contains_v<DbTagsList, SimpleTags>...>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Tried to apply ActionTesting::InitializeDataBox even though one or "
        "more of its tags are already in the DataBox. Did you call next_action "
        "too many times in the initialization phase?");
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

namespace ActionTesting_detail {
// A mock class for the Charm++ generated CProxyElement_AlgorithmArray (we use
// an array for everything, so no need to mock groups, nodegroups, singletons).
template <typename Component, typename InboxTagList>
class MockDistributedObjectProxy {
 public:
  using Inbox = tuples::tagged_tuple_from_typelist<InboxTagList>;

  MockDistributedObjectProxy(
      size_t mock_node, size_t mock_local_core,
      MockDistributedObject<Component>& mock_distributed_object, Inbox& inbox)
      : mock_node_(mock_node),
        mock_local_core_(mock_local_core),
        mock_distributed_object_(mock_distributed_object),
        inbox_(inbox) {}

  template <typename InboxTag, typename Data>
  void receive_data(const typename InboxTag::temporal_id& id, Data&& data,
                    const bool enable_if_disabled = false) {
    // The variable `enable_if_disabled` might be useful in the future but is
    // not needed now. However, it is required by the interface to be compliant
    // with the Algorithm invocations.
    (void)enable_if_disabled;
    InboxTag::insert_into_inbox(make_not_null(&tuples::get<InboxTag>(inbox_)),
                                id, std::forward<Data>(data));
  }

  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args) noexcept {
    mock_distributed_object_.template simple_action<Action>(std::move(args));
  }

  template <typename Action>
  void simple_action() noexcept {
    mock_distributed_object_.template simple_action<Action>();
  }

  template <typename Action, typename... Args>
  void threaded_action(std::tuple<Args...> args) noexcept {
    mock_distributed_object_.template threaded_action<Action>(std::move(args));
  }

  template <typename Action>
  void threaded_action() noexcept {
    mock_distributed_object_.template threaded_action<Action>();
  }

  void set_terminate(bool t) noexcept {
    mock_distributed_object_.set_terminate(t);
  }

  // Actions may call this, but since tests step through actions manually it has
  // no effect.
  void perform_algorithm() noexcept {}
  void perform_algorithm(const bool /*restart_if_terminated*/) noexcept {}

  MockDistributedObject<Component>* ckLocal() {
    return (mock_distributed_object_.my_node() ==
                static_cast<int>(mock_node_) and
            mock_distributed_object_.my_local_rank() ==
                static_cast<int>(mock_local_core_))
               ? &mock_distributed_object_
               : nullptr;
  }

 private:
  // mock_node_ and mock_local_core_ are the (mocked) node and core
  // that this MockDistributedObjectProxy lives on.  This is different
  // than the (mock) node and core that the referred-to MockDistributedObject
  // lives on.
  size_t mock_node_{0};
  size_t mock_local_core_{0};
  MockDistributedObject<Component>& mock_distributed_object_;
  Inbox& inbox_;
};

// A mock class for the Charm++ generated CProxy_AlgorithmArray or
// CProxy_AlgorithmGroup or CProxy_AlgorithmNodeGroup. (for singletons, just
// use an array with a single element).
// Here ChareType is MockArrayChare or MockGroupChare or MockNodeGroupChare.
template <typename Component, typename Index, typename InboxTagList,
          typename ChareType>
class MockCollectionOfDistributedObjectsProxy {
 public:
  using Inboxes =
      std::unordered_map<Index,
                         tuples::tagged_tuple_from_typelist<InboxTagList>>;
  using CollectionOfMockDistributedObjects =
      std::unordered_map<Index, MockDistributedObject<Component>>;

  MockCollectionOfDistributedObjectsProxy() : inboxes_(nullptr) {}

  template <typename InboxTag, typename Data>
  void receive_data(const typename InboxTag::temporal_id& id, const Data& data,
                    const bool enable_if_disabled = false) {
    // Create (and call) a proxy on the local node and core that references
    // each of the mock_distributed_objects.
    for (const auto& key_value_pair : *mock_distributed_objects_) {
      MockDistributedObjectProxy<Component, InboxTagList>(
          mock_node_, mock_local_core_,
          mock_distributed_objects_->at(key_value_pair.first),
          inboxes_->operator[](key_value_pair.first))
          .template receive_data<InboxTag>(id, data, enable_if_disabled);
    }
  }

  void set_data(CollectionOfMockDistributedObjects* mock_distributed_objects,
                Inboxes* inboxes, const size_t mock_node,
                const size_t mock_local_core, const size_t mock_global_core) {
    mock_distributed_objects_ = mock_distributed_objects;
    inboxes_ = inboxes;
    mock_node_ = mock_node;
    mock_local_core_ = mock_local_core;
    mock_global_core_ = mock_global_core;
  }

  MockDistributedObjectProxy<Component, InboxTagList> operator[](
      const Index& index) {
    ASSERT(mock_distributed_objects_->count(index) == 1,
           "Should have exactly one mock distributed object with key '"
               << index << "' but found "
               << mock_distributed_objects_->count(index)
               << ". The known keys are " << keys_of(*mock_distributed_objects_)
               << ". Did you forget to add a mock distributed object when "
                  "constructing "
                  "the MockRuntimeSystem?");
    return MockDistributedObjectProxy<Component, InboxTagList>(
        mock_node_, mock_local_core_, mock_distributed_objects_->at(index),
        inboxes_->operator[](index));
  }

  // ckLocalBranch should never be called on an array or singleton
  // chare, because there is probably not a local branch on this processor.
  // We include it here to mock groups and nodegroups.  For a mocked group,
  // there is always one element per global core, so the index of the array
  // is the same as the global core index.
  // For a mocked nodegroup, there is one element per node, so the index
  // of the array is the node index.
  MockDistributedObject<Component>* ckLocalBranch() noexcept {
    if constexpr (std::is_same_v<ChareType, MockGroupChare>) {
      return std::addressof(mock_distributed_objects_->at(mock_global_core_));
    } else if constexpr (std::is_same_v<ChareType, MockNodeGroupChare>) {
      return std::addressof(mock_distributed_objects_->at(mock_node_));
    } else {
      static_assert(std::is_same_v<Component, NoSuchType>,
                  "Do not call ckLocalBranch for arrays or singletons");
    }
  }

  // ckLocal_enabled is used by Parallel/Invoke.hpp to allow or
  // prevent compilation of ckLocal (the idea is to prevent
  // compilation in cases where it will static_assert).
  // ckLocal_enabled can be eliminated if we replace the static_assert
  // in ckLocal() by an 'if constexpr' that returns nullptr when
  // ChareType is not a singleton, but that comes at the expense of a
  // compile-time check.
  using ckLocal_enabled = std::is_same<ChareType, MockSingletonChare>;

  // ckLocal should be called only on a singleton.
  MockDistributedObject<Component>* ckLocal() noexcept {
    static_assert(std::is_same_v<ChareType, MockSingletonChare>,
                  "Do not call ckLocal for other than a Singleton");
    auto& object = mock_distributed_objects_->at(0);
    return (object.my_node() == static_cast<int>(mock_node_) and
            object.my_local_rank() == static_cast<int>(mock_local_core_))
               ? std::addressof(object)
               : nullptr;
  }

  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args) noexcept {
    alg::for_each(*mock_distributed_objects_,
                  [&args](auto& index_and_mock_distributed_object) noexcept {
                    index_and_mock_distributed_object.second
                        .template simple_action<Action>(args);
                  });
  }

  template <typename Action>
  void simple_action() noexcept {
    alg::for_each(*mock_distributed_objects_,
                  [](auto& index_and_mock_distributed_object) noexcept {
                    index_and_mock_distributed_object.second
                        .template simple_action<Action>();
                  });
  }

  template <typename Action, typename... Args>
  void threaded_action(std::tuple<Args...> args) noexcept {
    static_assert(std::is_same_v<ChareType, MockNodeGroupChare>,
                  "Do not call threaded_action for other than a Nodegroup");
    alg::for_each(*mock_distributed_objects_,
                  [&args](auto& index_and_mock_distributed_object) noexcept {
                    index_and_mock_distributed_object.second
                        .template threaded_action<Action>(args);
                  });
  }

  template <typename Action>
  void threaded_action() noexcept {
    static_assert(std::is_same_v<ChareType, MockNodeGroupChare>,
                  "Do not call threaded_action for other than a Nodegroup");
    alg::for_each(*mock_distributed_objects_,
                  [](auto& index_and_mock_distributed_object) noexcept {
                    index_and_mock_distributed_object.second
                        .template threaded_action<Action>();
                  });
  }

  // clang-tidy: no non-const references
  void pup(PUP::er& /*p*/) noexcept {  // NOLINT
    ERROR(
        "Should not try to serialize the CollectionOfMockDistributedObjects. "
        "If you encountered this error you are using the mocking framework "
        "in a way that it was not intended to be used. It may be possible "
        "to extend it to more use cases but it is recommended you file an "
        "issue to discuss before modifying the mocking framework.");
  }

 private:
  CollectionOfMockDistributedObjects* mock_distributed_objects_;
  Inboxes* inboxes_;
  // mock_node_, mock_local_core_, and mock_global_core_ are the
  // (mock) node and core that the
  // MockCollectionOfDistributedObjectsProxy lives on.  This is
  // different than the (mock) nodes and cores that each element of the
  // referred-to CollectionOfMockDistributedObjects lives on.
  size_t mock_node_{0};
  size_t mock_local_core_{0};
  size_t mock_global_core_{0};
};
}  // namespace ActionTesting_detail

/// A mock class for the CMake-generated `Parallel::Algorithms::Array`
struct MockArrayChare {
  template <typename Component, typename Index>
  using cproxy = ActionTesting_detail::MockCollectionOfDistributedObjectsProxy<
      Component, Index,
      typename MockDistributedObject<Component>::inbox_tags_list,
      MockArrayChare>;
};
/// A mock class for the CMake-generated `Parallel::Algorithms::Group`
struct MockGroupChare {
  template <typename Component, typename Index>
  using cproxy = ActionTesting_detail::MockCollectionOfDistributedObjectsProxy<
      Component, Index,
      typename MockDistributedObject<Component>::inbox_tags_list,
      MockGroupChare>;
};
/// A mock class for the CMake-generated `Parallel::Algorithms::NodeGroup`
struct MockNodeGroupChare {
  template <typename Component, typename Index>
  using cproxy = ActionTesting_detail::MockCollectionOfDistributedObjectsProxy<
      Component, Index,
      typename MockDistributedObject<Component>::inbox_tags_list,
      MockNodeGroupChare>;
};
/// A mock class for the CMake-generated `Parallel::Algorithms::Singleton`
struct MockSingletonChare {
  template <typename Component, typename Index>
  using cproxy = ActionTesting_detail::MockCollectionOfDistributedObjectsProxy<
      Component, Index,
      typename MockDistributedObject<Component>::inbox_tags_list,
      MockSingletonChare>;
};
}  // namespace ActionTesting

/// \cond HIDDEN_SYMBOLS
namespace Parallel {
template <>
struct get_array_index<ActionTesting::MockArrayChare> {
  template <typename Component>
  using f = typename Component::array_index;
};
template <>
struct get_array_index<ActionTesting::MockGroupChare> {
  template <typename Component>
  using f = typename Component::array_index;
};
template <>
struct get_array_index<ActionTesting::MockNodeGroupChare> {
  template <typename Component>
  using f = typename Component::array_index;
};
template <>
struct get_array_index<ActionTesting::MockSingletonChare> {
  template <typename Component>
  using f = typename Component::array_index;
};
}  // namespace Parallel
/// \endcond
