// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Framework/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace ActionTesting {
template <typename Component>
class MockDistributedObject;
template <typename SimpleTagsList, typename ComputeTagsList = tmpl::list<>>
struct InitializeDataBox;
struct MockArrayChare;
struct MockGroupChare;
struct MockNodeGroupChare;
struct MockSingletonChare;
}  // namespace ActionTesting
/// \endcond

namespace ActionTesting {

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
template <typename Metavariables, typename = std::void_t<>>
struct has_initialization_phase : std::false_type {};

template <typename Metavariables>
struct has_initialization_phase<
    Metavariables, std::void_t<decltype(Metavariables::Phase::Initialization)>>
    : std::true_type {};

template <typename Metavariables>
constexpr bool has_initialization_phase_v =
    has_initialization_phase<Metavariables>::value;
}  // namespace detail

/// \ingroup TestingFrameworkGroup
/// A class that mocks the infrastructure needed to run actions.  It simulates
/// message passing using the inbox infrastructure and handles most of the
/// arguments to the apply and is_ready action methods. This mocks the Charm++
/// runtime system as well as the layer built on top of it as part of SpECTRE.
template <typename Metavariables>
class MockRuntimeSystem {
 public:
  // No moving, since MockCollectionOfDistributedObjectsProxy holds a pointer to
  // us.
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

  using GlobalCache = Parallel::GlobalCache<Metavariables>;
  using CacheTuple = tuples::tagged_tuple_from_typelist<
      Parallel::get_const_global_cache_tags<Metavariables>>;
  using MutableCacheTuple = tuples::tagged_tuple_from_typelist<
      Parallel::get_mutable_global_cache_tags<Metavariables>>;

  using mock_objects_tags =
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<MockDistributedObjectsTag, tmpl::_1>>;
  using CollectionOfMockDistributedObjects =
      tuples::tagged_tuple_from_typelist<mock_objects_tags>;
  using Inboxes = tuples::tagged_tuple_from_typelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<InboxesTag, tmpl::_1>>>;


  /// Construct from the tuple of GlobalCache objects.
  explicit MockRuntimeSystem(
      CacheTuple cache_contents, MutableCacheTuple mutable_cache_contents = {},
      const std::vector<size_t>& number_of_mock_cores_on_each_mock_node = {1})
      : mock_global_cores_(
            [&number_of_mock_cores_on_each_mock_node]() noexcept {
              std::unordered_map<NodeId,
                                 std::unordered_map<LocalCoreId, GlobalCoreId>>
                  mock_global_cores{};
              size_t global_core = 0;
              for (size_t node = 0;
                   node < number_of_mock_cores_on_each_mock_node.size();
                   ++node) {
                std::unordered_map<LocalCoreId, GlobalCoreId> global_cores;
                for (size_t local_core = 0;
                     local_core < number_of_mock_cores_on_each_mock_node[node];
                     ++local_core, ++global_core) {
                  global_cores.insert(
                      {LocalCoreId{local_core}, GlobalCoreId{global_core}});
                }
                mock_global_cores.insert({NodeId{node}, global_cores});
              }
              return mock_global_cores;
            }()),
        mock_nodes_and_local_cores_(
            [&number_of_mock_cores_on_each_mock_node]() noexcept {
              std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
                  mock_nodes_and_local_cores{};
              size_t global_core = 0;
              for (size_t node = 0;
                   node < number_of_mock_cores_on_each_mock_node.size();
                   ++node) {
                for (size_t local_core = 0;
                     local_core < number_of_mock_cores_on_each_mock_node[node];
                     ++local_core, ++global_core) {
                  mock_nodes_and_local_cores.insert(
                      {GlobalCoreId{global_core},
                       std::make_pair(NodeId{node}, LocalCoreId{local_core})});
                }
              }
              return mock_nodes_and_local_cores;
            }()),
        mutable_caches_(mock_nodes_and_local_cores_.size()),
        caches_(mock_nodes_and_local_cores_.size()) {
    // Fill the parallel components in each cache.  There is one cache
    // per mock core.  The parallel components held in each cache (i.e. the
    // MockProxies) are filled with the same GlobalCache data, and are
    // passed the mock node and mock core associated with that cache.
    // The actual chares held in each cache are emplaced by the user explicitly
    // on a chosen node and core, via `emplace_component`.
    for (const auto& [global_core_id, node_and_local_core] :
         mock_nodes_and_local_cores_) {
      const size_t global_core = global_core_id.value;
      mutable_caches_.at(global_core) =
          std::make_unique<Parallel::MutableGlobalCache<Metavariables>>(
              serialize_and_deserialize(mutable_cache_contents));
      caches_.at(global_core) = std::make_unique<GlobalCache>(
          serialize_and_deserialize(cache_contents),
          mutable_caches_.at(global_core).get());
      // Note that lambdas cannot capture structured bindings,
      // so we define node and local_core here.
      const auto& node = node_and_local_core.first.value;
      const auto& local_core = node_and_local_core.second.value;
      tmpl::for_each<typename Metavariables::component_list>(
          [this, &global_core, &node, &local_core](auto component) {
            using Component = tmpl::type_from<decltype(component)>;
            Parallel::get_parallel_component<Component>(
                *caches_.at(global_core))
                .set_data(&tuples::get<MockDistributedObjectsTag<Component>>(
                              mock_distributed_objects_),
                          &tuples::get<InboxesTag<Component>>(inboxes_),
                          node, local_core, global_core);
          });
    }
  }

  /// Construct from the tuple of GlobalCache and MutableGlobalCache
  /// objects that might be in a different order.
  template <typename... CacheTags, typename... MutableCacheTags>
  MockRuntimeSystem(
      tuples::TaggedTuple<CacheTags...> cache_contents,
      tuples::TaggedTuple<MutableCacheTags...> mutable_cache_contents = {},
      const std::vector<size_t>& number_of_mock_cores_on_each_mock_node = {1})
      : MockRuntimeSystem(
            tuples::reorder<CacheTuple>(std::move(cache_contents)),
            tuples::reorder<MutableCacheTuple>(
                std::move(mutable_cache_contents)),
            number_of_mock_cores_on_each_mock_node) {}

  /// Emplace an array component that does not need to be initialized.
  template <typename Component, typename... Options>
  void emplace_array_component(
      const NodeId node_id, const LocalCoreId local_core_id,
      const typename Component::array_index& array_index,
      Options&&... opts) noexcept {
    mock_distributed_objects<Component>().emplace(
        array_index,
        MockDistributedObject<Component>(
            node_id, local_core_id, mock_global_cores_,
            mock_nodes_and_local_cores_, array_index,
            caches_.at(mock_global_cores_.at(node_id).at(local_core_id).value)
                .get(),
            // Next line inserts element into inboxes_ and returns ptr to it.
            &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
            std::forward<Options>(opts)...));
  }

  /// Emplace a singleton component that does not need to be initialized.
  template <typename Component, typename... Options>
  void emplace_singleton_component(const NodeId node_id,
                                   const LocalCoreId local_core_id,
                                   Options&&... opts) noexcept {
    static_assert(
        std::is_same_v<typename Component::chare_type, MockSingletonChare>,
        "emplace_singleton_component expects a MockSingletonChare");
    // For a singleton, use an array index of zero.
    const typename Component::array_index array_index{0};
    mock_distributed_objects<Component>().emplace(
        array_index,
        MockDistributedObject<Component>(
            node_id, local_core_id, mock_global_cores_,
            mock_nodes_and_local_cores_, array_index,
            caches_.at(mock_global_cores_.at(node_id).at(local_core_id).value)
                .get(),
            // Next line inserts element into inboxes_ and returns ptr to it.
            &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
            std::forward<Options>(opts)...));
  }

  /// Emplace a group component that does not need to be initialized.
  template <typename Component, typename... Options>
  void emplace_group_component(Options&&... opts) noexcept {
    static_assert(
        std::is_same_v<typename Component::chare_type, MockGroupChare>,
        "emplace_group_component expects a MockGroupChare");
    // Emplace once for each core, index by global_core.
    for (const auto& [global_core_id, node_and_local_core] :
         mock_nodes_and_local_cores_) {
      const size_t global_core = global_core_id.value;
      const typename Component::array_index array_index = global_core;
      mock_distributed_objects<Component>().emplace(
          array_index,
          MockDistributedObject<Component>(
              node_and_local_core.first, node_and_local_core.second,
              mock_global_cores_, mock_nodes_and_local_cores_, array_index,
              caches_.at(global_core).get(),
              // Next line inserts element into inboxes_ and returns ptr to it.
              &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
              std::forward<Options>(opts)...));
    }
  }

  /// Emplace a nodegroup component that does not need to be initialized.
  template <typename Component, typename... Options>
  void emplace_nodegroup_component(Options&&... opts) noexcept {
    static_assert(
        std::is_same_v<typename Component::chare_type, MockNodeGroupChare>,
        "emplace_nodegroup_component expects a MockNodeGroupChare");
    // Emplace once for each node, index by node number.
    for (const auto& [node, local_and_global_cores] : mock_global_cores_) {
      const typename Component::array_index array_index = node.value;
      // Use first proc on each node as global_core
      const size_t global_core =
          local_and_global_cores.at(LocalCoreId{0}).value;
      mock_distributed_objects<Component>().emplace(
          array_index,
          MockDistributedObject<Component>(
              node, LocalCoreId{0}, mock_global_cores_,
              mock_nodes_and_local_cores_, array_index,
              caches_.at(global_core).get(),
              // Next line inserts element into inboxes_ and returns ptr to it.
              &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
              std::forward<Options>(opts)...));
    }
  }

  /// Emplace a component that does not need to be initialized.
  /// emplace_component is deprecated in favor of
  /// emplace_array_component, emplace_singleton_component,
  /// emplace_group_component, and emplace_nodegroup_component.
  template <typename Component, typename... Options>
  void emplace_component(const typename Component::array_index& array_index,
                         Options&&... opts) noexcept {
    emplace_array_component<Component>(NodeId{0}, LocalCoreId{0}, array_index,
                                       std::forward<Options>(opts)...);
  }

  /// Emplace an array component that needs to be initialized.
  template <typename Component, typename... Options>
  void emplace_array_component_and_initialize(
      const NodeId node_id, const LocalCoreId local_core_id,
      const typename Component::array_index& array_index,
      const typename detail::get_initialization<Component>::InitialValues&
          initial_values,
      Options&&... opts) noexcept {
    detail::get_initialization<Component>::initialize_databox_action::
        set_initial_values(initial_values);
    auto [iterator, emplace_was_successful] =
        mock_distributed_objects<Component>().emplace(
            array_index,
            MockDistributedObject<Component>(
                node_id, local_core_id, mock_global_cores_,
                mock_nodes_and_local_cores_, array_index,
                caches_
                    .at(mock_global_cores_.at(node_id).at(local_core_id).value)
                    .get(),
                // Next line inserts element into inboxes_ and returns ptr to
                // it.
                &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
                std::forward<Options>(opts)...));
    if (not emplace_was_successful) {
      ERROR("Failed to insert parallel component '"
            << pretty_type::get_name<Component>() << "' with index "
            << array_index);
    }
    iterator->second.set_phase(Metavariables::Phase::Initialization);
    iterator->second.next_action();
  }

  /// Emplace a singleton component that needs to be initialized.
  template <typename Component, typename... Options>
  void emplace_singleton_component_and_initialize(
      const NodeId node_id, const LocalCoreId local_core_id,
      const typename detail::get_initialization<Component>::InitialValues&
          initial_values,
      Options&&... opts) noexcept {
    static_assert(
        std::is_same_v<typename Component::chare_type, MockSingletonChare>,
        "emplace_singleton_component_and_initialize expects a "
        "MockSingletonChare");
    // Use 0 for the array_index
    const typename Component::array_index& array_index{0};
    detail::get_initialization<Component>::initialize_databox_action::
        set_initial_values(initial_values);
    auto [iterator, emplace_was_successful] =
        mock_distributed_objects<Component>().emplace(
            array_index,
            MockDistributedObject<Component>(
                node_id, local_core_id, mock_global_cores_,
                mock_nodes_and_local_cores_, array_index,
                caches_
                    .at(mock_global_cores_.at(node_id).at(local_core_id).value)
                    .get(),
                // Next line inserts element into inboxes_ and returns ptr to
                // it.
                &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
                std::forward<Options>(opts)...));
    if (not emplace_was_successful) {
      ERROR("Failed to insert parallel component '"
            << pretty_type::get_name<Component>() << "' with index "
            << array_index);
    }
    iterator->second.set_phase(Metavariables::Phase::Initialization);
    iterator->second.next_action();
  }

  /// Emplace a group component that needs to be initialized.
  template <typename Component, typename... Options>
  void emplace_group_component_and_initialize(
      const typename detail::get_initialization<Component>::InitialValues&
          initial_values,
      Options&&... opts) noexcept {
    static_assert(
        std::is_same_v<typename Component::chare_type, MockGroupChare>,
        "emplace_group_component_and_initialize expects a MockGroupChare");
    // Emplace once for each core, index by global_core.
    for (const auto& [global_core_id, node_and_local_core] :
         mock_nodes_and_local_cores_) {
      // Need to set initial values for each component, since the
      // InitialDataBox action does a std::move.
      detail::get_initialization<Component>::initialize_databox_action::
          set_initial_values(initial_values);
      const size_t global_core = global_core_id.value;
      const typename Component::array_index array_index = global_core;
      auto [iterator, emplace_was_successful] =
          mock_distributed_objects<Component>().emplace(
              array_index,
              MockDistributedObject<Component>(
                  node_and_local_core.first, node_and_local_core.second,
                  mock_global_cores_, mock_nodes_and_local_cores_, array_index,
                  caches_.at(global_core).get(),
                  // Next line inserts element into inboxes_ and returns ptr to
                  // it.
                  &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
                  std::forward<Options>(opts)...));
      if (not emplace_was_successful) {
        ERROR("Failed to insert parallel component '"
              << pretty_type::get_name<Component>() << "' with index "
              << array_index);
      }
      iterator->second.set_phase(Metavariables::Phase::Initialization);
      iterator->second.next_action();
    }
  }

  /// Emplace a nodegroup component that needs to be initialized.
  template <typename Component, typename... Options>
  void emplace_nodegroup_component_and_initialize(
      const typename detail::get_initialization<Component>::InitialValues&
          initial_values,
      Options&&... opts) noexcept {
    static_assert(
        std::is_same_v<typename Component::chare_type, MockNodeGroupChare>,
        "emplace_nodegroup_component_and_initialize expects a "
        "MockNodeGroupChare");
    // Emplace once for each node, index by node number.
    for (const auto& [node, local_and_global_cores] : mock_global_cores_) {
      // Need to set initial values for each component, since the
      // InitialDataBox action does a std::move.
      detail::get_initialization<Component>::initialize_databox_action::
          set_initial_values(initial_values);
      const typename Component::array_index array_index = node.value;
      // Use first proc on each node as global_core
      const size_t global_core =
          local_and_global_cores.at(LocalCoreId{0}).value;
      auto [iterator, emplace_was_successful] =
          mock_distributed_objects<Component>().emplace(
              array_index,
              MockDistributedObject<Component>(
                  node, LocalCoreId{0}, mock_global_cores_,
                  mock_nodes_and_local_cores_, array_index,
                  caches_.at(global_core).get(),
                  // Next line inserts element into inboxes_ and returns ptr to
                  // it.
                  &(tuples::get<InboxesTag<Component>>(inboxes_)[array_index]),
                  std::forward<Options>(opts)...));
      if (not emplace_was_successful) {
        ERROR("Failed to insert parallel component '"
              << pretty_type::get_name<Component>() << "' with index "
              << array_index);
      }
      iterator->second.set_phase(Metavariables::Phase::Initialization);
      iterator->second.next_action();
    }
    }

  /// Emplace a component that needs to be initialized.
  /// emplace_component_and_initialize is deprecated in favor of
  /// emplace_array_component_and_initialize,
  /// emplace_singleton_component_and_initialize,
  /// emplace_group_component_and_initialize, and
  /// emplace_nodegroup_component_and_initialize.
  template <typename Component, typename... Options,
            typename Metavars = Metavariables,
            Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
  void emplace_component_and_initialize(
      const typename Component::array_index& array_index,
      const typename detail::get_initialization<Component>::InitialValues&
          initial_values,
      Options&&... opts) noexcept {
    emplace_array_component_and_initialize<Component>(
        NodeId{0}, LocalCoreId{0}, array_index, initial_values,
        std::forward<Options>(opts)...);
  }

  // @{
  /// Invoke the simple action `Action` on the `Component` labeled by
  /// `array_index` immediately.
  template <typename Component, typename Action, typename Arg0,
            typename... Args>
  void simple_action(const typename Component::array_index& array_index,
                     Arg0&& arg0, Args&&... args) noexcept {
    mock_distributed_objects<Component>()
        .at(array_index)
        .template simple_action<Action>(
            std::make_tuple(std::forward<Arg0>(arg0),
                            std::forward<Args>(args)...),
            true);
  }

  template <typename Component, typename Action>
  void simple_action(
      const typename Component::array_index& array_index) noexcept {
    mock_distributed_objects<Component>()
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
    mock_distributed_objects<Component>()
        .at(array_index)
        .template threaded_action<Action>(
            std::make_tuple(std::forward<Arg0>(arg0),
                            std::forward<Args>(args)...),
            true);
  }

  template <typename Component, typename Action>
  void threaded_action(
      const typename Component::array_index& array_index) noexcept {
    mock_distributed_objects<Component>()
        .at(array_index)
        .template threaded_action<Action>(true);
  }
  // @}

  /// Return true if there are no queued simple actions on the
  /// `Component` labeled by `array_index`.
  template <typename Component>
  bool is_simple_action_queue_empty(
      const typename Component::array_index& array_index) const noexcept {
    return mock_distributed_objects<Component>()
        .at(array_index)
        .is_simple_action_queue_empty();
  }

  /// Invoke the next queued simple action on the `Component` labeled by
  /// `array_index`.
  template <typename Component>
  void invoke_queued_simple_action(
      const typename Component::array_index& array_index) noexcept {
    mock_distributed_objects<Component>()
        .at(array_index)
        .invoke_queued_simple_action();
  }

  /// Return true if there are no queued threaded actions on the
  /// `Component` labeled by `array_index`.
  template <typename Component>
  bool is_threaded_action_queue_empty(
      const typename Component::array_index& array_index) const noexcept {
    return mock_distributed_objects<Component>()
        .at(array_index)
        .is_threaded_action_queue_empty();
  }

  /// Invoke the next queued threaded action on the `Component` labeled by
  /// `array_index`.
  template <typename Component>
  void invoke_queued_threaded_action(
      const typename Component::array_index& array_index) noexcept {
    mock_distributed_objects<Component>()
        .at(array_index)
        .invoke_queued_threaded_action();
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
    const auto invoke_for_phase =
        [this, &array_index, &found_matching_phase](auto phase_dep_v) noexcept {
          using PhaseDep = decltype(phase_dep_v);
          constexpr typename Metavariables::Phase phase = PhaseDep::type::phase;
          using actions_list = typename PhaseDep::type::action_list;
          auto& distributed_object =
              this->mock_distributed_objects<Component>().at(array_index);
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
          << static_cast<int>(this->mock_distributed_objects<Component>()
                                  .at(array_index)
                                  .get_phase()));
    }
  }

  /// Obtain the index into the action list of the next action.
  template <typename Component>
  size_t get_next_action_index(
      const typename Component::array_index& array_index) const noexcept {
    return mock_distributed_objects<Component>()
        .at(array_index)
        .get_next_action_index();
  }

  /// Invoke the next action in the ActionList on the parallel component
  /// `Component` on the component labeled by `array_index`.
  template <typename Component>
  void next_action(
      const typename Component::array_index& array_index) noexcept {
    mock_distributed_objects<Component>().at(array_index).next_action();
  }

  /// Call is_ready on the next action in the action list as if on the portion
  /// of Component labeled by array_index.
  template <typename Component>
  bool is_ready(const typename Component::array_index& array_index) noexcept {
    return mock_distributed_objects<Component>().at(array_index).is_ready();
  }

  // @{
  /// Access the inboxes for a given component.
  template <typename Component>
  auto inboxes() noexcept -> std::unordered_map<
      typename Component::array_index,
      tuples::tagged_tuple_from_typelist<
          typename MockDistributedObject<Component>::inbox_tags_list>>& {
    return tuples::get<InboxesTag<Component>>(inboxes_);
  }

  template <typename Component>
  auto inboxes() const noexcept -> const std::unordered_map<
      typename Component::array_index,
      tuples::tagged_tuple_from_typelist<
          typename MockDistributedObject<Component>::inbox_tags_list>>& {
    return tuples::get<InboxesTag<Component>>(inboxes_);
  }
  // @}

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

  /// Access the mocked distributed objects for a component, indexed by array
  /// index.
  template <typename Component>
  auto& mock_distributed_objects() noexcept {
    return tuples::get<MockDistributedObjectsTag<Component>>(
        mock_distributed_objects_);
  }

  template <typename Component>
  const auto& mock_distributed_objects() const noexcept {
    return tuples::get<MockDistributedObjectsTag<Component>>(
        mock_distributed_objects_);
  }

  /// Set the phase of all parallel components to `next_phase`
  void set_phase(const typename Metavariables::Phase next_phase) noexcept {
    tmpl::for_each<mock_objects_tags>(
        [this, &next_phase](auto component_v) noexcept {
          for (auto& object : tuples::get<typename decltype(component_v)::type>(
                   mock_distributed_objects_)) {
            object.second.set_phase(next_phase);
          }
        });
  }

 private:
  std::unordered_map<NodeId, std::unordered_map<LocalCoreId, GlobalCoreId>>
      mock_global_cores_{};
  std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
      mock_nodes_and_local_cores_{};
  // There is one MutableGlobalCache and one GlobalCache per
  // global_core.  We need to keep a vector of unique_ptrs rather than
  // a vector of objects because MutableGlobalCache and GlobalCache
  // are not move-assignable or copyable (because of their charm++
  // base classes, in particular CkReductionMgr).
  std::vector<std::unique_ptr<Parallel::MutableGlobalCache<Metavariables>>>
      mutable_caches_{};
  std::vector<std::unique_ptr<GlobalCache>> caches_{};
  Inboxes inboxes_{};
  CollectionOfMockDistributedObjects mock_distributed_objects_{};
};
}  // namespace ActionTesting
