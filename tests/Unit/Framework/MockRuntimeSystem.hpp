// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ErrorHandling/Error.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
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

  using mock_objects_tags =
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<MockDistributedObjectsTag, tmpl::_1>>;
  using CollectionOfMockDistributedObjects =
      tuples::tagged_tuple_from_typelist<mock_objects_tags>;
  using Inboxes = tuples::tagged_tuple_from_typelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<InboxesTag, tmpl::_1>>>;

  /// Construct from the tuple of GlobalCache objects.
  explicit MockRuntimeSystem(CacheTuple cache_contents)
      : mutable_cache_(tuples::TaggedTuple<>{}),
        // serialize_and_deserialize is not necessary here, but we
        // serialize_and_deserialize to reveal bugs in ActionTesting
        // tests where cache contents are not serializable.
        cache_(serialize_and_deserialize(cache_contents), &mutable_cache_)
  {
    tmpl::for_each<typename Metavariables::component_list>(
        [this](auto component) {
          using Component = tmpl::type_from<decltype(component)>;
          Parallel::get_parallel_component<Component>(cache_).set_data(
              &tuples::get<MockDistributedObjectsTag<Component>>(
                  mock_distributed_objects_),
              &tuples::get<InboxesTag<Component>>(inboxes_));
        });
  }

  /// Construct from the tuple of GlobalCache objects that might
  /// be in a different order.
  template <typename... Tags>
  explicit MockRuntimeSystem(tuples::TaggedTuple<Tags...> cache_contents)
      : MockRuntimeSystem(
            tuples::reorder<CacheTuple>(std::move(cache_contents))) {}

  /// Emplace a component that does not need to be initialized.
  template <typename Component, typename... Options>
  void emplace_component(const typename Component::array_index& array_index,
                         Options&&... opts) noexcept {
    mock_distributed_objects<Component>().emplace(
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
    auto iterator_bool = mock_distributed_objects<Component>().emplace(
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

  GlobalCache& cache() noexcept { return cache_; }

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
  Parallel::MutableGlobalCache<Metavariables> mutable_cache_{};
  GlobalCache cache_{};
  Inboxes inboxes_{};
  CollectionOfMockDistributedObjects mock_distributed_objects_{};
};
}  // namespace ActionTesting
