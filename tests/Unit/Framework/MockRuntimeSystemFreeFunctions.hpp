// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines free functions that wrap MockRuntimeSystem member functions.

#pragma once

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "Framework/MockRuntimeSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace ActionTesting {
/// Set the phase of all parallel components to `phase`
template <typename Metavariables>
void set_phase(const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
               const typename Metavariables::Phase& phase) noexcept {
  runner->set_phase(phase);
}

/// Emplaces a distributed object with index `array_index` into the parallel
/// component `Component`. The options `opts` are forwarded to be used in a call
/// to `detail::ForwardAllOptionsToDataBox::apply`.
template <typename Component, typename... Options>
void emplace_component(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const typename Component::array_index& array_index,
    Options&&... opts) noexcept {
  runner->template emplace_component<Component>(array_index,
                                                std::forward<Options>(opts)...);
}

/// Emplaces a distributed array object with index `array_index` into
/// the parallel component `Component`. The options `opts` are
/// forwarded to be used in a call to
/// `detail::ForwardAllOptionsToDataBox::apply`.
template <typename Component, typename... Options>
void emplace_array_component(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const NodeId node_id, const LocalCoreId local_core_id,
    const typename Component::array_index& array_index,
    Options&&... opts) noexcept {
  runner->template emplace_array_component<Component>(
      node_id, local_core_id, array_index, std::forward<Options>(opts)...);
}

/// Emplaces a distributed singleton object into
/// the parallel component `Component`. The options `opts` are
/// forwarded to be used in a call to
/// `detail::ForwardAllOptionsToDataBox::apply`.
template <typename Component, typename... Options>
void emplace_singleton_component(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const NodeId node_id, const LocalCoreId local_core_id,
    Options&&... opts) noexcept {
  runner->template emplace_singleton_component<Component>(
      node_id, local_core_id, std::forward<Options>(opts)...);
}

/// Emplaces a distributed group object into
/// the parallel component `Component`. The options `opts` are
/// forwarded to be used in a call to
/// `detail::ForwardAllOptionsToDataBox::apply`.
template <typename Component, typename... Options>
void emplace_group_component(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    Options&&... opts) noexcept {
  runner->template emplace_group_component<Component>(
      std::forward<Options>(opts)...);
}

/// Emplaces a distributed nodegroup object into
/// the parallel component `Component`. The options `opts` are
/// forwarded to be used in a call to
/// `detail::ForwardAllOptionsToDataBox::apply`.
template <typename Component, typename... Options>
void emplace_nodegroup_component(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    Options&&... opts) noexcept {
  runner->template emplace_nodegroup_component<Component>(
      std::forward<Options>(opts)...);
}

/// Emplaces a distributed object with index `array_index` into the parallel
/// component `Component`. The options `opts` are forwarded to be used in a call
/// to `detail::ForwardAllOptionsToDataBox::apply` Additionally, the simple tags
/// in the DataBox are initialized from the values set in `initial_values`.
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

/// Emplaces a distributed array object with index `array_index` into the
/// parallel component `Component`. The options `opts` are forwarded to be used
/// in a call to `detail::ForwardAllOptionsToDataBox::apply` Additionally, the
/// simple tags in the DataBox are initialized from the values set in
/// `initial_values`.
template <typename Component, typename... Options,
          typename Metavars = typename Component::metavariables,
          Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
void emplace_array_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const NodeId node_id, const LocalCoreId local_core_id,
    const typename Component::array_index& array_index,
    const typename detail::get_initialization<Component>::InitialValues&
        initial_values,
    Options&&... opts) noexcept {
  runner->template emplace_array_component_and_initialize<Component>(
      node_id, local_core_id, array_index, initial_values,
      std::forward<Options>(opts)...);
}

/// Emplaces a distributed singleton object into the parallel
/// component `Component`. The options `opts` are forwarded to be used
/// in a call to `detail::ForwardAllOptionsToDataBox::apply`
/// Additionally, the simple tags in the DataBox are initialized from
/// the values set in `initial_values`.
template <typename Component, typename... Options,
          typename Metavars = typename Component::metavariables,
          Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
void emplace_singleton_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const NodeId node_id, const LocalCoreId local_core_id,
    const typename detail::get_initialization<Component>::InitialValues&
        initial_values,
    Options&&... opts) noexcept {
  runner->template emplace_singleton_component_and_initialize<Component>(
      node_id, local_core_id, initial_values, std::forward<Options>(opts)...);
}

/// Emplaces a distributed group object into the
/// parallel component `Component`. The options `opts` are forwarded to be used
/// in a call to `detail::ForwardAllOptionsToDataBox::apply` Additionally, the
/// simple tags in the DataBox are initialized from the values set in
/// `initial_values`.
template <typename Component, typename... Options,
          typename Metavars = typename Component::metavariables,
          Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
void emplace_group_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const typename detail::get_initialization<Component>::InitialValues&
        initial_values,
    Options&&... opts) noexcept {
  runner->template emplace_group_component_and_initialize<Component>(
      initial_values, std::forward<Options>(opts)...);
}

/// Emplaces a distributed nodegroup object into the
/// parallel component `Component`. The options `opts` are forwarded to be used
/// in a call to `detail::ForwardAllOptionsToDataBox::apply` Additionally, the
/// simple tags in the DataBox are initialized from the values set in
/// `initial_values`.
template <typename Component, typename... Options,
          typename Metavars = typename Component::metavariables,
          Requires<detail::has_initialization_phase_v<Metavars>> = nullptr>
void emplace_nodegroup_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<typename Component::metavariables>*>
        runner,
    const typename detail::get_initialization<Component>::InitialValues&
        initial_values,
    Options&&... opts) noexcept {
  runner->template emplace_nodegroup_component_and_initialize<Component>(
      initial_values, std::forward<Options>(opts)...);
}

// @{
/// Retrieves the DataBox with tags `TagsList` (omitting the `GlobalCache`
/// and `add_from_options` tags) from the parallel component `Component` with
/// index `array_index`.
template <typename Component, typename TagsList, typename Metavariables>
const auto& get_databox(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template mock_distributed_objects<Component>()
      .at(array_index)
      .template get_databox<TagsList>();
}

template <typename Component, typename TagsList, typename Metavariables>
auto& get_databox(const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
                  const typename Component::array_index& array_index) noexcept {
  return runner->template mock_distributed_objects<Component>()
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
  return runner.template mock_distributed_objects<Component>()
      .at(array_index)
      .template get_databox_tag<Tag>();
}

// @{
/// Returns the `InboxTag` from the parallel component `Component` with array
/// index `array_index`.
template <typename Component, typename InboxTag, typename Metavariables>
const auto& get_inbox_tag(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return tuples::get<InboxTag>(
      runner.template inboxes<Component>().at(array_index));
}

template <typename Component, typename InboxTag, typename Metavariables>
auto& get_inbox_tag(
    const gsl::not_null<MockRuntimeSystem<Metavariables>*> runner,
    const typename Component::array_index& array_index) noexcept {
  return tuples::get<InboxTag>(
      runner->template inboxes<Component>().at(array_index));
}
// @}

/// Returns `true` if the current DataBox of `Component` with index
/// `array_index` contains the tag `Tag`. If the tag is not contained, returns
/// `false`.
template <typename Component, typename Tag, typename Metavariables>
bool box_contains(const MockRuntimeSystem<Metavariables>& runner,
                  const typename Component::array_index& array_index) noexcept {
  return runner.template mock_distributed_objects<Component>()
      .at(array_index)
      .template box_contains<Tag>();
}

/// Returns `true` if the tag `Tag` can be retrieved from the current DataBox
/// of `Component` with index `array_index`.
template <typename Component, typename Tag, typename Metavariables>
bool tag_is_retrievable(
    const MockRuntimeSystem<Metavariables>& runner,
    const typename Component::array_index& array_index) noexcept {
  return runner.template mock_distributed_objects<Component>()
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
  return runner.template mock_distributed_objects<Component>()
      .at(array_index)
      .get_terminate();
}

/// Returns the GlobalCache of `Component` with index `array_index`.
template <typename Component, typename Metavariables, typename ArrayIndex>
Parallel::GlobalCache<Metavariables>& cache(
    MockRuntimeSystem<Metavariables>& runner,
    const ArrayIndex& array_index) noexcept {
  return runner.template mock_distributed_objects<Component>()
      .at(array_index)
      .cache();
}

/// Returns a vector of all the indices of the Components in the
/// ComponentList that have queued simple actions, for a particular
/// array_index.
template <typename ComponentList, typename MockRuntimeSystem,
          typename ArrayIndex>
std::vector<size_t> indices_of_components_with_queued_simple_actions(
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
template<typename Component>
struct ArrayIndicesTag {
  using type = std::vector<typename Component::array_index>;
};

template <typename ComponentList>
using array_indices_tags =
    tmpl::transform<ComponentList, tmpl::bind<ArrayIndicesTag, tmpl::_1>>;

template <typename ComponentList>
using array_indices_for_each_component =
    tuples::tagged_tuple_from_typelist<array_indices_tags<ComponentList>>;
}  // namespace detail
/// \endcond

/// Returns a vector of array_indices for each Component.  The vector
/// is filled with only those array_indices for which there are queued
/// simple actions.
template <typename ComponentList, typename MockRuntimeSystem>
auto array_indices_with_queued_simple_actions(
    const gsl::not_null<MockRuntimeSystem*> runner) noexcept
    -> detail::array_indices_for_each_component<ComponentList> {
  detail::array_indices_for_each_component<ComponentList> result;
  tmpl::for_each<ComponentList>([&](auto component) noexcept {
    using Component = typename decltype(component)::type;
    using array_indices_tag = detail::ArrayIndicesTag<Component>;
    auto& result_this_component = tuples::get<array_indices_tag>(result);
    for (auto& [index, mock_distributed_object] :
         runner->template mock_distributed_objects<Component>()) {
      if (not mock_distributed_object.is_simple_action_queue_empty()) {
        result_this_component.push_back(index);
      }
    }
  });
  return result;
}

/// Given the output of `array_indices_with_queued_simple_actions`, returns
/// the total number of array indices that have queued simple actions.
template <typename ComponentList>
size_t number_of_elements_with_queued_simple_actions(
    const detail::array_indices_for_each_component<ComponentList>&
        array_indices) noexcept {
  size_t num_queued_actions = 0;
  tmpl::for_each<ComponentList>([&](auto component) noexcept {
    using Component = typename decltype(component)::type;
    using array_indices_tag = detail::ArrayIndicesTag<Component>;
    num_queued_actions += tuples::get<array_indices_tag>(array_indices).size();
  });
  return num_queued_actions;
}

/// Invokes the next queued action on a random Component on a random
/// array_index of that component.  `array_indices` is the thing returned
/// by `array_indices_with_queued_simple_actions`
template <typename ComponentList, typename MockRuntimeSystem,
          typename Generator>
void invoke_random_queued_simple_action(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const gsl::not_null<Generator*> generator,
    const detail::array_indices_for_each_component<ComponentList>&
        array_indices) noexcept {
  // Choose one queued action at random.
  const size_t num_queued_actions =
      number_of_elements_with_queued_simple_actions<ComponentList>(
          array_indices);
  std::uniform_int_distribution<size_t> ran(0, num_queued_actions - 1);
  const size_t index_of_action_to_invoke = ran(*generator);

  // Invoke the chosen queued action.
  size_t queued_action_count = 0;
  tmpl::for_each<ComponentList>([&runner, &array_indices,
                                 &index_of_action_to_invoke,
                                 &queued_action_count](
                                    auto component) noexcept {
    using Component = typename decltype(component)::type;
    using array_indices_tag = detail::ArrayIndicesTag<Component>;
    const size_t num_queued_actions_this_comp =
        tuples::get<array_indices_tag>(array_indices).size();
    if (index_of_action_to_invoke >= queued_action_count and
        index_of_action_to_invoke <
            queued_action_count + num_queued_actions_this_comp) {
      runner->template invoke_queued_simple_action<Component>(
          tuples::get<array_indices_tag>(array_indices)
              .at(index_of_action_to_invoke - queued_action_count));
    }
    queued_action_count += num_queued_actions_this_comp;
  });
}
}  // namespace ActionTesting
