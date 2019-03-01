// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"         // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/IO/Observers/ObserverHelpers.hpp"

namespace {
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace TestObservers_detail;

template <observers::TypeOfObservation TypeOfObservation>
void check_observer_registration() {
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          Metavariables>::TupleOfMockDistributedObjects;
  using obs_component = observer_component<Metavariables>;
  using obs_writer = observer_writer_component<Metavariables>;
  using element_comp = element_component<Metavariables>;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using ObserverMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          obs_component>;
  using WriterMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          obs_writer>;
  using ElementMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          element_comp>;
  TupleOfMockDistributedObjects dist_objects{};
  tuples::get<ObserverMockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<obs_component>{});
  tuples::get<WriterMockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<obs_writer>{});

  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    tuples::get<ElementMockDistributedObjectsTag>(dist_objects)
        .emplace(ElementIndex<2>{id},
                 ActionTesting::MockDistributedObject<element_comp>{});
  }

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {}, std::move(dist_objects)};

  runner.simple_action<obs_component,
                       observers::Actions::Initialize<Metavariables>>(0);
  runner.simple_action<obs_writer,
                       observers::Actions::InitializeWriter<Metavariables>>(0);
  // Test initial state
  const auto& observer_box =
      runner.template algorithms<obs_component>()
          .at(0)
          .template get_databox<typename obs_component::initial_databox>();
  CHECK(db::get<observers::Tags::NumberOfEvents>(observer_box).empty());
  CHECK(db::get<observers::Tags::ReductionArrayComponentIds>(observer_box)
            .empty());
  CHECK(
      db::get<observers::Tags::VolumeArrayComponentIds>(observer_box).empty());
  CHECK(db::get<observers::Tags::TensorData>(observer_box).empty());
  CHECK(db::get<observers::Tags::ReductionObserversContributed>(observer_box)
            .empty());

  // We should test writer_box too.
  const auto& writer_box =
      runner.template algorithms<obs_writer>()
          .at(0)
          .template get_databox<typename obs_writer::initial_databox>();
  CHECK(db::get<observers::Tags::TensorData>(writer_box).empty());
  CHECK(
      db::get<observers::Tags::VolumeObserversRegistered>(writer_box).empty());
  CHECK(
      db::get<observers::Tags::VolumeObserversContributed>(writer_box).empty());
  CHECK(db::get<observers::Tags::ReductionObserversRegistered>(writer_box)
            .empty());
  CHECK(db::get<observers::Tags::ReductionObserversRegisteredNodes>(writer_box)
            .empty());
  CHECK(db::get<observers::Tags::ReductionObserversContributed>(writer_box)
            .empty());

  // Register elements
  for (const auto& id : element_ids) {
    runner.simple_action<
        element_comp,
        observers::Actions::RegisterWithObservers<TypeOfObservation>>(
        id, observers::ObservationId(
                3., typename Metavariables::element_observation_type{}));
    // Invoke the simple_action RegisterSenderWithSelf that was called on the
    // observer component by the RegisterWithObservers action.
    runner.invoke_queued_simple_action<obs_component>(0);
    if (TypeOfObservation != observers::TypeOfObservation::Volume) {
      // Invoke the simple_action
      // RegisterReductionContributorWithObserverWriter.
      runner.invoke_queued_simple_action<obs_writer>(0);
    }
    if (TypeOfObservation != observers::TypeOfObservation::Reduction) {
      // Invoke the simple_action RegisterVolumeContributorWithObserverWriter.
      runner.invoke_queued_simple_action<obs_writer>(0);
    }
  }

  // Test registration occurred as expected
  CHECK(db::get<observers::Tags::NumberOfEvents>(observer_box).empty());
  CHECK(db::get<observers::Tags::ReductionArrayComponentIds>(observer_box)
            .size() ==
        (TypeOfObservation == observers::TypeOfObservation::Volume
             ? size_t{0}
             : element_ids.size()));
  CHECK(
      db::get<observers::Tags::VolumeArrayComponentIds>(observer_box).size() ==
      (TypeOfObservation == observers::TypeOfObservation::Reduction
           ? size_t{0}
           : element_ids.size()));
  CHECK(db::get<observers::Tags::TensorData>(observer_box).empty());
  for (const auto& id : element_ids) {
    CHECK(
        db::get<observers::Tags::ReductionArrayComponentIds>(observer_box)
            .count(observers::ArrayComponentId(
                std::add_pointer_t<element_comp>{nullptr},
                Parallel::ArrayIndex<ElementIndex<2>>(ElementIndex<2>(id)))) ==
        (TypeOfObservation == observers::TypeOfObservation::Volume ? 0 : 1));
    CHECK(
        db::get<observers::Tags::VolumeArrayComponentIds>(observer_box)
            .count(observers::ArrayComponentId(
                std::add_pointer_t<element_comp>{nullptr},
                Parallel::ArrayIndex<ElementIndex<2>>(ElementIndex<2>(id)))) ==
        (TypeOfObservation == observers::TypeOfObservation::Reduction ? 0 : 1));
  }
  if (TypeOfObservation != observers::TypeOfObservation::Volume) {
    CHECK(
        db::get<observers::Tags::ReductionObserversRegisteredNodes>(writer_box)
            .empty());
    const auto hash =
        observers::ObservationId(
            3., typename Metavariables::element_observation_type{})
            .observation_type_hash();
    CHECK(db::get<observers::Tags::ReductionObserversRegistered>(writer_box)
              .at(hash)
              .size() == 1);
  }
  if (TypeOfObservation != observers::TypeOfObservation::Reduction) {
    const auto hash =
        observers::ObservationId(
            3., typename Metavariables::element_observation_type{})
            .observation_type_hash();
    CHECK(db::get<observers::Tags::VolumeObserversRegistered>(writer_box)
              .at(hash)
              .size() == 1);
  }
}

SPECTRE_TEST_CASE("Unit.IO.Observers.RegisterElements", "[Unit][Observers]") {
  SECTION("Register as requiring reduction observer support") {
    check_observer_registration<observers::TypeOfObservation::Reduction>();
  }
  SECTION("Register as requiring volume observer support") {
    check_observer_registration<observers::TypeOfObservation::Volume>();
  }
  SECTION("Register as requiring both reduction and volume  observer support") {
    check_observer_registration<
        observers::TypeOfObservation::ReductionAndVolume>();
  }
}
}  // namespace
