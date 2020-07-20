// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"         // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"      // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace TestObservers_detail;

template <observers::TypeOfObservation TypeOfObservation>
void check_observer_registration() {
  using metavariables = Metavariables<TypeOfObservation>;
  using obs_component = observer_component<metavariables>;
  using obs_writer = observer_writer_component<metavariables>;
  using element_comp = element_component<metavariables, TypeOfObservation>;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<obs_component>(&runner, 0);
  ActionTesting::next_action<obs_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<obs_writer>(&runner, 0);
  ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);

  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    ActionTesting::emplace_component<element_comp>(&runner, id);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::RegisterWithObservers);

  CHECK(
      ActionTesting::get_databox_tag<obs_component,
                                     observers::Tags::NumberOfEvents>(runner, 0)
          .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          obs_component, observers::Tags::ReductionArrayComponentIds>(runner, 0)
          .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_component, observers::Tags::VolumeArrayComponentIds>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::TensorData>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_component, observers::Tags::ReductionObserversContributed>(
            runner, 0)
            .empty());

  CHECK(ActionTesting::get_databox_tag<obs_writer, observers::Tags::TensorData>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::VolumeObserversRegistered>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::VolumeObserversContributed>(runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          obs_writer, observers::Tags::ReductionObserversRegistered>(runner, 0)
          .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::ReductionObserversRegisteredNodes>(
            runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          obs_writer, observers::Tags::ReductionObserversContributed>(runner, 0)
          .empty());

  // Register elements
  for (const auto& id : element_ids) {
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterSenderWithSelf that was called on the
    // observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
    if (TypeOfObservation != observers::TypeOfObservation::Volume) {
      // Invoke the simple_action
      // RegisterReductionContributorWithObserverWriter.
      ActionTesting::invoke_queued_simple_action<obs_writer>(
          make_not_null(&runner), 0);
    }
    if (TypeOfObservation != observers::TypeOfObservation::Reduction) {
      // Invoke the simple_action RegisterVolumeContributorWithObserverWriter.
      ActionTesting::invoke_queued_simple_action<obs_writer>(
          make_not_null(&runner), 0);
    }
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  // Test registration occurred as expected
  CHECK(
      ActionTesting::get_databox_tag<obs_component,
                                     observers::Tags::NumberOfEvents>(runner, 0)
          .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          obs_component, observers::Tags::ReductionArrayComponentIds>(runner, 0)
          .size() == (TypeOfObservation == observers::TypeOfObservation::Volume
                          ? size_t{0}
                          : element_ids.size()));
  CHECK(ActionTesting::get_databox_tag<
            obs_component, observers::Tags::VolumeArrayComponentIds>(runner, 0)
            .size() ==
        (TypeOfObservation == observers::TypeOfObservation::Reduction
             ? size_t{0}
             : element_ids.size()));
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::TensorData>(runner, 0)
            .empty());
  for (const auto& id : element_ids) {
    CHECK(ActionTesting::get_databox_tag<
              obs_component, observers::Tags::ReductionArrayComponentIds>(
              runner, 0)
              .count(observers::ArrayComponentId(
                  std::add_pointer_t<element_comp>{nullptr},
                  Parallel::ArrayIndex<ElementId<2>>(ElementId<2>(id)))) ==
          (TypeOfObservation == observers::TypeOfObservation::Volume ? 0 : 1));
    CHECK(
        ActionTesting::get_databox_tag<
            obs_component, observers::Tags::VolumeArrayComponentIds>(runner, 0)
            .count(observers::ArrayComponentId(
                std::add_pointer_t<element_comp>{nullptr},
                Parallel::ArrayIndex<ElementId<2>>(ElementId<2>(id)))) ==
        (TypeOfObservation == observers::TypeOfObservation::Reduction ? 0 : 1));
  }
  if (TypeOfObservation != observers::TypeOfObservation::Volume) {
    CHECK(ActionTesting::get_databox_tag<
              obs_writer, observers::Tags::ReductionObserversRegisteredNodes>(
              runner, 0)
              .empty());
    const auto hash =
        observers::ObservationId(
            3., typename TestObservers_detail::RegisterThisObsType<
                    TypeOfObservation>::ElementObservationType{})
            .observation_type_hash();
    CHECK(ActionTesting::get_databox_tag<
              obs_writer, observers::Tags::ReductionObserversRegistered>(runner,
                                                                         0)
              .at(hash)
              .size() == 1);
  }
  if (TypeOfObservation != observers::TypeOfObservation::Reduction) {
    const auto hash =
        observers::ObservationId(
            3., typename TestObservers_detail::RegisterThisObsType<
                    TypeOfObservation>::ElementObservationType{})
            .observation_type_hash();
    CHECK(ActionTesting::get_databox_tag<
              obs_writer, observers::Tags::VolumeObserversRegistered>(runner, 0)
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
