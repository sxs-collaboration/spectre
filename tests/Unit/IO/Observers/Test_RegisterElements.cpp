// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace TestObservers_detail;

template <observers::TypeOfObservation TypeOfObservation>
void check_observer_registration() {
  using registration_list =
      tmpl::list<observers::Actions::RegisterWithObservers<
                     RegisterObservers<TypeOfObservation>>,
                 Parallel::Actions::TerminatePhase>;

  using metavariables = Metavariables<registration_list>;
  using obs_component = observer_component<metavariables>;
  using obs_writer = observer_writer_component<metavariables>;
  using element_comp = element_component<metavariables, registration_list>;

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

  // Check observer component
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::ObservationsRegistered>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::ReductionsContributed>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_component, observers::Tags::ContributorsOfTensorData>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::TensorData>(runner, 0)
            .empty());

  // Check observer writer component
  CHECK(ActionTesting::get_databox_tag<obs_writer,
                                       observers::Tags::ObservationsRegistered>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_writer,
                                       observers::Tags::ReductionsContributed>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::ContributorsOfTensorData>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_writer, observers::Tags::TensorData>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::NodesExpectedToContributeReductions>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::NodeReductionsContributedForWriting>(
            runner, 0)
            .empty());

  const size_t number_of_obs_writer_actions =
      TypeOfObservation == observers::TypeOfObservation::Volume ? 1 : 2;

  // Register elements
  for (const auto& element_id : element_ids) {
    ActionTesting::next_action<element_comp>(make_not_null(&runner),
                                             element_id);
    // Invoke the simple_action RegisterContributorWithObserver that was called
    // on the observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
  }
  for (size_t j = 0; j < number_of_obs_writer_actions; ++j) {
    ActionTesting::invoke_queued_simple_action<obs_writer>(
        make_not_null(&runner), 0);
  }
  REQUIRE(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 0));
  REQUIRE(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));
  REQUIRE(
      ActionTesting::is_simple_action_queue_empty<obs_component>(runner, 0));
  REQUIRE(
      ActionTesting::is_threaded_action_queue_empty<obs_component>(runner, 0));

  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  // Test registration occurred as expected
  const observers::ObservationKey obs_id_key{"ElementObservationType"};
  std::unordered_map<observers::ObservationKey,
                     std::unordered_set<observers::ArrayComponentId>>
      expected_obs_ids{};
  for (const auto& id : element_ids) {
    expected_obs_ids[obs_id_key].insert(
        observers::ArrayComponentId{std::add_pointer_t<element_comp>{nullptr},
                                    Parallel::ArrayIndex<ElementId<2>>(id)});
  }
  std::unordered_map<observers::ObservationKey,
                     std::unordered_set<observers::ArrayComponentId>>
      expected_obs_writer_ids{};
  expected_obs_writer_ids[obs_id_key].insert(
      observers::ArrayComponentId{std::add_pointer_t<obs_component>{nullptr},
                                  Parallel::ArrayIndex<int>(0)});

  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::ObservationsRegistered>(
            runner, 0) == expected_obs_ids);
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::ReductionsContributed>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_component, observers::Tags::ContributorsOfTensorData>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::TensorData>(runner, 0)
            .empty());

  CHECK(ActionTesting::get_databox_tag<obs_writer,
                                       observers::Tags::ObservationsRegistered>(
            runner, 0) == expected_obs_writer_ids);
  CHECK(ActionTesting::get_databox_tag<obs_writer,
                                       observers::Tags::ReductionsContributed>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_writer, observers::Tags::ContributorsOfTensorData>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_writer, observers::Tags::TensorData>(
            runner, 0)
            .empty());
}

SPECTRE_TEST_CASE("Unit.IO.Observers.RegisterElements", "[Unit][Observers]") {
  // Tests RegisterWithObservers as well
  SECTION("Register as requiring reduction observer support") {
    check_observer_registration<observers::TypeOfObservation::Reduction>();
  }
  SECTION("Register as requiring volume observer support") {
    check_observer_registration<observers::TypeOfObservation::Volume>();
  }
}
}  // namespace
