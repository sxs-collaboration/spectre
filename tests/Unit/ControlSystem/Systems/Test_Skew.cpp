// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Skew.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Systems/Skew.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace control_system {
namespace {
using both_horizons = control_system::measurements::BothHorizons;
using skew = control_system::Systems::Skew<2, both_horizons>;
static_assert(
    tt::assert_conforms_to_v<skew, control_system::protocols::ControlSystem>);
using measurement_queue = skew::MeasurementQueue;

using all_tags = measurement_queue::type::queue_tags_list;

size_t message_queue_call_count = 0;  // NOLINT

template <typename LinkedMessageQueueTag, typename Processor,
          typename... QueueTags>
struct MockUpdateMessageQueue {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<typename LinkedMessageQueueTag::type::IdType>&
      /*id_and_previous*/,
      typename QueueTags::type... /*message*/) {
    ++message_queue_call_count;
  }
};

// The Nvidia compiler crashes if we define these lists inside the MockComponent
// struct.
using replace_these_simple_actions_mock_component = tmpl::transform<
    all_tags,
    tmpl::bind<::Actions::UpdateMessageQueue, tmpl::pin<measurement_queue>,
               tmpl::pin<control_system::UpdateControlSystem<skew>>, tmpl::_1>>;
using with_these_simple_actions_mock_component = tmpl::transform<
    all_tags,
    tmpl::bind<MockUpdateMessageQueue, tmpl::pin<measurement_queue>,
               tmpl::pin<control_system::UpdateControlSystem<skew>>, tmpl::_1>>;

template <typename Metavariables>
struct MockComponent
    : public control_system::TestHelpers::MockControlComponent<Metavariables,
                                                               skew> {
  using replace_these_simple_actions =
      replace_these_simple_actions_mock_component;
  using with_these_simple_actions = with_these_simple_actions_mock_component;
};

struct Metavars {
  using observed_reduction_data_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, control_system::Tags::Verbosity>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 control_system::Tags::MeasurementTimescales>;
  using component_list =
      tmpl::list<MockComponent<Metavars>,
                 ::TestHelpers::observers::MockObserverWriter<Metavars>>;
};

void test_skew_process_measurement() {
  domain::FunctionsOfTime::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::creators::register_derived_with_charm();
  const domain::creators::Brick creator{
      std::array{-10.0, -10.0, -10.0},
      std::array{10.0, 10.0, 10.0},
      std::array{0_st, 0_st, 0_st},
      std::array{8_st, 8_st, 8_st},
      std::array{false, false, false},
      {},
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0})};

  using component = MockComponent<Metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavars>;
  MockRuntimeSystem runner{
      {creator.create_domain(), ::Verbosity::Silent, 4, false,
       std::unordered_map<std::string, bool>{},
       tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}},
       tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}},
       std::unordered_map<std::string, std::string>{}},
      {creator.functions_of_time(),
       control_system::Tags::MeasurementTimescales::type{}}};

  ActionTesting::emplace_singleton_component<component>(make_not_null(&runner),
                                                        {0}, {0});

  auto& cache = ActionTesting::cache<component>(runner, 0);

  // The data here doesn't matter except for the excision surface because it's
  // the only one that is used in any calculations. And even then, the only
  // restriction is that is must be contained within our domain above.
  const LinkedMessageId<double> id{0.0, std::nullopt};
  const ylm::Strahlkorper<Frame::Distorted> horizon_a{
      4_st, 1.0, std::array{0.0, 0.0, 0.0}};
  const ylm::Strahlkorper<Frame::Distorted> horizon_b{
      4_st, 2.0, std::array{0.0, 0.0, 0.0}};

  skew::process_measurement::apply(
      both_horizons::FindHorizon<domain::ObjectLabel::A>{}, horizon_a, cache,
      id);

  // We only check that the proper number of actions have been called.
  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  CHECK(message_queue_call_count == 0);
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);

  skew::process_measurement::apply(
      both_horizons::FindHorizon<domain::ObjectLabel::B>{}, horizon_b, cache,
      id);

  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  CHECK(message_queue_call_count == 1);
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);

  CHECK(ActionTesting::is_simple_action_queue_empty<component>(runner, 0));
  CHECK(message_queue_call_count == 2);
}

void test_names() {
  CHECK(pretty_type::name<skew>() == "Skew");
  CHECK(*skew::component_name(0, 2) == "Y");
  CHECK(*skew::component_name(1, 2) == "Z");
}

SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.Skew", "[ControlSystem][Unit]") {
  test_names();
  test_skew_process_measurement();
}
}  // namespace
}  // namespace control_system
