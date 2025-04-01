// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/GridCenters.hpp"
#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Systems/GridCenters.hpp"
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
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace control_system {
namespace {
using both_centers = control_system::measurements::BothNSCenters;
using GridCenters = control_system::Systems::GridCenters<2, both_centers>;
static_assert(tt::assert_conforms_to_v<
              GridCenters, control_system::protocols::ControlSystem>);
using measurement_queue = GridCenters::MeasurementQueue;

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
using replace_these_simple_actions_mock_component =
    tmpl::list<::Actions::UpdateMessageQueue<
        measurement_queue, control_system::UpdateControlSystem<GridCenters>,
        QueueTags::Center<::domain::ObjectLabel::A, Frame::Grid>,
        QueueTags::Center<::domain::ObjectLabel::B, Frame::Grid>,
        QueueTags::Center<::domain::ObjectLabel::A, Frame::Inertial>,
        QueueTags::Center<::domain::ObjectLabel::B, Frame::Inertial>>>;
using with_these_simple_actions_mock_component =
    tmpl::list<MockUpdateMessageQueue<
        measurement_queue, control_system::UpdateControlSystem<GridCenters>,
        QueueTags::Center<::domain::ObjectLabel::A, Frame::Grid>,
        QueueTags::Center<::domain::ObjectLabel::B, Frame::Grid>,
        QueueTags::Center<::domain::ObjectLabel::A, Frame::Inertial>,
        QueueTags::Center<::domain::ObjectLabel::B, Frame::Inertial>>>;

template <typename Metavariables>
struct MockComponent
    : public TestHelpers::control_system::MockControlComponent<Metavariables,
                                                               GridCenters> {
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

void test_grid_centers_process_measurement() {
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
       tnsr::I<double, 3, Frame::Grid>{std::array{-16.0, 0.0, 0.0}},
       tnsr::I<double, 3, Frame::Grid>{std::array{16.0, 0.0, 0.0}},
       std::unordered_map<std::string, std::string>{}},
      {creator.functions_of_time(),
       control_system::Tags::MeasurementTimescales::type{}}};

  ActionTesting::emplace_singleton_component<component>(make_not_null(&runner),
                                                        {0}, {0});

  auto& cache = ActionTesting::cache<component>(runner, 0);

  const LinkedMessageId<double> id{0.0, std::nullopt};
  const std::array grid_centers_a{-16.0, 0.0, 0.0};
  const std::array grid_centers_b{16.0, 0.0, 0.0};
  const std::array inertial_centers_a{-14.0, 2.0, 0.0};
  const std::array inertial_centers_b{14.0, 2.0, 0.0};

  GridCenters::process_measurement::apply(
      both_centers::FindTwoCenters{}, grid_centers_a, grid_centers_b,
      inertial_centers_a, inertial_centers_b, cache, id);

  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  CHECK(message_queue_call_count == 0);
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);

  CHECK(ActionTesting::is_simple_action_queue_empty<component>(runner, 0));
  CHECK(message_queue_call_count == 1);
}

void test_names() {
  CHECK(pretty_type::name<GridCenters>() == "GridCenters");
  CHECK(*GridCenters::component_name(0, 6) == "A_x");
  CHECK(*GridCenters::component_name(1, 6) == "A_y");
  CHECK(*GridCenters::component_name(2, 6) == "A_z");
  CHECK(*GridCenters::component_name(3, 6) == "B_x");
  CHECK(*GridCenters::component_name(4, 6) == "B_y");
  CHECK(*GridCenters::component_name(5, 6) == "B_z");
}

SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.GridCenters",
                  "[ControlSystem][Unit]") {
  test_names();
  test_grid_centers_process_measurement();
}
}  // namespace
}  // namespace control_system
