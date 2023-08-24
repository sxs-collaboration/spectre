// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "ControlSystem/Measurements/CharSpeed.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Systems/Size.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
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
using size = control_system::Systems::Size<domain::ObjectLabel::None, 2>;
static_assert(
    tt::assert_conforms_to_v<size, control_system::protocols::ControlSystem>);
using measurement_queue = size::MeasurementQueue;
using char_speed =
    control_system::measurements::CharSpeed<domain::ObjectLabel::None>;

using all_tags = measurement_queue::type::queue_tags_list;

static size_t message_queue_call_count = 0;

template <typename QueueTag, typename LinkedMessageQueueTag, typename Processor>
struct MockUpdateMessageQueue {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<typename LinkedMessageQueueTag::type::IdType>&
      /*id_and_previous*/,
      typename QueueTag::type /*message*/) {
    ++message_queue_call_count;
  }
};

template <typename Metavariables>
struct MockComponent
    : public control_system::TestHelpers::MockControlComponent<Metavariables,
                                                               size> {
  using replace_these_simple_actions = tmpl::transform<
      all_tags,
      tmpl::bind<::Actions::UpdateMessageQueue, tmpl::_1,
                 tmpl::pin<measurement_queue>,
                 tmpl::pin<control_system::UpdateControlSystem<size>>>>;
  using with_these_simple_actions = tmpl::transform<
      all_tags,
      tmpl::bind<MockUpdateMessageQueue, tmpl::_1, tmpl::pin<measurement_queue>,
                 tmpl::pin<control_system::UpdateControlSystem<size>>>>;
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

void test_size_process_measurement() {
  domain::FunctionsOfTime::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::creators::register_derived_with_charm();
  std::unique_ptr<DomainCreator<3>> creator =
      std::make_unique<domain::creators::Brick>(
          std::array{-10.0, -10.0, -10.0}, std::array{10.0, 10.0, 10.0},
          std::array{0_st, 0_st, 0_st}, std::array{8_st, 8_st, 8_st},
          std::array{false, false, false},
          std::make_unique<
              domain::creators::time_dependence::UniformTranslation<3>>(
              0.0, std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0}));

  using component = MockComponent<Metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavars>;
  MockRuntimeSystem runner{
      {creator->create_domain(), ::Verbosity::Silent, 4, false,
       std::unordered_map<std::string, bool>{},
       tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}},
       tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}}},
      {creator->functions_of_time(),
       control_system::Tags::MeasurementTimescales::type{}}};

  ActionTesting::emplace_singleton_component<component>(make_not_null(&runner),
                                                        {0}, {0});

  auto& cache = ActionTesting::cache<component>(runner, 0);

  // The data here doesn't matter except for the excision surface because it's
  // the only one that is used in any calculations. And even then, the only
  // restriction is that is must be contained within our domain above.
  const LinkedMessageId<double> id{0.0, std::nullopt};
  const Scalar<DataVector> zero_scalar{};
  const tnsr::I<DataVector, 3, Frame::Distorted> zero_tnsr_I{};
  const tnsr::ii<DataVector, 3, Frame::Distorted> zero_tnsr_ii{};
  const tnsr::II<DataVector, 3, Frame::Distorted> zero_tnsr_II{};
  const Strahlkorper<Frame::Distorted> horizon{};
  const Strahlkorper<Frame::Grid> excision{2, 2.0, std::array{0.0, 0.0, 0.0}};

  size::process_measurement::apply<Metavars>(
      char_speed::Excision{}, excision, zero_scalar, zero_tnsr_I, zero_tnsr_ii,
      zero_tnsr_II, cache, id);

  // We only check that the proper number of actions have been called.
  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  CHECK(message_queue_call_count == 0);
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);

  size::process_measurement::apply<Metavars>(char_speed::Horizon{}, horizon,
                                             horizon, cache, id);

  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  CHECK(message_queue_call_count == 1);
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);

  CHECK(ActionTesting::is_simple_action_queue_empty<component>(runner, 0));
  CHECK(message_queue_call_count == 2);
}

void test_names() {
  CHECK(pretty_type::name<size>() == "Size");
  CHECK(*size::component_name(0, 1) == "Size");
  CHECK(*size::component_name(1, 1) == "Size");
}

SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.Size", "[ControlSystem][Unit]") {
  test_names();
  test_size_process_measurement();
}
}  // namespace
}  // namespace control_system
