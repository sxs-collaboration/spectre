// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "ControlSystem/Actions/Initialization.hpp"
#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct LabelA {};

template <typename Label, typename Measurement>
struct MockControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return pretty_type::short_name<Label>(); }
  using measurement = Measurement;
  static constexpr size_t deriv_order = 2;
  using simple_tags = tmpl::list<>;
};

using mock_control_sys =
    MockControlSystem<LabelA, control_system::TestHelpers::Measurement<LabelA>>;

template <typename Metavariables>
struct MockControlComponent {
  using array_index = int;
  using component_being_mocked =
      ControlComponent<Metavariables, mock_control_sys>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;

  using simple_tags =
      tmpl::list<control_system::Tags::ControlSystemInputs<MockControlSystem<
                     LabelA, control_system::TestHelpers::Measurement<LabelA>>>,
                 control_system::Tags::Averager<2>,
                 control_system::Tags::TimescaleTuner,
                 control_system::Tags::ControlSystemName,
                 control_system::Tags::Controller<2>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>,
                 control_system::Actions::Initialize<Metavariables,
                                                     mock_control_sys>>>>;
};

struct MockMetavars {
  using component_list = tmpl::list<MockControlComponent<MockMetavars>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Initialization",
                  "[Unit][ControlSystem]") {
  using component = MockControlComponent<MockMetavars>;
  using tags = typename component::simple_tags;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<MockMetavars>;
  Averager<2> averager{0.5, true};
  Averager<2> averager_empty{};
  TimescaleTuner tuner{{1.0}, 10.0, 0.1, 2.0, 0.1, 1.01, 0.99};
  TimescaleTuner tuner_empty{};
  Controller<2> controller{0.3};
  Controller<2> controller_empty{};
  std::string controlsys_name{"LabelA"};
  std::string controlsys_name_empty{};

  tuples::tagged_tuple_from_typelist<tags> init_tuple{
      control_system::OptionHolder<mock_control_sys>{averager, controller,
                                                     tuner},
      averager_empty, tuner_empty, controlsys_name_empty, controller_empty};

  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_singleton_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, init_tuple);

  // This has to come after we create the component so that it isn't initialized
  // immediately
  controller.assign_time_between_updates(1.0);

  const auto& box_averager =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::Averager<2>>(runner,
                                                                        0);
  const auto& box_tuner =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::TimescaleTuner>(
          runner, 0);
  const auto& box_controller =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::Controller<2>>(
          runner, 0);
  const auto& box_controlsys_name =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::ControlSystemName>(
          runner, 0);

  // Check that things haven't been initialized
  CHECK(box_averager != averager);
  CHECK(box_tuner != tuner);
  CHECK(box_controlsys_name != controlsys_name);
  // We don't check the controller now because one of its members is
  // inititalized with signaling_NaN() so comparing now would throw an FPE.

  // Now initialize everything
  runner.next_action<component>(0);

  // Check that box values are same as expected values
  CHECK(box_averager == averager);
  CHECK(box_tuner == tuner);
  CHECK(box_controlsys_name == controlsys_name);
  CHECK(box_controller == controller);
}
