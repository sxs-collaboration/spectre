// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <vector>

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
  static std::string name() {
    return pretty_type::short_name<Label>();
  }
  using measurement = Measurement;
  using simple_tags = tmpl::list<control_system::Tags::ControlSystemName>;
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

  using simple_tags = tmpl::list<control_system::Tags::Averager<2>,
                                 control_system::Tags::TimescaleTuner,
                                 control_system::Tags::ControlSystemName,
                                 control_system::Tags::Controller<2>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>,
                 Initialization::Actions::InitializeControlSystem<
                     Metavariables, mock_control_sys>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
};

struct MockMetavars {
  using component_list = tmpl::list<MockControlComponent<MockMetavars>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Initialization",
                  "[Unit][ControlSystem]") {
  using component = MockControlComponent<MockMetavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<MockMetavars>;
  Averager<2> averager{0.5, true};
  TimescaleTuner tuner{
      std::vector<double>{1.0}, 10.0, 0.1, 2.0, 0.1, 1.01, 0.99};
  Controller<2> controller{};
  std::string controlsys_name{"LabelA"};

  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_singleton_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0},
      {averager, tuner, controlsys_name, controller});

  const auto& box_averager =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::Averager<2>>(runner,
                                                                        0);
  const auto& box_tuner =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::TimescaleTuner>(
          runner, 0);
  const auto& box_controlsys_name =
      ActionTesting::get_databox_tag<component,
                                     control_system::Tags::ControlSystemName>(
          runner, 0);

  CHECK(box_averager == averager);
  CHECK(box_tuner == tuner);
  CHECK(box_controlsys_name == controlsys_name);
  // We don't check the controller because it currently doesn't have an
  // operator==. It doesn't have an operator== because it currently stores no
  // member data so it would be pointless. Once it stores member data, an
  // operator== should be added, and this test should be updated.
}
