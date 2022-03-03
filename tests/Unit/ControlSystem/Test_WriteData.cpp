// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/WriteData.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {
namespace {
struct LabelA {};

constexpr size_t total_components = 3;

struct FakeControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static constexpr size_t deriv_order = 2;
  static std::string name() {
    return pretty_type::short_name<FakeControlSystem>();
  }
  static std::string component_name(const size_t i) {
    return i == 0 ? "Foo" : (i == 1 ? "Bar" : "Baz");
  }
  using measurement = control_system::TestHelpers::Measurement<LabelA>;
  using simple_tags = tmpl::list<>;
  struct process_measurement {
    using argument_tags = tmpl::list<>;
  };
};

struct FakeQuatControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static constexpr size_t deriv_order = 3;
  static std::string name() {
    return pretty_type::short_name<FakeQuatControlSystem>();
  }
  static std::string component_name(const size_t i) { return get_output(i); }
  using measurement = control_system::TestHelpers::Measurement<LabelA>;
  using simple_tags = tmpl::list<>;
  struct process_measurement {
    using argument_tags = tmpl::list<>;
  };
};

template <typename Metavariables>
struct MockObserverWriter {
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using const_global_cache_tags =
      tmpl::list<observers::Tags::ReductionFileName>;

  using initialize_action_list =
      tmpl::list<::Actions::SetupDataBox,
                 observers::Actions::InitializeWriter<Metavariables>>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::WriteData, tmpl::list<>>>;
};

template <typename Metavariables, typename ControlSystem>
struct MockControlComponent {
  using component_being_mocked = ControlComponent<Metavariables, ControlSystem>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;
  using array_index = int;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;

  using register_action =
      tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
          control_system::Registration<ControlSystem>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Register, register_action>>;
};

struct TestMetavars {
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::list<WriterHelper>>;

  enum class Phase { Initialization, Register, WriteData, Exit };

  using component_list =
      tmpl::list<MockObserverWriter<TestMetavars>,
                 MockControlComponent<TestMetavars, FakeControlSystem>,
                 MockControlComponent<TestMetavars, FakeQuatControlSystem>>;
};

using FoTPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

template <typename ControlSystem>
void check_written_data(
    const h5::H5File<h5::AccessType::ReadOnly>& read_file,
    const std::vector<double>& times, FoTPtr& fot,
    const std::vector<std::array<DataVector, ControlSystem::deriv_order>>&
        q_and_derivs,
    const std::vector<DataVector>& control_signal) {
  std::array<DataVector, 3> func_and_2_derivs{};
  for (size_t component_num = 0; component_num < total_components;
       component_num++) {
    // per file checks
    const auto& dataset = read_file.get<h5::Dat>(
        "/ControlSystems/" + ControlSystem::name() + "/" +
        ControlSystem::component_name(component_num));
    const Matrix data = dataset.get_data();
    const std::vector<std::string>& legend = dataset.get_legend();
    // Check legend is correct
    for (size_t i = 0; i < legend.size(); i++) {
      CHECK(legend[i] == WriterHelper::legend[i]);
    }
    CHECK(data.rows() == times.size());

    // per time per file checks
    for (size_t time_num = 0; time_num < times.size(); time_num++) {
      const double time = times[time_num];
      const auto* const quat_func_of_time =
          dynamic_cast<const domain::FunctionsOfTime::QuaternionFunctionOfTime<
              ControlSystem::deriv_order>*>(fot.get());
      if (quat_func_of_time == nullptr) {
        func_and_2_derivs = fot->func_and_2_derivs(time);
      } else {
        func_and_2_derivs = quat_func_of_time->angle_func_and_2_derivs(time);
      }

      // check time is correct
      size_t offset = 0;
      CHECK(data(time_num, offset) == time);
      ++offset;

      // check values for lambda are correct
      for (size_t deriv_num = 0; deriv_num < 3; deriv_num++) {
        CHECK(data(time_num, offset) ==
              func_and_2_derivs[deriv_num][component_num]);
        ++offset;
      }

      // check control error and derivs are correct
      for (size_t deriv_num = 0; deriv_num < 2; deriv_num++) {
        CHECK(data(time_num, offset) ==
              q_and_derivs[time_num][deriv_num][component_num]);
        ++offset;
      }

      // check control signal
      CHECK(data(time_num, offset) == control_signal[time_num][component_num]);
    }
  }
}

SPECTRE_TEST_CASE("Unit.ControlSystem.WriteData", "[Unit][ControlSystem]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  constexpr size_t deriv_order = FakeControlSystem::deriv_order;
  constexpr size_t quat_deriv_order = FakeQuatControlSystem::deriv_order;
  using observer = MockObserverWriter<TestMetavars>;
  using control_comp = MockControlComponent<TestMetavars, FakeControlSystem>;
  using quat_control_comp =
      MockControlComponent<TestMetavars, FakeQuatControlSystem>;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  const std::string filename = "WriteDataTest_Output";

  // clean up just in case
  if (file_system::check_if_file_exists(filename + ".h5")) {
    file_system::rm(filename + ".h5", true);
  }

  // set up runner and stuff
  ActionTesting::MockRuntimeSystem<TestMetavars> runner{{filename}};
  runner.set_phase(TestMetavars::Phase::Initialization);
  ActionTesting::emplace_nodegroup_component<observer>(make_not_null(&runner));
  ActionTesting::emplace_singleton_component<control_comp>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0});
  ActionTesting::emplace_singleton_component<quat_control_comp>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0});
  auto& cache = ActionTesting::cache<observer>(runner, 0);

  // the initialization actions
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<observer>(make_not_null(&runner), 0);
  }
  // Register singletons with observer writer
  runner.set_phase(TestMetavars::Phase::Register);
  ActionTesting::next_action<control_comp>(make_not_null(&runner), 0);
  CHECK(ActionTesting::number_of_queued_simple_actions<observer>(runner, 0) ==
        1);
  ActionTesting::invoke_queued_simple_action<observer>(make_not_null(&runner),
                                                       0);
  ActionTesting::next_action<quat_control_comp>(make_not_null(&runner), 0);
  CHECK(ActionTesting::number_of_queued_simple_actions<observer>(runner, 0) ==
        1);
  ActionTesting::invoke_queued_simple_action<observer>(make_not_null(&runner),
                                                       0);

  runner.set_phase(TestMetavars::Phase::WriteData);

  // set up data to write
  FoTPtr normal_fot = std::make_unique<
      domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>>(
      0.0,
      std::array<DataVector, deriv_order + 1>{
          {make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0}),
           make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0}),
           make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0})}},
      5.0);

  FoTPtr quat_fot = std::make_unique<
      domain::FunctionsOfTime::QuaternionFunctionOfTime<quat_deriv_order>>(
      0.0, std::array<DataVector, 1>{{{1.0, 0.0, 0.0, 0.0}}},
      std::array<DataVector, quat_deriv_order + 1>{
          {make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0}),
           make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0}),
           make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0}),
           make_with_random_values<DataVector>(
               make_not_null(&gen), dist, DataVector{total_components, 0.0})}},
      5.0);

  const std::vector<double> times{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  std::vector<std::array<DataVector, deriv_order>> normal_q_and_derivs{
      times.size()};
  std::vector<std::array<DataVector, quat_deriv_order>> quat_q_and_derivs{
      times.size()};
  std::vector<DataVector> normal_control_signals{times.size()};
  std::vector<DataVector> quat_control_signals{times.size()};

  // write some data
  for (size_t i = 0; i < times.size(); i++) {
    const double time = times[i];
    for (size_t j = 0; j < deriv_order; j++) {
      gsl::at(normal_q_and_derivs[i], j) = make_with_random_values<DataVector>(
          make_not_null(&gen), dist, DataVector{total_components, 0.0});
    }
    for (size_t j = 0; j < quat_deriv_order; j++) {
      gsl::at(quat_q_and_derivs[i], j) = make_with_random_values<DataVector>(
          make_not_null(&gen), dist, DataVector{total_components, 0.0});
    }
    normal_control_signals[i] = make_with_random_values<DataVector>(
        make_not_null(&gen), dist, DataVector{total_components, 0.0});
    quat_control_signals[i] = make_with_random_values<DataVector>(
        make_not_null(&gen), dist, DataVector{total_components, 0.0});

    write_components_to_disk<FakeControlSystem>(time, cache, normal_fot,
                                                normal_q_and_derivs[i],
                                                normal_control_signals[i]);
    write_components_to_disk<FakeQuatControlSystem>(
        time, cache, quat_fot, quat_q_and_derivs[i], quat_control_signals[i]);

    // 3 for each control system
    size_t num_threaded_actions =
        ActionTesting::number_of_queued_threaded_actions<observer>(runner, 0);
    CHECK(num_threaded_actions == total_components * 2);
    for (size_t j = 0; j < total_components * 2; j++) {
      ActionTesting::invoke_queued_threaded_action<observer>(
          make_not_null(&runner), 0);
    }

    num_threaded_actions =
        ActionTesting::number_of_queued_threaded_actions<observer>(runner, 0);
    CHECK(not num_threaded_actions);
  }

  // scoped to close file
  {
    h5::H5File<h5::AccessType::ReadOnly> read_file{filename + ".h5"};
    check_written_data<FakeControlSystem>(read_file, times, normal_fot,
                                          normal_q_and_derivs,
                                          normal_control_signals);
    check_written_data<FakeQuatControlSystem>(
        read_file, times, quat_fot, quat_q_and_derivs, quat_control_signals);
  }

  // clean up now that we are done
  if (file_system::check_if_file_exists(filename + ".h5")) {
    file_system::rm(filename + ".h5", true);
  }
}
}  // namespace
}  // namespace control_system
