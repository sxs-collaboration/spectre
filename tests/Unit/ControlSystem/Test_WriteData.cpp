// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
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
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
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
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return i == 0 ? std::optional<std::string>{"Foo"}
                  : (i == 1 ? std::optional<std::string>{"Bar"} : std::nullopt);
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
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = control_system::TestHelpers::Measurement<LabelA>;
  using simple_tags = tmpl::list<>;
  struct process_measurement {
    using argument_tags = tmpl::list<>;
  };
};

template <typename Metavariables, typename ControlSystem>
struct MockControlComponent {
  using component_being_mocked = ControlComponent<Metavariables, ControlSystem>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;
  using array_index = int;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct TestMetavars {
  using observed_reduction_data_tags = tmpl::list<>;

  using component_list =
      tmpl::list<::TestHelpers::observers::MockObserverWriter<TestMetavars>,
                 MockControlComponent<TestMetavars, FakeControlSystem>,
                 MockControlComponent<TestMetavars, FakeQuatControlSystem>>;
};

using FoTPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

template <typename ControlSystem, typename Metavariables>
void check_written_data(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
    const std::vector<double>& times, FoTPtr& fot,
    const std::vector<std::array<DataVector, 2>>& q_and_derivs) {
  std::array<DataVector, 3> func_and_2_derivs{};
  // This has to be the same as in control_system::write_components_to_disk
  const std::vector<std::string> compare_legend{
      "Time",         "Lambda",         "dtLambda",     "d2tLambda",
      "ControlError", "dtControlError", "ControlSignal"};

  auto& read_file = ActionTesting::get_databox_tag<
      ::TestHelpers::observers::MockObserverWriter<Metavariables>,
      ::TestHelpers::observers::MockReductionFileTag>(runner, 0);
  for (size_t component_num = 0; component_num < total_components;
       component_num++) {
    // per file checks
    const auto component_name_opt =
        ControlSystem::component_name(component_num, total_components);
    if constexpr (std::is_same_v<ControlSystem, FakeControlSystem>) {
      if (component_num == 2) {
        CHECK_FALSE(component_name_opt);
        continue;
      }
    }
    const auto& dataset = read_file.get_dat(
        "/ControlSystems/" + ControlSystem::name() + "/" + *component_name_opt);
    const Matrix& data = dataset.get_data();
    const std::vector<std::string>& legend = dataset.get_legend();
    // Check legend is correct
    for (size_t i = 0; i < legend.size(); i++) {
      CHECK(legend[i] == compare_legend[i]);
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
    }
  }
}

SPECTRE_TEST_CASE("Unit.ControlSystem.WriteData", "[Unit][ControlSystem]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  constexpr size_t deriv_order = FakeControlSystem::deriv_order;
  constexpr size_t quat_deriv_order = FakeQuatControlSystem::deriv_order;
  using observer = ::TestHelpers::observers::MockObserverWriter<TestMetavars>;
  using control_comp = MockControlComponent<TestMetavars, FakeControlSystem>;
  using quat_control_comp =
      MockControlComponent<TestMetavars, FakeQuatControlSystem>;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};

  // set up runner and stuff
  ActionTesting::MockRuntimeSystem<TestMetavars> runner{{}};
  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_nodegroup_component_and_initialize<observer>(
      make_not_null(&runner), {});
  ActionTesting::emplace_singleton_component<control_comp>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0});
  ActionTesting::emplace_singleton_component<quat_control_comp>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0});
  auto& cache = ActionTesting::cache<observer>(runner, 0);

  runner.set_phase(Parallel::Phase::Execute);

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
  std::vector<std::array<DataVector, 2>> normal_q_and_derivs{times.size()};
  std::vector<std::array<DataVector, 2>> quat_q_and_derivs{times.size()};

  // write some data
  for (size_t i = 0; i < times.size(); i++) {
    const double time = times[i];
    for (size_t j = 0; j < 2; j++) {
      gsl::at(normal_q_and_derivs[i], j) = make_with_random_values<DataVector>(
          make_not_null(&gen), dist, DataVector{total_components, 0.0});
      gsl::at(quat_q_and_derivs[i], j) = make_with_random_values<DataVector>(
          make_not_null(&gen), dist, DataVector{total_components, 0.0});
    }

    write_components_to_disk<FakeControlSystem>(time, cache, normal_fot,
                                                normal_q_and_derivs[i]);
    write_components_to_disk<FakeQuatControlSystem>(time, cache, quat_fot,
                                                    quat_q_and_derivs[i]);

    // 3 for one control system, 2 for the other (because of the nullopt)
    size_t num_threaded_actions =
        ActionTesting::number_of_queued_threaded_actions<observer>(runner, 0);
    const size_t expected_num_threaded_actions = 2 * total_components - 1;
    CHECK(num_threaded_actions == expected_num_threaded_actions);
    for (size_t j = 0; j < expected_num_threaded_actions; j++) {
      ActionTesting::invoke_queued_threaded_action<observer>(
          make_not_null(&runner), 0);
    }

    num_threaded_actions =
        ActionTesting::number_of_queued_threaded_actions<observer>(runner, 0);
    CHECK(not num_threaded_actions);
  }

  check_written_data<FakeControlSystem>(runner, times, normal_fot,
                                        normal_q_and_derivs);
  check_written_data<FakeQuatControlSystem>(runner, times, quat_fot,
                                            quat_q_and_derivs);
}
}  // namespace
}  // namespace control_system
