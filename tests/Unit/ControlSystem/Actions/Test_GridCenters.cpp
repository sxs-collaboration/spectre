// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "ControlSystem/Actions/GridCenters.hpp"
#include "ControlSystem/Tags/IsActiveMap.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTime,
                 control_system::Tags::IsActiveMap>;
  using simple_tags = db::AddSimpleTags<::domain::Tags::Element<3>,
                                        Tags::TimeStepId, ::Tags::Time>;
  using compute_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<control_system::Actions::SwitchGridRotationToSettle>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
};

struct UpdateFot {
  static void apply(gsl::not_null<domain::FunctionsOfTimeMap*> current_fots,
                    const domain::FunctionsOfTimeMap& new_fots) {
    *current_fots = clone_unique_ptrs(new_fots);
  }
};

void test_action() {
  register_classes_with_charm<
      domain::FunctionsOfTime::PiecewisePolynomial<3>,
      domain::FunctionsOfTime::QuaternionFunctionOfTime<3>,
      domain::FunctionsOfTime::SettleToConstantQuaternion>();
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using component = Component<Metavariables>;

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["GridCenters"] =
      std::make_unique<::domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          0.0,
          std::array<DataVector, 4>{{{16.0, 0.0, 0.0, -16.0, 0.0, 0.0},
                                     {-0.001, 0.0, 0.0, 0.002, 0.0, 0.0},
                                     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}},
          std::numeric_limits<double>::max());
  const double initial_omega_z = 0.01;
  auto init_func_rotation = make_array<4, DataVector>(DataVector{3, 0.0});
  init_func_rotation[1][2] = initial_omega_z;
  auto init_quaternion = make_array<1, DataVector>(DataVector{4, 0.0});
  init_quaternion[0][0] = 1.0;
  functions_of_time["Rotation"] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          0.0, init_quaternion, init_func_rotation, 1.0e5);
  std::unordered_map<std::string, bool> is_active_map{{"GridCenters", true},
                                                      {"Rotation", true}};

  MockRuntimeSystem runner{
      {control_system::DisableRotationWhen{6.0, 60.0}},
      {clone_unique_ptrs(functions_of_time), is_active_map}};

  const ElementId<3> element_id_zero{0, {}};
  const ElementId<3> element_id_nonzero{1, {}};
  const std::vector element_ids{element_id_zero, element_id_nonzero};
  TimeStepId time_step_id{true, 1, Time{Slab{0.0, 1.0e3}, Rational{2, 256}}};
  const double time = time_step_id.step_time().value();

  ActionTesting::emplace_array_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, element_id_zero,
      {::Element<3>{element_id_zero, {}}, time_step_id, time});
  ActionTesting::emplace_array_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, element_id_nonzero,
      {::Element<3>{element_id_nonzero, {}}, time_step_id, time});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
      make_not_null(&runner), element_id_zero,
      Catch::Matchers::ContainsSubstring(
          "Expected to be at a Slab boundary when changing the "
          "Rotation function of time to a SettleToConstant")));

  // No error if not on element zero
  CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                       element_id_nonzero));

  time_step_id = TimeStepId{true, 1, Time{Slab{0.0, 1.0e3}, Rational{0, 256}}};

  for (const auto& id : element_ids) {
    db::mutate<::Tags::TimeStepId, ::Tags::Time>(
        [&time_step_id](const gsl::not_null<TimeStepId*> box_time_step_id,
                        const gsl::not_null<double*> box_time) {
          *box_time_step_id = time_step_id;
          *box_time = box_time_step_id->step_time().value();
        },
        make_not_null(&ActionTesting::get_databox<component>(
            make_not_null(&runner), id)));
  }

  CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
      make_not_null(&runner), element_id_zero,
      Catch::Matchers::ContainsSubstring(
          "Disabling the rotation control system should happen when the "
          "separation is less than or equal to ")));

  // No error if not on element zero
  CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                       element_id_nonzero));

  time_step_id =
      TimeStepId{true, 2, Time{Slab{12.0e3, 17.0e3}, Rational{0, 256}}};

  for (const auto& id : element_ids) {
    db::mutate<::Tags::TimeStepId, ::Tags::Time>(
        [&time_step_id](const gsl::not_null<TimeStepId*> box_time_step_id,
                        const gsl::not_null<double*> box_time) {
          *box_time_step_id = time_step_id;
          *box_time = box_time_step_id->step_time().value();
        },
        make_not_null(&ActionTesting::get_databox<component>(
            make_not_null(&runner), id)));
  }

  // Test that a non-zero Element doesn't change the function of time.
  CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                       element_id_nonzero));
  for (const auto& id : element_ids) {
    CHECK(get<control_system::Tags::IsActiveMap>(
              ActionTesting::cache<component>(runner, id))
              .at("Rotation"));
    REQUIRE(get<domain::Tags::FunctionsOfTime>(
                ActionTesting::cache<component>(runner, id))
                .at("Rotation") != nullptr);
    CHECK(dynamic_cast<
              const domain::FunctionsOfTime::SettleToConstantQuaternion* const>(
              get<domain::Tags::FunctionsOfTime>(
                  ActionTesting::cache<component>(runner, id))
                  .at("Rotation")
                  .get()) == nullptr);
  }

  // Remove things we need from FunctionsOfTime map to test ERRORs
  auto& cache = ActionTesting::cache<component>(runner, element_id_zero);
  Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFot>(
      cache, [&functions_of_time]() {
        auto local_fots = clone_unique_ptrs(functions_of_time);
        local_fots.erase("GridCenters");
        return local_fots;
      }());
  CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
      make_not_null(&runner), element_id_zero,
      Catch::Matchers::ContainsSubstring("There is no function of time named "
                                         "'GridCenters', which is required ")));
  Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFot>(
      cache, [&functions_of_time]() {
        auto local_fots = clone_unique_ptrs(functions_of_time);
        local_fots.erase("Rotation");
        return local_fots;
      }());
  CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
      make_not_null(&runner), element_id_zero,
      Catch::Matchers::ContainsSubstring("There is no function of time named "
                                         "'Rotation', which means that ")));
  Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFot>(
      cache, [&functions_of_time]() {
        auto local_fots = clone_unique_ptrs(functions_of_time);
        return local_fots;
      }());

  // Check that the zero Element changes the function of time.
  REQUIRE(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         element_id_zero));
  for (const auto& id : element_ids) {
    CHECK_FALSE(get<control_system::Tags::IsActiveMap>(
                    ActionTesting::cache<component>(runner, id))
                    .at("Rotation"));
    REQUIRE(get<domain::Tags::FunctionsOfTime>(
                ActionTesting::cache<component>(runner, id))
                .at("Rotation") != nullptr);
    CHECK(dynamic_cast<
              const domain::FunctionsOfTime::SettleToConstantQuaternion* const>(
              get<domain::Tags::FunctionsOfTime>(
                  ActionTesting::cache<component>(runner, id))
                  .at("Rotation")
                  .get()) != nullptr);
  }
}

void test_tags() {
  TestHelpers::db::test_simple_tag<control_system::Tags::DisableRotationWhen>(
      "DisableRotationWhen");

  const control_system::DisableRotationWhen disable_opts =
      serialize_and_deserialize(
          control_system::Tags::DisableRotationWhen::create_from_options(
              TestHelpers::test_option_tag<
                  control_system::OptionTags::DisableRotationWhen>(
                  "DisableAtSeparation: 6\n"
                  "RotationDecayTimescale: 40\n")));
  // REQUIRE because the action tests will fail if this doesn't work.
  REQUIRE(disable_opts.disable_at_separation == 6.0);
  REQUIRE(disable_opts.rotation_decay_timescale == 40.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Actions.GridCenters",
                  "[Unit][ControlSystem]") {
  test_tags();
  test_action();
}
