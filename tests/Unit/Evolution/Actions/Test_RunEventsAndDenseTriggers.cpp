// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PrimVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

class TestTrigger : public DenseTrigger {
 public:
  TestTrigger() = default;
  explicit TestTrigger(CkMigrateMessage* const msg) noexcept
      : DenseTrigger(msg) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestTrigger);  // NOLINT
#pragma GCC diagnostic pop

  // All triggers are evaluated once at the start of the evolution, so
  // we have to handle that call and set up for triggering at the
  // interesting time.
  TestTrigger(const double init_time, const double trigger_time,
              const bool is_ready_arg, const bool is_triggered) noexcept
      : init_time_(init_time),
        trigger_time_(trigger_time),
        is_ready_(is_ready_arg),
        is_triggered_(is_triggered) {}

  using is_triggered_argument_tags = tmpl::list<Tags::Time>;
  Result is_triggered(const double time) const noexcept {
    if (time == init_time_) {
      return {false, trigger_time_};
    }
    CHECK(time == trigger_time_);
    return {is_triggered_, (trigger_time_ > init_time_ ? 1.0 : -1.0) *
                               std::numeric_limits<double>::infinity()};
  }

  using is_ready_argument_tags = tmpl::list<Tags::Time>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/,
                const double time) const noexcept {
    if (time == init_time_) {
      return true;
    }
    CHECK(time == trigger_time_);
    return is_ready_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    DenseTrigger::pup(p);
    p | init_time_;
    p | trigger_time_;
    p | is_ready_;
    p | is_triggered_;
  }

 private:
  double init_time_ = std::numeric_limits<double>::signaling_NaN();
  double trigger_time_ = std::numeric_limits<double>::signaling_NaN();
  bool is_ready_ = false;
  bool is_triggered_ = false;
};

PUP::able::PUP_ID TestTrigger::my_PUP_ID = 0;  // NOLINT

struct TestEvent : public Event {
  TestEvent() = default;
  explicit TestEvent(CkMigrateMessage* const /*msg*/) noexcept {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestEvent);  // NOLINT
#pragma GCC diagnostic pop

  explicit TestEvent(const bool needs_evolved_variables)
      : needs_evolved_variables_(needs_evolved_variables) {}

  using argument_tags = tmpl::list<Tags::Time, Var, PrimVar>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(const double time, Var::type var, PrimVar::type prim_var,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/) const noexcept {
    calls.emplace_back(time, std::move(var), std::move(prim_var));
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const noexcept {
    // We use the triggers to control readiness for this test.
    return true;
  }

  bool needs_evolved_variables() const noexcept override {
    return needs_evolved_variables_;
  }

  template <bool HasPrimitiveAndConservativeVars>
  static void check_calls(
      const std::vector<std::tuple<double, Variables<tmpl::list<Var>>,
                                   Variables<tmpl::list<PrimVar>>>>&
          expected) noexcept {
    CAPTURE(get_output(expected));
    CAPTURE(get_output(calls));
    REQUIRE(calls.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      CHECK(std::get<0>(calls[i]) == std::get<0>(expected[i]));
      CHECK(std::get<1>(calls[i]) == get<Var>(std::get<1>(expected[i])));
      if (HasPrimitiveAndConservativeVars) {
        CHECK(std::get<2>(calls[i]) == get<PrimVar>(std::get<2>(expected[i])));
      }
    }
    calls.clear();
  }

 private:
  bool needs_evolved_variables_ = false;

  static std::vector<std::tuple<double, Var::type, PrimVar::type>> calls;
};

std::vector<std::tuple<double, Var::type, PrimVar::type>> TestEvent::calls{};

PUP::able::PUP_ID TestEvent::my_PUP_ID = 0;  // NOLINT

struct PrimFromCon {
  using return_tags = tmpl::list<PrimVar>;
  using argument_tags = tmpl::list<Var>;
  static void apply(const gsl::not_null<Scalar<DataVector>*> prim,
                    const Scalar<DataVector>& con) noexcept {
    get(*prim) = -get(con);
  }
};

struct BoundaryCorrection;

struct BoundaryCorrectionBase {
  BoundaryCorrectionBase() = default;
  BoundaryCorrectionBase(const BoundaryCorrectionBase&) = default;
  BoundaryCorrectionBase(BoundaryCorrectionBase&&) = default;
  BoundaryCorrectionBase& operator=(const BoundaryCorrectionBase&) = default;
  BoundaryCorrectionBase& operator=(BoundaryCorrectionBase&&) = default;
  virtual ~BoundaryCorrectionBase() = default;
  using creatable_classes = tmpl::list<BoundaryCorrection>;
};

struct BoundaryCorrection final : BoundaryCorrectionBase {
  using dg_package_field_tags = tmpl::list<Var>;
  static void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> correction,
      const Scalar<DataVector>& /*interior*/,
      const Scalar<DataVector>& exterior,
      const dg::Formulation /*formulation*/) noexcept {
    *correction = exterior;
  }
};

template <bool HasPrimitiveAndConservativeVars>
struct System {
  static constexpr size_t volume_dim = 1;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  static constexpr bool has_primitive_and_conservative_vars = false;
  using boundary_correction_base = BoundaryCorrectionBase;
};

template <>
struct System<true> {
  static constexpr size_t volume_dim = 1;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  using primitive_variables_tag = Tags::Variables<tmpl::list<PrimVar>>;
  static constexpr bool has_primitive_and_conservative_vars = true;
  using boundary_correction_base = BoundaryCorrectionBase;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using variables_tag = typename metavariables::system::variables_tag;
  // Not unconditionally defined in the system to make sure the action
  // only accesses it when it should.
  using primitives_tag = Tags::Variables<tmpl::list<PrimVar>>;
  using initialization_tags = tmpl::append<
      tmpl::list<Tags::TimeStepId, Tags::TimeStep, Tags::Time,
                 evolution::Tags::PreviousTriggerTime, variables_tag,
                 primitives_tag, Tags::HistoryEvolvedVariables<variables_tag>,
                 evolution::Tags::EventsAndDenseTriggers>,
      tmpl::conditional_t<
          Metavariables::local_time_stepping,
          tmpl::list<domain::Tags::Mesh<1>, evolution::dg::Tags::MortarMesh<1>,
                     evolution::dg::Tags::MortarSize<1>,
                     Tags::Next<Tags::TimeStepId>, dg::Tags::Formulation,
                     evolution::dg::Tags::NormalCovectorAndMagnitude<1>,
                     evolution::Tags::BoundaryCorrection<
                         typename metavariables::system>,
                     evolution::dg::Tags::MortarDataHistory<
                         1, typename db::add_tag_prefix<::Tags::dt,
                                                        variables_tag>::type>,
                     evolution::dg::Tags::MortarNextTemporalId<1>>,
          tmpl::list<>>>;

  using prim_from_con = tmpl::conditional_t<
      metavariables::system::has_primitive_and_conservative_vars, PrimFromCon,
      void>;
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<1>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<
          evolution::Actions::RunEventsAndDenseTriggers<prim_from_con>>>>;
};

template <bool HasPrimitiveAndConservativeVars, bool LocalTimeStepping>
struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using system = System<HasPrimitiveAndConservativeVars>;
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<
      tmpl::conditional_t<LocalTimeStepping, LtsTimeStepper, TimeStepper>>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DenseTrigger, tmpl::list<TestTrigger>>,
                  tmpl::pair<Event, tmpl::list<TestEvent>>>;
  };
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Metavariables>
bool run_if_ready(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner) noexcept {
  using component = Component<Metavariables>;
  using system = typename Metavariables::system;
  using variables_tag = typename system::variables_tag;
  const auto get_prims = [&runner]() noexcept {
    if constexpr (system::has_primitive_and_conservative_vars) {
      return ActionTesting::get_databox_tag<
          component, typename system::primitive_variables_tag>(*runner, 0);
    } else {
      (void)runner;
      return 0;
    }
  };

  const auto time_before =
      ActionTesting::get_databox_tag<component, ::Tags::Time>(*runner, 0);
  const auto vars_before =
      ActionTesting::get_databox_tag<component, variables_tag>(*runner, 0);
  const auto prims_before = get_prims();
  const bool was_ready =
      ActionTesting::next_action_if_ready<component>(runner, 0);
  const auto time_after =
      ActionTesting::get_databox_tag<component, ::Tags::Time>(*runner, 0);
  const auto vars_after =
      ActionTesting::get_databox_tag<component, variables_tag>(*runner, 0);
  const auto prims_after = get_prims();
  CHECK(time_before == time_after);
  CHECK(vars_before == vars_after);
  CHECK(prims_before == prims_after);
  return was_ready;
}

template <bool HasPrimitiveAndConservativeVars>
void test(const bool time_runs_forward) noexcept {
  using metavars = Metavariables<HasPrimitiveAndConservativeVars, false>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using component = Component<metavars>;
  using system = typename metavars::system;
  using variables_tag = typename system::variables_tag;
  using primitives_tag = typename component::primitives_tag;
  using VarsType = typename variables_tag::type;
  using PrimsType = typename primitives_tag::type;
  using DtVarsType =
      typename db::add_tag_prefix<::Tags::dt, variables_tag>::type;
  using History = TimeSteppers::History<VarsType, DtVarsType>;

  const Slab slab(0.0, 4.0);
  const TimeStepId time_step_id(time_runs_forward, 0,
                                slab.start() + slab.duration() / 2);
  const TimeDelta exact_step_size =
      (time_runs_forward ? 1 : -1) * slab.duration() / 4;
  const double start_time = time_step_id.step_time().value();
  const double step_size = exact_step_size.value();
  const double step_center = start_time + 0.5 * step_size;
  const VarsType initial_vars{1, 8.0};
  const DtVarsType deriv_vars{1, 1.0};
  const PrimsType unset_primitives{1, 1234.5};

  const auto set_up_component =
      [&deriv_vars, &exact_step_size, &initial_vars, &start_time, &time_step_id,
       &unset_primitives](
          const gsl::not_null<MockRuntimeSystem*> runner,
          const std::vector<std::tuple<double, bool, bool, bool>>&
              triggers) noexcept {
        History history(1);
        history.insert(time_step_id, deriv_vars);
        history.most_recent_value() = initial_vars;

        evolution::EventsAndDenseTriggers::ConstructionType
            events_and_dense_triggers{};
        for (auto [trigger_time, is_ready, is_triggered,
                   needs_evolved_variables] : triggers) {
          events_and_dense_triggers.emplace(
              std::make_unique<TestTrigger>(start_time, trigger_time, is_ready,
                                            is_triggered),
              make_vector<std::unique_ptr<Event>>(
                  std::make_unique<TestEvent>(needs_evolved_variables)));
        }

        ActionTesting::emplace_array_component<component>(
            runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
            time_step_id, exact_step_size, start_time, std::optional<double>{},
            initial_vars, unset_primitives, std::move(history),
            evolution::EventsAndDenseTriggers(
                std::move(events_and_dense_triggers)));
        ActionTesting::set_phase(runner, metavars::Phase::Testing);
      };

  // Tests start here

  // Nothing should happen in self-start
  const auto check_self_start = [&set_up_component, &start_time, &step_size](
                                    const bool trigger_is_ready) noexcept {
    // This isn't a valid time for the trigger to reschedule to (it is
    // in the past), but the triggers should be completely ignored in
    // this check.
    const double invalid_time = start_time - step_size;

    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(
        &runner, {{invalid_time, trigger_is_ready, trigger_is_ready, false}});
    {
      auto& box = ActionTesting::get_databox<component, tmpl::list<>>(
          make_not_null(&runner), 0);
      db::mutate<Tags::TimeStepId>(
          make_not_null(&box),
          [](const gsl::not_null<TimeStepId*> id) noexcept {
            *id = TimeStepId(id->time_runs_forward(), -1, id->step_time());
          });
    }
    CHECK(run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>({});
  };
  check_self_start(true);
  check_self_start(false);

  // No triggers
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {});
    CHECK(run_if_ready(make_not_null(&runner)));
  }

  // Triggers too far in the future
  const auto check_not_reached = [&set_up_component, &start_time, &step_size](
                                     const bool trigger_is_ready) noexcept {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{start_time + 1.5 * step_size, trigger_is_ready,
                                trigger_is_ready, false}});
    CHECK(run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>({});
  };
  check_not_reached(true);
  check_not_reached(false);

  // Trigger isn't ready
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, false, true, false}});
    CHECK(not run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>({});
  }

  // Variables not needed
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, true, true, false}});
    CHECK(run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>(
        {{step_center, initial_vars, unset_primitives}});
  }

  // Variables needed
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, true, true, true}});
    CHECK(run_if_ready(make_not_null(&runner)));
    const VarsType dense_var = initial_vars + 0.5 * step_size * deriv_vars;
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>(
        {{step_center, dense_var, -dense_var}});
  }

  // Missing dense output data
  const auto check_missing_dense_data = [&initial_vars, &set_up_component,
                                         &step_center, &time_step_id,
                                         &unset_primitives](
                                            const bool data_needed) noexcept {
    MockRuntimeSystem runner{{std::make_unique<TimeSteppers::RungeKutta3>()}};
    set_up_component(&runner, {{step_center, true, true, data_needed}});
    {
      auto& box = ActionTesting::get_databox<component, tmpl::list<>>(
          make_not_null(&runner), 0);
      db::mutate<Tags::HistoryEvolvedVariables<variables_tag>>(
          make_not_null(&box),
          [&initial_vars,
           &time_step_id](const gsl::not_null<History*> history) noexcept {
            *history = History(3);
            history->insert(TimeStepId(time_step_id.time_runs_forward(), 0,
                                       time_step_id.step_time(), 1,
                                       time_step_id.step_time()),
                            {1, 1.0});
            history->most_recent_value() = initial_vars;
          });
    }
    CHECK(run_if_ready(make_not_null(&runner)));
    if (data_needed) {
      TestEvent::check_calls<HasPrimitiveAndConservativeVars>({});
    } else {
      // If we don't need the data, it shouldn't matter whether it is missing.
      TestEvent::check_calls<HasPrimitiveAndConservativeVars>(
          {{step_center, initial_vars, unset_primitives}});
    }
  };
  check_missing_dense_data(true);
  check_missing_dense_data(false);

  // Multiple triggers
  {
    const double second_trigger = start_time + 0.75 * step_size;
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, true, true, false},
                               {second_trigger, true, true, false}});
    CHECK(run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>(
        {{step_center, initial_vars, unset_primitives},
         {second_trigger, initial_vars, unset_primitives}});
  }
}

template <bool HasPrimitiveAndConservativeVars>
void test_lts(const bool time_runs_forward) noexcept {
  using metavars = Metavariables<HasPrimitiveAndConservativeVars, true>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using component = Component<metavars>;
  using system = typename metavars::system;
  using variables_tag = typename system::variables_tag;
  using primitives_tag = typename component::primitives_tag;
  using VarsType = typename variables_tag::type;
  using PrimsType = typename primitives_tag::type;
  using DtVarsType =
      typename db::add_tag_prefix<::Tags::dt, variables_tag>::type;
  using History = TimeSteppers::History<VarsType, DtVarsType>;

  const Slab slab(0.0, 4.0);
  const TimeStepId time_step_id(time_runs_forward, 0,
                                slab.start() + slab.duration() / 2);
  const TimeDelta exact_step_size =
      (time_runs_forward ? 1 : -1) * slab.duration() / 4;
  const TimeStepId next_time_step_id(
      time_runs_forward, 0, time_step_id.step_time() + exact_step_size);
  const double start_time = time_step_id.step_time().value();
  const double step_size = exact_step_size.value();
  const TimeStepId half_time_step_id(
      time_runs_forward, 0, time_step_id.step_time() + exact_step_size / 2);
  const TimeStepId quarter_time_step_id(
      time_runs_forward, 0, time_step_id.step_time() + exact_step_size / 4);
  const double half_time = half_time_step_id.step_time().value();
  const VarsType initial_vars{2, 8.0};
  const DtVarsType deriv_vars{2, 1.0};
  const PrimsType unset_primitives{2, 1234.5};

  const double first_correction = 2.0;
  const double second_correction = 3.0;

  const std::pair<Direction<1>, ElementId<1>> neighbor{Direction<1>::upper_xi(),
                                                       0};

  const auto set_up_component =
      [&deriv_vars, &exact_step_size, &first_correction, &half_time,
       &initial_vars, &neighbor, &next_time_step_id, &quarter_time_step_id,
       &start_time, &time_step_id,
       &unset_primitives](const gsl::not_null<MockRuntimeSystem*> runner,
                          const bool needs_evolved_variables) noexcept {
        const Mesh<1> mesh(2, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto);
        History history(1);
        history.insert(time_step_id, deriv_vars);
        history.most_recent_value() = initial_vars;

        typename evolution::dg::Tags::MortarDataHistory<1, DtVarsType>::type
            mortar_history{};
        auto& neighbor_history = mortar_history[neighbor];
        neighbor_history.integration_order(1);

        // We skip setting some data fields that we don't use for the test.
        evolution::dg::MortarData<1> local_data{};
        local_data.insert_local_mortar_data(time_step_id, Mesh<0>{}, {});
        local_data.insert_local_face_normal_magnitude(
            Scalar<DataVector>{{{{1.0}}}});
        evolution::dg::MortarData<1> remote_data{};
        remote_data.insert_neighbor_mortar_data(time_step_id, Mesh<0>{},
                                                {first_correction});

        neighbor_history.local_insert(time_step_id, std::move(local_data));
        neighbor_history.remote_insert(time_step_id, std::move(remote_data));

        evolution::EventsAndDenseTriggers::ConstructionType
            events_and_dense_triggers{};
        events_and_dense_triggers.emplace(
            std::make_unique<TestTrigger>(start_time, half_time, true, true),
            make_vector<std::unique_ptr<Event>>(
                std::make_unique<TestEvent>(needs_evolved_variables)));

        ActionTesting::emplace_array_component<component>(
            runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
            time_step_id, exact_step_size, start_time, std::optional<double>{},
            initial_vars, unset_primitives, std::move(history),
            evolution::EventsAndDenseTriggers(
                std::move(events_and_dense_triggers)),
            mesh,
            evolution::dg::Tags::MortarMesh<1>::type{{neighbor, Mesh<0>{}}},
            evolution::dg::Tags::MortarSize<1>::type{{neighbor, {}}},
            next_time_step_id,
            // Only passed to the boundary correction.  Our test
            // correction ignores it.
            dg::Tags::Formulation::type{},
            // Only used in GTS mode, but fetched unconditionally for
            // control-flow convenience.
            evolution::dg::Tags::NormalCovectorAndMagnitude<1>::type{},
            std::unique_ptr<BoundaryCorrectionBase>(
                std::make_unique<BoundaryCorrection>()),
            std::move(mortar_history),
            evolution::dg::Tags::MortarNextTemporalId<1>::type{
                {neighbor, quarter_time_step_id}});
        ActionTesting::set_phase(runner, metavars::Phase::Testing);
      };

  // Tests start here

  // Variables not needed
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, false);
    CHECK(run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>(
        {{half_time, initial_vars, unset_primitives}});
  }

  // Variables needed
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, true);
    REQUIRE_FALSE(run_if_ready(make_not_null(&runner)));
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>({});

    ActionTesting::get_inbox_tag<
        component,
        evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<1>>(
        make_not_null(&runner), 0)
        .insert(
            {quarter_time_step_id,
             {{neighbor,
               {Mesh<0>{}, {}, {{second_correction}}, next_time_step_id}}}});
    CHECK(run_if_ready(make_not_null(&runner)));
    VarsType dense_var = initial_vars + 0.5 * step_size * deriv_vars;
    get(get<Var>(dense_var))[1] -=
        (0.5 * (first_correction + second_correction)) * (0.5 * step_size);
    TestEvent::check_calls<HasPrimitiveAndConservativeVars>(
        {{half_time, dense_var, -dense_var}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.RunEventsAndDenseTriggers",
                  "[Unit][Evolution][Actions]") {
  Parallel::register_classes_with_charm<TimeSteppers::AdamsBashforthN,
                                        TimeSteppers::RungeKutta3>();
  // Same lists for true and false
  Parallel::register_factory_classes_with_charm<Metavariables<false, false>>();
  test<false>(true);
  test<false>(false);
  test<true>(true);
  test<true>(false);

  test_lts<false>(true);
  test_lts<false>(false);
  test_lts<true>(true);
  test_lts<true>(false);
}
