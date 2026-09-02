// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryVariables.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/Conservative/UpdatePrimitives.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/UpdateU.hpp"
#include "Time/UpdateU.tpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

class TimeStepper;

namespace {
struct TemporalId {
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct Var : db::SimpleTag {
  using type = double;
};

struct ComplexVar : db::SimpleTag {
  using type = std::complex<double>;
};

struct PrimitiveVar : db::SimpleTag {
  using type = double;
};

constexpr size_t Dim = 1;

struct VolumeVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using BoundaryVar = evolution::dg::Tags::BoundaryValue<VolumeVar>;
using volume_tag = ::Tags::Variables<tmpl::list<VolumeVar>>;
using boundary_tag = ::Tags::BoundaryVariables<Dim, tmpl::list<BoundaryVar>>;

using dt_volume_tag = db::add_tag_prefix<Tags::dt, volume_tag>;
using dt_boundary_tag = db::add_tag_prefix<Tags::dt, boundary_tag>;

using volume_history_tag = Tags::HistoryEvolvedVariables<volume_tag>;
using boundary_history_tag = Tags::HistoryEvolvedVariables<boundary_tag>;

struct BoundaryVariablesSystem {
  static constexpr bool has_primitive_and_conservative_vars = false;
  using variables_tag = tmpl::list<volume_tag, boundary_tag>;
  // Only used by the test
  using test_primitive_variables_tags = tmpl::list<>;
};

// Initial values of the evolved variables; self-start must save and restore
// exactly these.
template <bool BoundaryVariables>
auto initial_values() {
  if constexpr (BoundaryVariables) {
    typename volume_tag::type volume{1};
    get(get<VolumeVar>(volume)) = -1.2;
    DirectionMap<Dim, size_t> points_per_direction{};
    points_per_direction[Direction<Dim>::lower_xi()] = 2;
    return tuples::TaggedTuple<volume_tag, boundary_tag>{
        std::move(volume),
        typename boundary_tag::type{std::move(points_per_direction), -0.8}};
  } else {
    return tuples::TaggedTuple<Var>{-1.};
  }
}

struct ComputeTimeDerivative {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if constexpr (evolution::dg::system_has_boundary_variables_v<
                      typename Metavariables::system>) {
      db::mutate<dt_volume_tag, dt_boundary_tag>(
          [](const gsl::not_null<typename dt_volume_tag::type*> dt_volume,
             const gsl::not_null<typename dt_boundary_tag::type*> dt_boundary,
             const typename volume_tag::type& volume,
             const typename boundary_tag::type& boundary) {
            get(get<Tags::dt<VolumeVar>>(*dt_volume)) =
                exp(get(get<VolumeVar>(volume)));
            for (auto& [direction, dt_vars] : dt_boundary->variables()) {
              get(get<Tags::dt<BoundaryVar>>(dt_vars)) = exp(
                  get(get<BoundaryVar>(boundary.variables().at(direction))));
            }
          },
          make_not_null(&box), db::get<volume_tag>(box),
          db::get<boundary_tag>(box));
    } else {
      using argument_tag = tmpl::conditional_t<
          Metavariables::system::has_primitive_and_conservative_vars,
          PrimitiveVar, Var>;
      db::mutate<Tags::dt<Var>>([](const gsl::not_null<double*> dt_var,
                                   const double var) { *dt_var = exp(var); },
                                make_not_null(&box),
                                db::get<argument_tag>(box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <bool HasPrimitives = false>
struct System {
  static constexpr bool has_primitive_and_conservative_vars = false;
  using variables_tag = Var;
  // Do not define primitive_variables_tag here.  Actions must work without it.

  // Only used by the test
  using test_primitive_variables_tags = tmpl::list<>;
};

template <>
struct System<true> {
  static constexpr bool has_primitive_and_conservative_vars = true;
  using variables_tag = Var;
  using primitive_variables_tag = PrimitiveVar;
  // Only used by the test
  using test_primitive_variables_tags = tmpl::list<primitive_variables_tag>;

  template <typename>
  struct primitive_from_conservative {
    using return_tags = tmpl::list<PrimitiveVar>;
    using argument_tags = tmpl::list<Var>;
    static void apply(const gsl::not_null<double*> prim, const double cons) {
      *prim = cons;
    }
  };
};

using history_tag = Tags::HistoryEvolvedVariables<Var>;
using additional_history_tag = Tags::HistoryEvolvedVariables<ComplexVar>;

template <typename Metavariables>
struct Component;

template <bool HasPrimitives = false, bool MultipleHistories = false,
          bool BoundaryVariables = false>
struct Metavariables {
  static constexpr bool has_primitives = HasPrimitives;
  static constexpr bool multiple_histories = MultipleHistories;
  using system = tmpl::conditional_t<BoundaryVariables, BoundaryVariablesSystem,
                                     System<HasPrimitives>>;
  using component_list = tmpl::list<Component<Metavariables>>;
  using ordered_list_of_primitive_recovery_schemes = tmpl::list<>;
  using temporal_id = TemporalId;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<Tags::ConcreteTimeStepper<TimeStepper>,
                 Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>;
  using simple_tags = tmpl::flatten<db::AddSimpleTags<
      typename metavariables::system::variables_tag,
      typename metavariables::system::test_primitive_variables_tags,
      typename Initialization::TimeStepperHistory<
          typename metavariables::system>::simple_tags,
      tmpl::conditional_t<Metavariables::multiple_histories,
                          additional_history_tag, tmpl::list<>>,
      Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
      Tags::Time, Tags::StepNumberWithinSlab,
      Tags::AdaptiveSteppingDiagnostics>>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;

  static constexpr bool has_primitives = Metavariables::has_primitives;

  using step_actions = tmpl::list<
      ComputeTimeDerivative,
      Actions::MutateApply<
          RecordTimeStepperData<typename metavariables::system>>,
      Actions::MutateApply<UpdateU<typename metavariables::system>>,
      Actions::MutateApply<CleanHistory<typename metavariables::system>>,
      tmpl::conditional_t<has_primitives, Actions::UpdatePrimitives,
                          tmpl::list<>>>;
  using action_list = tmpl::flatten<
      tmpl::list<SelfStart::self_start_procedure<
                     step_actions, typename metavariables::system>,
                 step_actions>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, action_list>>;
};

template <bool HasPrimitives = false, bool MultipleHistories = false,
          bool BoundaryVariables = false>
using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<
    Metavariables<HasPrimitives, MultipleHistories, BoundaryVariables>>;

template <bool HasPrimitives = false, bool MultipleHistories = false>
void emplace_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<HasPrimitives, MultipleHistories>*>
        runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    const double initial_value) {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<HasPrimitives, MultipleHistories>>>(
      runner, 0,
      {initial_value, 0., typename history_tag::type{1}, TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN(),
       uint64_t{0}, Tags::AdaptiveSteppingDiagnostics::type{}});
}

template <>
void emplace_component_and_initialize<true, false>(
    const gsl::not_null<MockRuntimeSystem<true, false>*> runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    const double initial_value) {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<true, false>>>(
      runner, 0,
      {initial_value, initial_value, 0., typename history_tag::type{1},
       TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN(),
       uint64_t{0}, Tags::AdaptiveSteppingDiagnostics::type{}});
}

template <>
void emplace_component_and_initialize<false, true>(
    const gsl::not_null<MockRuntimeSystem<false, true>*> runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    const double initial_value) {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<false, true>>>(
      runner, 0,
      {initial_value, 0., typename history_tag::type{1},
       typename additional_history_tag::type{1}, TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN(),
       uint64_t{0}, Tags::AdaptiveSteppingDiagnostics::type{}});
}

template <>
void emplace_component_and_initialize<true, true>(
    const gsl::not_null<MockRuntimeSystem<true, true>*> runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    const double initial_value) {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<true, true>>>(
      runner, 0,
      {initial_value, initial_value, 0., typename history_tag::type{1},
       typename additional_history_tag::type{1}, TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN(),
       uint64_t{0}, Tags::AdaptiveSteppingDiagnostics::type{}});
}

void emplace_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<false, false, true>*> runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    typename volume_tag::type volume, typename boundary_tag::type boundary) {
  // dt containers sized like the values; entries are overwritten before use.
  typename dt_boundary_tag::type dt_boundary{
      boundary.points_per_direction(),
      std::numeric_limits<double>::signaling_NaN()};
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<false, false, true>>>(
      runner, 0,
      {std::move(volume), std::move(boundary),
       typename dt_volume_tag::type{
           1, std::numeric_limits<double>::signaling_NaN()},
       std::move(dt_boundary), typename volume_history_tag::type{1},
       typename boundary_history_tag::type{1}, TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN(),
       uint64_t{0}, Tags::AdaptiveSteppingDiagnostics::type{}});
}

template <typename T>
struct is_initialize : std::false_type {};

template <typename System, template <typename> typename CacheTagPrefix>
struct is_initialize<SelfStart::Actions::Initialize<System, CacheTagPrefix>>
    : std::true_type {};

using not_self_start_action = std::negation<std::disjunction<
    is_initialize<tmpl::_1>,
    tt::is_a<SelfStart::Actions::CheckForCompletion, tmpl::_1>,
    std::is_same<SelfStart::Actions::CheckForOrderIncrease, tmpl::_1>,
    std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>>>;

// Run until an action satisfying the Stop metalambda is executed.
// Fail a REQUIRE if any action not passing the Whitelist metalambda
// is run first (as that would often lead to an infinite loop).
// Returns true if the last action jumped.
template <typename Stop, typename Whitelist, bool MultipleHistories,
          bool HasPrimitives, bool BoundaryVariables = false>
bool run_past(
    const gsl::not_null<
        MockRuntimeSystem<HasPrimitives, MultipleHistories, BoundaryVariables>*>
        runner) {
  using component = Component<
      Metavariables<HasPrimitives, MultipleHistories, BoundaryVariables>>;
  for (;;) {
    bool done = false;
    const size_t current_action =
        ActionTesting::get_next_action_index<component>(*runner, 0);
    size_t action_to_check = current_action;
    tmpl::for_each<typename component::action_list>(
        [&action_to_check, &done](const auto action) {
          using Action = tmpl::type_from<decltype(action)>;
          if (action_to_check-- == 0) {
            INFO(pretty_type::get_name<Action>());
            done = tmpl::apply<Stop, Action>::value;
            REQUIRE((done or tmpl::apply<Whitelist, Action>::value));
          }
        });
    ActionTesting::next_action<component>(runner, 0);
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Branch) false positive
    if (done) {
      // Self-start does not use the automatic algorithm looping, so
      // we don't have to check for the end.
      return current_action + 1 !=
             ActionTesting::get_next_action_index<component>(*runner, 0);
    }
  }
}

template <bool BoundaryVariables = false>
void test_actions(const size_t order, const int step_denominator) {
  using component = Component<Metavariables<false, false, BoundaryVariables>>;
  using var_entries = tmpl::flatten<tmpl::list<typename Metavariables<
      false, false, BoundaryVariables>::system::variables_tag>>;
  const bool forward_in_time = step_denominator > 0;
  const Slab slab(1., 3.);
  const TimeDelta initial_time_step = slab.duration() / step_denominator;
  const Time initial_time = forward_in_time ? slab.start() : slab.end();
  const auto expected = initial_values<BoundaryVariables>();

  MockRuntimeSystem<false, false, BoundaryVariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(order),
       EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{order}}};
  if constexpr (BoundaryVariables) {
    emplace_component_and_initialize(make_not_null(&runner), forward_in_time,
                                     initial_time, initial_time_step, order,
                                     tuples::get<volume_tag>(expected),
                                     tuples::get<boundary_tag>(expected));
  } else {
    emplace_component_and_initialize(make_not_null(&runner), forward_in_time,
                                     initial_time, initial_time_step, order,
                                     tuples::get<Var>(expected));
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  {
    INFO("Initialize");
    const bool jumped =
        run_past<is_initialize<tmpl::_1>, not_self_start_action>(
            make_not_null(&runner));
    CHECK(not jumped);
    CHECK(ActionTesting::get_databox_tag<component, Tags::StepNumberWithinSlab>(
              runner, 0) == 0);
    CHECK(get<0>(ActionTesting::get_databox_tag<
                 component, SelfStart::Tags::InitialValue<Tags::TimeStep>>(
              runner, 0)) == initial_time_step);
    tmpl::for_each<var_entries>([&runner, &expected](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      CHECK(get<0>(ActionTesting::get_databox_tag<
                   component, SelfStart::Tags::InitialValue<Tag>>(runner, 0)) ==
            tuples::get<Tag>(expected));
      CHECK(ActionTesting::get_databox_tag<component, Tag>(runner, 0) ==
            tuples::get<Tag>(expected));
      CHECK(ActionTesting::get_databox_tag<component,
                                           Tags::HistoryEvolvedVariables<Tag>>(
                runner, 0)
                .size() == 0);
    });
  }

  for (size_t current_order = 1; current_order < order; ++current_order) {
    CAPTURE(current_order);
    for (size_t points = 0; points <= current_order; ++points) {
      CAPTURE(points);
      {
        INFO("CheckForCompletion");
        const bool jumped =
            run_past<tt::is_a<SelfStart::Actions::CheckForCompletion, tmpl::_1>,
                     not_self_start_action>(make_not_null(&runner));
        CHECK(not jumped);
        tmpl::for_each<var_entries>([&runner, &current_order](auto tag_v) {
          using Tag = tmpl::type_from<decltype(tag_v)>;
          CHECK(ActionTesting::get_databox_tag<
                    component, Tags::HistoryEvolvedVariables<Tag>>(runner, 0)
                    .integration_order() == current_order);
        });
      }
      {
        INFO("CheckForOrderIncrease");
        const bool jumped = run_past<
            std::is_same<SelfStart::Actions::CheckForOrderIncrease, tmpl::_1>,
            not_self_start_action>(make_not_null(&runner));
        CHECK(not jumped);
        const auto next_time =
            ActionTesting::get_databox_tag<component,
                                           Tags::Next<Tags::TimeStepId>>(runner,
                                                                         0)
                .step_time();
        CHECK((next_time == initial_time) == (points == current_order));
      }
    }
  }

  {
    INFO("CheckForCompletion");
    const bool jumped =
        run_past<tt::is_a<SelfStart::Actions::CheckForCompletion, tmpl::_1>,
                 not_self_start_action>(make_not_null(&runner));
    CHECK(jumped);
  }
  {
    INFO("Cleanup");
    // Make sure we reach Cleanup to check the flow is sane...
    run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
             not_self_start_action>(make_not_null(&runner));
    // ...and then finish the procedure.
    while (not ActionTesting::get_terminate<component>(runner, 0)) {
      ActionTesting::next_action<component>(make_not_null(&runner), 0);
    }
    CHECK(ActionTesting::get_databox_tag<component, Tags::StepNumberWithinSlab>(
              runner, 0) == 0);
    CHECK(ActionTesting::get_databox_tag<component, Tags::TimeStep>(
              runner, 0) == initial_time_step);
    CHECK(ActionTesting::get_databox_tag<component, Tags::TimeStepId>(
              runner, 0) == TimeStepId(forward_in_time, 0, initial_time));
    // This test only uses Adams-Bashforth.
    CHECK(
        ActionTesting::get_databox_tag<component, Tags::Next<Tags::TimeStepId>>(
            runner, 0) ==
        TimeStepId(forward_in_time, 0, initial_time + initial_time_step));
    tmpl::for_each<var_entries>([&runner, &expected, &order](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      CHECK(ActionTesting::get_databox_tag<component, Tag>(runner, 0) ==
            tuples::get<Tag>(expected));
      CHECK(ActionTesting::get_databox_tag<component,
                                           Tags::HistoryEvolvedVariables<Tag>>(
                runner, 0)
                .integration_order() == order);
    });
  }
}

double exact_solution(const double initial_value, const double time_offset) {
  return -log(exp(-initial_value) - time_offset);
}

template <bool TestPrimitives, bool MultipleHistories>
double error_in_step(const size_t order, const double step) {
  const bool forward_in_time = step > 0.;
  const auto slab = forward_in_time ? Slab::with_duration_from_start(1., step)
                                    : Slab::with_duration_to_end(1., -step);
  const TimeDelta initial_time_step =
      (forward_in_time ? 1 : -1) * slab.duration();
  const Time initial_time = forward_in_time ? slab.start() : slab.end();
  const double initial_value = -1.;

  using component = Component<Metavariables<TestPrimitives, MultipleHistories>>;
  MockRuntimeSystem<TestPrimitives, MultipleHistories> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(order),
       EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{order}}};
  emplace_component_and_initialize<TestPrimitives, MultipleHistories>(
      make_not_null(&runner), forward_in_time, initial_time, initial_time_step,
      order, initial_value);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
           tmpl::bool_<true>, MultipleHistories>(make_not_null(&runner));
  run_past<std::is_same<
               tmpl::pin<Actions::MutateApply<UpdateU<System<TestPrimitives>>>>,
               tmpl::_1>,
           tmpl::bool_<true>, MultipleHistories>(make_not_null(&runner));

  return ActionTesting::get_databox_tag<component, Var>(runner, 0) -
         exact_solution(initial_value, step);
}

double convergence_rate(const double coarse_error, const double fine_error) {
  return (log(abs(coarse_error)) - log(abs(fine_error))) / log(2.);
}

template <bool TestPrimitives, bool MultipleHistories>
void test_convergence(const size_t order, const bool forward_in_time) {
  const double step = forward_in_time ? 0.1 : -0.1;
  // This measures the local truncation error, so order + 1.  It
  // should be converging to an integer, so just check that it looks
  // like the right one and don't worry too much about how close it
  // is.
  CHECK(convergence_rate(
            error_in_step<TestPrimitives, MultipleHistories>(order, step),
            error_in_step<TestPrimitives, MultipleHistories>(
                order, 0.5 * step)) == approx(order + 1).margin(0.1));
}

std::pair<double, std::optional<double>> boundary_variable_error_in_step(
    const size_t order, const double step, const bool with_boundary_face) {
  const bool forward_in_time = step > 0.;
  const auto slab = forward_in_time ? Slab::with_duration_from_start(1., step)
                                    : Slab::with_duration_to_end(1., -step);
  const TimeDelta initial_time_step =
      (forward_in_time ? 1 : -1) * slab.duration();
  const Time initial_time = forward_in_time ? slab.start() : slab.end();

  using component = Component<Metavariables<false, false, true>>;
  MockRuntimeSystem<false, false, true> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(order),
       EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{order}}};

  const auto initial = initial_values<true>();
  const double volume_initial_value =
      get(get<VolumeVar>(tuples::get<volume_tag>(initial)))[0];
  emplace_component_and_initialize(
      make_not_null(&runner), forward_in_time, initial_time, initial_time_step,
      order, tuples::get<volume_tag>(initial),
      with_boundary_face ? tuples::get<boundary_tag>(initial)
                         : typename boundary_tag::type{});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
           tmpl::bool_<true>, false>(make_not_null(&runner));
  run_past<
      std::is_same<
          tmpl::pin<Actions::MutateApply<UpdateU<BoundaryVariablesSystem>>>,
          tmpl::_1>,
      tmpl::bool_<true>, false>(make_not_null(&runner));

  const auto& final_volume =
      ActionTesting::get_databox_tag<component, volume_tag>(runner, 0);
  const double volume_error = get(get<VolumeVar>(final_volume))[0] -
                              exact_solution(volume_initial_value, step);
  const auto& final_boundary =
      ActionTesting::get_databox_tag<component, boundary_tag>(runner, 0);
  if (not with_boundary_face) {
    CHECK(final_boundary.variables().empty());
    return {volume_error, std::nullopt};
  }
  const double boundary_initial_value =
      get(get<BoundaryVar>(tuples::get<boundary_tag>(initial).variables().at(
          Direction<Dim>::lower_xi())))[0];
  return {volume_error, get(get<BoundaryVar>(final_boundary.variables().at(
                            Direction<Dim>::lower_xi())))[0] -
                            exact_solution(boundary_initial_value, step)};
}

void test_boundary_variables_convergence(const size_t order,
                                         const bool forward_in_time) {
  const double step = forward_in_time ? 0.1 : -0.1;
  const auto [coarse_volume, coarse_boundary] =
      boundary_variable_error_in_step(order, step, true);
  const auto [fine_volume, fine_boundary] =
      boundary_variable_error_in_step(order, 0.5 * step, true);
  CHECK(convergence_rate(coarse_volume, fine_volume) ==
        approx(order + 1).margin(0.1));
  REQUIRE(coarse_boundary.has_value());
  REQUIRE(fine_boundary.has_value());
  CHECK(convergence_rate(*coarse_boundary, *fine_boundary) ==
        approx(order + 1).margin(0.1));
}

void test_empty_boundary_variables(const size_t order,
                                   const bool forward_in_time) {
  INFO("empty boundary container is a no-op");
  const double step = forward_in_time ? 0.1 : -0.1;
  const auto [coarse_volume, coarse_boundary] =
      boundary_variable_error_in_step(order, step, false);
  const auto [fine_volume, fine_boundary] =
      boundary_variable_error_in_step(order, 0.5 * step, false);
  CHECK(not coarse_boundary.has_value());
  CHECK(not fine_boundary.has_value());
  CHECK(convergence_rate(coarse_volume, fine_volume) ==
        approx(order + 1).margin(0.1));
}

struct DummyType {};
struct DummyTag : db::SimpleTag {
  using type = DummyType;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.SelfStart", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  for (size_t order = 1; order < 5; ++order) {
    CAPTURE(order);
    for (const int step_denominator : {1, -1, 2, -2, 20, -20}) {
      CAPTURE(step_denominator);
      test_actions<false>(order, step_denominator);
      test_actions<true>(order, step_denominator);
    }
    for (const bool forward_in_time : {true, false}) {
      CAPTURE(forward_in_time);
      test_convergence<false, false>(order, forward_in_time);
      test_convergence<true, false>(order, forward_in_time);
      test_convergence<false, true>(order, forward_in_time);
      test_convergence<true, true>(order, forward_in_time);
      test_boundary_variables_convergence(order, forward_in_time);
      test_empty_boundary_variables(order, forward_in_time);
    }
  }

  TestHelpers::db::test_prefix_tag<SelfStart::Tags::InitialValue<DummyTag>>(
      "InitialValue(DummyTag)");
}
