// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>  // IWYU pragma: keep
#include <limits>
#include <memory>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/Conservative/UpdatePrimitives.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"   // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
// IWYU pragma: no_include "Time/History.hpp"

class TimeStepper;
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct TemporalId {
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct Var : db::SimpleTag {
  using type = double;
};

struct PrimitiveVar : db::SimpleTag {
  using type = double;
};

template <bool HasPrimitives>
struct SystemBase {
  static constexpr bool has_primitive_and_conservative_vars = HasPrimitives;
  using variables_tag = Var;

  using ComputeTimeDerivative = struct {
    template <template <class> class StepPrefix>
    using return_tags = tmpl::list<db::add_tag_prefix<StepPrefix, Var>>;

    using argument_tags =
        tmpl::list<tmpl::conditional_t<has_primitive_and_conservative_vars,
                                       PrimitiveVar, Var>>;
    static void apply(const gsl::not_null<double*> dt_var,
                      const double var) noexcept {
      *dt_var = exp(var);
    }
  };
};

template <bool HasPrimitives = false>
struct System : SystemBase<false> {
  // Do not define primitive_variables_tag here.  Actions must work without it.

  // Only used by the test
  using test_primitive_variables_tags = tmpl::list<>;
};

template <>
struct System<true> : SystemBase<true> {
  using primitive_variables_tag = PrimitiveVar;
  // Only used by the test
  using test_primitive_variables_tags = tmpl::list<primitive_variables_tag>;

  template <typename>
  struct primitive_from_conservative {
    using return_tags = tmpl::list<PrimitiveVar>;
    using argument_tags = tmpl::list<Var>;
    static void apply(const gsl::not_null<double*> prim,
                      const double cons) noexcept {
      *prim = cons;
    }
  };
};

using history_tag = Tags::HistoryEvolvedVariables<Var>;

template <typename Metavariables>
struct Component;  // IWYU pragma: keep

template <bool HasPrimitives = false>
struct Metavariables {
  static constexpr bool has_primitives = HasPrimitives;
  using system = System<HasPrimitives>;
  using component_list = tmpl::list<Component<Metavariables>>;
  using ordered_list_of_primitive_recovery_schemes = tmpl::list<>;
  using temporal_id = TemporalId;

  enum class Phase { Initialization, Testing, Exit };
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<TimeStepper>>;
  using simple_tags = tmpl::flatten<db::AddSimpleTags<
      typename metavariables::system::variables_tag,
      typename metavariables::system::test_primitive_variables_tags,
      db::add_tag_prefix<Tags::dt,
                         typename metavariables::system::variables_tag>,
      history_tag, Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
      Tags::TimeStep, Tags::Time>>;
  using compute_tags = db::AddComputeTags<Tags::SubstepTimeCompute>;

  static constexpr bool has_primitives = Metavariables::has_primitives;

  using step_actions =
      tmpl::list<Actions::ComputeTimeDerivative<
                     typename metavariables::system::ComputeTimeDerivative>,
                 Actions::RecordTimeStepperData<>,
                 tmpl::conditional_t<
                     has_primitives,
                     tmpl::list<Actions::UpdateU<>, Actions::UpdatePrimitives>,
                     Actions::UpdateU<>>>;
  using action_list = tmpl::flatten<
      tmpl::list<SelfStart::self_start_procedure<step_actions>, step_actions>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, action_list>>;
};

template <bool HasPrimitives = false>
using MockRuntimeSystem =
    ActionTesting::MockRuntimeSystem<Metavariables<HasPrimitives>>;

template <bool HasPrimitives = false>
void emplace_component_and_initialize(
    const gsl::not_null<MockRuntimeSystem<HasPrimitives>*> runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    const double initial_value) noexcept {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<HasPrimitives>>>(
      runner, 0,
      {initial_value, 0., db::item_type<history_tag>{}, TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN()});
}

template <>
void emplace_component_and_initialize<true>(
    const gsl::not_null<MockRuntimeSystem<true>*> runner,
    const bool forward_in_time, const Time& initial_time,
    const TimeDelta& initial_time_step, const size_t order,
    const double initial_value) noexcept {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<true>>>(
      runner, 0,
      {initial_value, initial_value, 0., db::item_type<history_tag>{},
       TimeStepId{},
       TimeStepId(forward_in_time, 1 - static_cast<int64_t>(order),
                  initial_time),
       initial_time_step, std::numeric_limits<double>::signaling_NaN()});
}

namespace detail {
template <template <typename> class U>
struct is_a_wrapper;

template <typename U, typename T>
struct wrapped_is_a;

template <template <typename> class U, typename T>
struct wrapped_is_a<is_a_wrapper<U>, T> : tt::is_a<U, T> {};
}  // namespace detail

template <template <typename> class U, typename T>
using is_a_lambda = detail::wrapped_is_a<detail::is_a_wrapper<U>, T>;

using not_self_start_action = cpp17::negation<cpp17::disjunction<
    std::is_same<SelfStart::Actions::Initialize, tmpl::_1>,
    is_a_lambda<SelfStart::Actions::CheckForCompletion, tmpl::_1>,
    std::is_same<SelfStart::Actions::CheckForOrderIncrease, tmpl::_1>,
    is_a_lambda<SelfStart::Actions::StartNextOrderIfReady, tmpl::_1>,
    std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>>>;

// Run until an action satisfying the Stop metalambda is executed.
// Fail a REQUIRE if any action not passing the Whitelist metalambda
// is run first (as that would often lead to an infinite loop).
// Returns true if the last action jumped.
template <typename Stop, typename Whitelist, bool HasPrimitives>
bool run_past(
    const gsl::not_null<MockRuntimeSystem<HasPrimitives>*> runner) noexcept {
  for (;;) {
    bool done;
    const size_t current_action = ActionTesting::get_next_action_index<
        Component<Metavariables<HasPrimitives>>>(*runner, 0);
    size_t action_to_check = current_action;
    tmpl::for_each<
        typename Component<Metavariables<HasPrimitives>>::action_list>(
        [&action_to_check, &done ](const auto action) noexcept {
          using Action = tmpl::type_from<decltype(action)>;
          if (action_to_check-- == 0) {
            INFO(pretty_type::get_name<Action>());
            done = tmpl::apply<Stop, Action>::value;
            REQUIRE((done or tmpl::apply<Whitelist, Action>::value));
          }
        });
    ActionTesting::next_action<Component<Metavariables<HasPrimitives>>>(runner,
                                                                        0);
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Branch) false positive
    if (done) {
      // Self-start does not use the automatic algorithm looping, so
      // we don't have to check for the end.
      return current_action + 1 !=
             ActionTesting::get_next_action_index<
                 Component<Metavariables<HasPrimitives>>>(*runner, 0);
    }
  }
}

void test_actions(const size_t order, const int step_denominator) noexcept {
  const bool forward_in_time = step_denominator > 0;
  const Slab slab(1., 3.);
  const TimeDelta initial_time_step = slab.duration() / step_denominator;
  const Time initial_time = forward_in_time ? slab.start() : slab.end();
  const double initial_value = -1.;

  using simple_tags = typename Component<Metavariables<>>::simple_tags;
  using compute_tags = typename Component<Metavariables<>>::compute_tags;

  MockRuntimeSystem<> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(order)}};
  emplace_component_and_initialize(make_not_null(&runner), forward_in_time,
                                   initial_time, initial_time_step, order,
                                   initial_value);

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables<>::Phase::Testing);

  {
    INFO("Initialize");
    const bool jumped =
        run_past<std::is_same<SelfStart::Actions::Initialize, tmpl::_1>,
                 not_self_start_action>(make_not_null(&runner));
    CHECK(not jumped);
    CHECK(
        get<0>(
            ActionTesting::get_databox_tag<Component<Metavariables<>>,
                                           SelfStart::Tags::InitialValue<Var>>(
                runner, 0)) == initial_value);
    CHECK(get<0>(ActionTesting::get_databox_tag<
                 Component<Metavariables<>>,
                 SelfStart::Tags::InitialValue<Tags::TimeStep>>(runner, 0)) ==
          initial_time_step);
    CHECK(ActionTesting::get_databox_tag<Component<Metavariables<>>, Var>(
              runner, 0) == initial_value);
    CHECK(
        ActionTesting::get_databox_tag<Component<Metavariables<>>, history_tag>(
            runner, 0)
            .size() == 0);
  }

  for (size_t current_order = 1; current_order < order; ++current_order) {
    CAPTURE(current_order);
    for (size_t points = 0; points <= current_order; ++points) {
      CAPTURE(points);
      const bool last_point = points == current_order;
      {
        INFO("CheckForCompletion");
        const bool jumped = run_past<
            is_a_lambda<SelfStart::Actions::CheckForCompletion, tmpl::_1>,
            not_self_start_action>(make_not_null(&runner));
        CHECK(not jumped);
      }
      {
        INFO("CheckForOrderIncrease");
        const bool jumped = run_past<
            std::is_same<SelfStart::Actions::CheckForOrderIncrease, tmpl::_1>,
            not_self_start_action>(make_not_null(&runner));
        CHECK(not jumped);
        const auto next_time =
            ActionTesting::get_databox_tag<Component<Metavariables<>>,
                                           Tags::Next<Tags::TimeStepId>>(runner,
                                                                         0)
                .step_time();
        CHECK((next_time == initial_time) == last_point);
      }
      {
        INFO("StartNextOrderIfReady");
        const bool jumped = run_past<
            is_a_lambda<SelfStart::Actions::StartNextOrderIfReady, tmpl::_1>,
            not_self_start_action>(make_not_null(&runner));
        CHECK(jumped == last_point);
        if (points != 0) {
          CHECK(
              (ActionTesting::get_databox_tag<Component<Metavariables<>>, Var>(
                   runner, 0) == initial_value) == last_point);
        }
        CHECK(ActionTesting::get_databox_tag<Component<Metavariables<>>,
                                             history_tag>(runner, 0)
                  .size() == current_order);
      }
    }
  }

  {
    INFO("CheckForCompletion");
    const bool jumped =
        run_past<is_a_lambda<SelfStart::Actions::CheckForCompletion, tmpl::_1>,
                 not_self_start_action>(make_not_null(&runner));
    CHECK(jumped);
  }
  {
    INFO("Cleanup");
    // Make sure we reach Cleanup to check the flow is sane...
    run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
             not_self_start_action>(make_not_null(&runner));
    // ...and then finish the procedure.
    while (not ActionTesting::get_terminate<Component<Metavariables<>>>(runner,
                                                                        0)) {
      ActionTesting::next_action<Component<Metavariables<>>>(
          make_not_null(&runner), 0);
    }
    const auto& box =
        ActionTesting::get_databox<Component<Metavariables<>>,
                                   tmpl::append<simple_tags, compute_tags>>(
            runner, 0);
    CHECK(db::get<Var>(box) == initial_value);
    CHECK(db::get<Tags::TimeStep>(box) == initial_time_step);
    CHECK(db::get<Tags::TimeStepId>(box) ==
          TimeStepId(forward_in_time, 0, initial_time));
    // This test only uses Adams-Bashforth.
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) ==
          TimeStepId(forward_in_time, 0, initial_time + initial_time_step));
    CHECK(db::get<history_tag>(box).size() == order - 1);
  }
}

template <bool TestPrimitives>
double error_in_step(const size_t order, const double step) noexcept {
  const bool forward_in_time = step > 0.;
  const auto slab = forward_in_time ? Slab::with_duration_from_start(1., step)
                                    : Slab::with_duration_to_end(1., -step);
  const TimeDelta initial_time_step =
      (forward_in_time ? 1 : -1) * slab.duration();
  const Time initial_time = forward_in_time ? slab.start() : slab.end();
  const double initial_value = -1.;

  using component = Component<Metavariables<TestPrimitives>>;
  MockRuntimeSystem<TestPrimitives> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(order)}};
  emplace_component_and_initialize<TestPrimitives>(
      make_not_null(&runner), forward_in_time, initial_time, initial_time_step,
      order, initial_value);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables<TestPrimitives>::Phase::Testing);

  run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
           tmpl::bool_<true>>(make_not_null(&runner));
  run_past<std::is_same<tmpl::pin<Actions::UpdateU<>>, tmpl::_1>,
           tmpl::bool_<true>>(make_not_null(&runner));

  const double exact = -log(exp(-initial_value) - step);
  return ActionTesting::get_databox_tag<component, Var>(runner, 0) - exact;
}

template <bool TestPrimitives>
void test_convergence(const size_t order, const bool forward_in_time) noexcept {
  const double step = forward_in_time ? 0.1 : -0.1;
  const double convergence_rate =
      (log(abs(error_in_step<TestPrimitives>(order, step))) -
       log(abs(error_in_step<TestPrimitives>(order, 0.5 * step)))) /
      log(2.);
  // This measures the local truncation error, so order + 1.  It
  // should be converging to an integer, so just check that it looks
  // like the right one and don't worry too much about how close it
  // is.
  CHECK(convergence_rate == approx(order + 1).margin(0.1));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.SelfStart", "[Unit][Time][Actions]") {
  for (size_t order = 1; order < 5; ++order) {
    CAPTURE(order);
    for (const int step_denominator : {1, -1, 2, -2, 20, -20}) {
      CAPTURE(step_denominator);
      test_actions(order, step_denominator);
    }
    for (const bool forward_in_time : {true, false}) {
      CAPTURE(forward_in_time);
      test_convergence<false>(order, forward_in_time);
      test_convergence<true>(order, forward_in_time);
    }
  }
}
