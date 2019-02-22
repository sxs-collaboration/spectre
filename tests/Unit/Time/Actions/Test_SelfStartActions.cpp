// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/Conservative/UpdatePrimitives.hpp"  // IWYU pragma: keep
#include "Time/Actions/AdvanceTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
// IWYU pragma: no_include "Time/History.hpp"

class TimeStepper;
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct TemporalId {
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = double;
};

struct PrimitiveVar : db::SimpleTag {
  static std::string name() noexcept { return "PrimitiveVar"; }
  using type = double;
};

template <bool HasPrimitives>
struct SystemBase {
  static constexpr bool has_primitive_and_conservative_vars = HasPrimitives;
  using variables_tag = Var;

  using compute_time_derivative = struct {
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

using history_tag = Tags::HistoryEvolvedVariables<Var, Tags::dt<Var>>;

template <bool HasPrimitives = false>
struct component;  // IWYU pragma: keep

template <bool HasPrimitives = false>
struct Metavariables {
  using system = System<HasPrimitives>;
  using component_list = tmpl::list<component<HasPrimitives>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using ordered_list_of_primitive_recovery_schemes = tmpl::list<>;
  using temporal_id = TemporalId;
};

template <bool HasPrimitives>
struct component {
  using metavariables = Metavariables<HasPrimitives>;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>>;
  using update_actions = tmpl::conditional_t<
      HasPrimitives, tmpl::list<Actions::UpdateU, Actions::UpdatePrimitives>,
      Actions::UpdateU>;
  using action_list = tmpl::flatten<
      tmpl::list<SelfStart::self_start_procedure<
                     tmpl::list<Actions::ComputeTimeDerivative,
                                Actions::RecordTimeStepperData>,
                     update_actions>,
                 Actions::AdvanceTime, Actions::ComputeTimeDerivative,
                 Actions::RecordTimeStepperData, update_actions>>;
  using simple_tags = tmpl::flatten<db::AddSimpleTags<
      typename metavariables::system::variables_tag,
      typename metavariables::system::test_primitive_variables_tags,
      db::add_tag_prefix<Tags::dt,
                         typename metavariables::system::variables_tag>,
      history_tag, Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>>;
  using compute_tags = db::AddComputeTags<Tags::Time>;
  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

template <bool HasPrimitives = false>
using self_start_box = std::tuple_element_t<
    0, decltype(SelfStart::Actions::Initialize::apply(
           std::declval<typename component<HasPrimitives>::initial_databox&>(),
           tuples::TaggedTuple<>{}, std::declval<Parallel::ConstGlobalCache<
                                        Metavariables<HasPrimitives>>>(),
           int{}, tmpl::list<>{},
           std::declval<const component<HasPrimitives>*>()))>;

template <bool HasPrimitives = false>
using MockRuntimeSystem =
    ActionTesting::MockRuntimeSystem<Metavariables<HasPrimitives>>;

template <bool HasPrimitives = false>
auto make_initial_box(const bool forward_in_time, const Time& initial_time,
                      const TimeDelta& initial_time_step, const size_t order,
                      const double initial_value) noexcept {
  return db::create<component<false>::simple_tags,
                    component<false>::compute_tags>(
      initial_value, 0., db::item_type<history_tag>{}, TimeId{},
      TimeId(forward_in_time, 1 - static_cast<int64_t>(order), initial_time),
      initial_time_step);
}

template <>
auto make_initial_box<true>(const bool forward_in_time,
                            const Time& initial_time,
                            const TimeDelta& initial_time_step,
                            const size_t order,
                            const double initial_value) noexcept {
  return db::create<component<true>::simple_tags,
                    component<true>::compute_tags>(
      initial_value, initial_value, 0., db::item_type<history_tag>{}, TimeId{},
      TimeId(forward_in_time, 1 - static_cast<int64_t>(order), initial_time),
      initial_time_step);
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
    const size_t current_action =
        runner->template get_next_action_index<component<HasPrimitives>>(0);
    size_t action_to_check = current_action;
    tmpl::for_each<typename component<HasPrimitives>::action_list>(
        [&action_to_check, &done](const auto action) noexcept {
          using Action = tmpl::type_from<decltype(action)>;
          if (action_to_check-- == 0) {
            INFO(pretty_type::get_name<Action>());
            done = tmpl::apply<Stop, Action>::value;
            REQUIRE((done or tmpl::apply<Whitelist, Action>::value));
          }
        });
    runner->template next_action<component<HasPrimitives>>(0);
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Branch) false positive
    if (done) {
      // Self-start does not use the automatic algorithm looping, so
      // we don't have to check for the end.
      return current_action + 1 !=
             runner->template get_next_action_index<component<HasPrimitives>>(
                 0);
    }
  }
}

void test_actions(const size_t order, const bool forward_in_time) noexcept {
  const Slab slab(1., 3.);
  const TimeDelta initial_time_step =
      (forward_in_time ? 1 : -1) * slab.duration() / 2;
  const Time initial_time = forward_in_time ? slab.start() : slab.end();
  const double initial_value = -1.;

  using MockDistributedObjectsTag =
      MockRuntimeSystem<>::MockDistributedObjectsTag<component<>>;
  MockRuntimeSystem<>::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<component<>>{
                   make_initial_box(forward_in_time, initial_time,
                                    initial_time_step, order, initial_value)});
  MockRuntimeSystem<> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(order)},
      std::move(dist_objects)};

  {
    INFO("Initialize");
    const bool jumped =
        run_past<std::is_same<SelfStart::Actions::Initialize, tmpl::_1>,
                 not_self_start_action>(make_not_null(&runner));
    CHECK(not jumped);
    const auto& box =
        runner.algorithms<component<>>().at(0).get_databox<self_start_box<>>();
    CHECK(get<0>(db::get<SelfStart::Tags::InitialValue<Var>>(box)) ==
          initial_value);
    CHECK(get<0>(db::get<SelfStart::Tags::InitialValue<Tags::TimeStep>>(box)) ==
          initial_time_step);
    CHECK(db::get<Var>(box) == initial_value);
    CHECK(db::get<history_tag>(box).size() == 0);
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
        const auto& box = runner.algorithms<component<>>()
                              .at(0)
                              .get_databox<self_start_box<>>();
        CHECK(abs(db::get<Tags::Time>(box) - initial_time) <
              abs(initial_time_step));
        const auto next_time = db::get<Tags::Next<Tags::TimeId>>(box).time();
        CHECK((next_time == initial_time) == last_point);
      }
      {
        INFO("StartNextOrderIfReady");
        const bool jumped = run_past<
            is_a_lambda<SelfStart::Actions::StartNextOrderIfReady, tmpl::_1>,
            not_self_start_action>(make_not_null(&runner));
        CHECK(jumped == last_point);
        const auto& box = runner.algorithms<component<>>()
                              .at(0)
                              .get_databox<self_start_box<>>();
        if (points != 0) {
          CHECK((db::get<Var>(box) == initial_value) == last_point);
        }
        CHECK(db::get<history_tag>(box).size() == current_order);
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
    run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
             not_self_start_action>(make_not_null(&runner));
    const auto& box = runner.algorithms<component<>>()
                          .at(0)
                          .get_databox<typename component<>::initial_databox>();
    CHECK(db::get<Var>(box) == initial_value);
    CHECK(db::get<Tags::TimeStep>(box) == initial_time_step);
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

  using MockDistributedObjectsTag = typename MockRuntimeSystem<TestPrimitives>::
      template MockDistributedObjectsTag<component<TestPrimitives>>;
  typename MockRuntimeSystem<TestPrimitives>::TupleOfMockDistributedObjects
      dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          0, ActionTesting::MockDistributedObject<component<TestPrimitives>>{
                 make_initial_box<TestPrimitives>(forward_in_time, initial_time,
                                                  initial_time_step, order,
                                                  initial_value)});
  MockRuntimeSystem<TestPrimitives> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(order)},
      std::move(dist_objects)};

  run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
           tmpl::bool_<true>>(make_not_null(&runner));
  run_past<std::is_same<Actions::UpdateU, tmpl::_1>, tmpl::bool_<true>>(
      make_not_null(&runner));

  const double exact = -log(exp(-initial_value) - step);
  const auto& box =
      runner.template algorithms<component<TestPrimitives>>()
          .at(0)
          .template get_databox<
              typename component<TestPrimitives>::initial_databox>();
  return get<Var>(box) - exact;
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
    for (bool forward_in_time : {true, false}) {
      CAPTURE(forward_in_time);
      test_actions(order, forward_in_time);
      test_convergence<false>(order, forward_in_time);
      test_convergence<true>(order, forward_in_time);
    }
  }
}
