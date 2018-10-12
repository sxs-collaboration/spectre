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
// IWYU pragma: no_include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"  // IWYU pragma: keep
#include "Time/Actions/AdvanceTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
// IWYU pragma: no_include "Time/History.hpp"
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

class TimeStepper;

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = double;
};

struct System {
  using variables_tag = Var;
  using du_dt = struct {
    using argument_tags = tmpl::list<Var>;
    static void apply(const gsl::not_null<double*> dt_var,
                      const double var) noexcept {
      *dt_var = exp(var);
    }
  };
};

using history_tag = Tags::HistoryEvolvedVariables<Var, Tags::dt<Var>>;

struct component;
struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};

struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>>;
  using action_list = tmpl::append<
      SelfStart::self_start_procedure<
          tmpl::list<Actions::ComputeVolumeDuDt,
                     Actions::RecordTimeStepperData>,
          Actions::UpdateU>,
      tmpl::list<Actions::AdvanceTime, Actions::ComputeVolumeDuDt,
                 Actions::RecordTimeStepperData, Actions::UpdateU>>;
  using simple_tags = db::AddSimpleTags<
      metavariables::system::variables_tag,
      db::add_tag_prefix<Tags::dt, metavariables::system::variables_tag>,
      history_tag, Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>;
  using compute_tags = db::AddComputeTags<Tags::Time>;
  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

using self_start_box = decltype(db::create_from<
    db::RemoveTags<>,
    db::AddSimpleTags<SelfStart::Tags::InitialValue<Tags::TimeStep>,
                      SelfStart::Tags::InitialValue<Var>>>(
    component::initial_databox{},
    db::item_type<SelfStart::Tags::InitialValue<::Tags::TimeStep>>{},
    db::item_type<SelfStart::Tags::InitialValue<Var>>{}));

using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

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
template <typename Stop, typename Whitelist>
bool run_past(const gsl::not_null<MockRuntimeSystem*> runner) noexcept {
  for (;;) {
    bool done;
    const size_t current_action = runner->get_next_action_index<component>(0);
    size_t action_to_check = current_action;
    tmpl::for_each<component::action_list>(
        [&action_to_check, &done](const auto action) noexcept {
          using Action = tmpl::type_from<decltype(action)>;
          if (action_to_check-- == 0) {
            INFO(pretty_type::get_name<Action>());
            done = tmpl::apply<Stop, Action>::value;
            REQUIRE((done or tmpl::apply<Whitelist, Action>::value));
          }
        });
    runner->next_action<component>(0);
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Branch) false positive
    if (done) {
      // Self-start does not use the automatic algorithm looping, so
      // we don't have to check for the end.
      return current_action + 1 != runner->get_next_action_index<component>(0);
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
      MockRuntimeSystem::MockDistributedObjectsTag<component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          0, ActionTesting::MockDistributedObject<component>{
                 db::create<component::simple_tags, component::compute_tags>(
                     initial_value, 0., db::item_type<history_tag>{}, TimeId{},
                     TimeId(forward_in_time, 1 - static_cast<int64_t>(order),
                            initial_time),
                     initial_time_step)});
  MockRuntimeSystem runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(order)},
      std::move(dist_objects)};

  {
    INFO("Initialize");
    const bool jumped =
        run_past<std::is_same<SelfStart::Actions::Initialize, tmpl::_1>,
                 not_self_start_action>(&runner);
    CHECK(not jumped);
    const auto& box =
        runner.algorithms<component>().at(0).get_databox<self_start_box>();
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
            not_self_start_action>(&runner);
        CHECK(not jumped);
      }
      {
        INFO("CheckForOrderIncrease");
        const bool jumped = run_past<
            std::is_same<SelfStart::Actions::CheckForOrderIncrease, tmpl::_1>,
            not_self_start_action>(&runner);
        CHECK(not jumped);
        const auto& box =
            runner.algorithms<component>().at(0).get_databox<self_start_box>();
        CHECK(abs(db::get<Tags::Time>(box) - initial_time) <
              abs(initial_time_step));
        const auto next_time = db::get<Tags::Next<Tags::TimeId>>(box).time();
        CHECK((next_time == initial_time) == last_point);
      }
      {
        INFO("StartNextOrderIfReady");
        const bool jumped = run_past<
            is_a_lambda<SelfStart::Actions::StartNextOrderIfReady, tmpl::_1>,
            not_self_start_action>(&runner);
        CHECK(jumped == last_point);
        const auto& box =
            runner.algorithms<component>().at(0).get_databox<self_start_box>();
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
                 not_self_start_action>(&runner);
    CHECK(jumped);
  }
  {
    INFO("Cleanup");
    run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
             not_self_start_action>(&runner);
    const auto& box = runner.algorithms<component>()
                          .at(0)
                          .get_databox<typename component::initial_databox>();
    CHECK(db::get<Var>(box) == initial_value);
    CHECK(db::get<Tags::TimeStep>(box) == initial_time_step);
    CHECK(db::get<history_tag>(box).size() == order - 1);
  }
}

double error_in_step(const size_t order, const double step) noexcept {
  const bool forward_in_time = step > 0.;
  const auto slab = forward_in_time ? Slab::with_duration_from_start(1., step)
                                    : Slab::with_duration_to_end(1., -step);
  const TimeDelta initial_time_step =
      (forward_in_time ? 1 : -1) * slab.duration();
  const Time initial_time = forward_in_time ? slab.start() : slab.end();
  const double initial_value = -1.;

  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          0, ActionTesting::MockDistributedObject<component>{
                 db::create<component::simple_tags, component::compute_tags>(
                     initial_value, 0., db::item_type<history_tag>{}, TimeId{},
                     TimeId(forward_in_time, 1 - static_cast<int64_t>(order),
                            initial_time),
                     initial_time_step)});
  MockRuntimeSystem runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(order)},
      std::move(dist_objects)};

  run_past<std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>,
           tmpl::bool_<true>>(&runner);
  run_past<std::is_same<Actions::UpdateU, tmpl::_1>, tmpl::bool_<true>>(
      &runner);

  const double exact = -log(exp(-initial_value) - step);
  const auto& box = runner.algorithms<component>()
                        .at(0)
                        .get_databox<typename component::initial_databox>();
  return get<Var>(box) - exact;
}

void test_convergence(const size_t order, const bool forward_in_time) noexcept {
  const double step = forward_in_time ? 0.1 : -0.1;
  const double convergence_rate = (log(abs(error_in_step(order, step))) -
                                   log(abs(error_in_step(order, 0.5 * step)))) /
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
      test_convergence(order, forward_in_time);
    }
  }
}
