// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "Time/Actions/AdvanceTime.hpp"         // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <initializer_list>
// IWYU pragma: no_include <unordered_map>

class TimeStepper;
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct Metavariables;
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>>;
  using action_list = tmpl::list<Actions::AdvanceTime>;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

struct Metavariables {
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};

void check_rk3(const Time& start, const TimeDelta& time_step) {
  const std::array<TimeDelta, 3> substep_offsets{
      {time_step * 0, time_step, time_step / 2}};

  using simple_tags =
      db::AddSimpleTags<Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<component>{
                      db::create<simple_tags>(
                          TimeId(time_step.is_positive(), 8, start),
                          TimeId(time_step.is_positive(), 8, start, 1,
                                 start + substep_offsets[1]),
                          time_step)});
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::RungeKutta3>()},
                           std::move(dist_objects)};

  for (const auto& step_start : {start, start + time_step}) {
    for (size_t substep = 0; substep < 3; ++substep) {
      auto& box = runner.algorithms<component>()
                      .at(0)
                      .get_databox<db::compute_databox_type<simple_tags>>();
      CHECK(db::get<Tags::TimeId>(box) ==
            TimeId(time_step.is_positive(), 8, step_start, substep,
                   step_start + gsl::at(substep_offsets, substep)));
      CHECK(db::get<Tags::TimeStep>(box) == time_step);
      runner.next_action<component>(0);
    }
  }

  auto& box = runner.algorithms<component>()
                  .at(0)
                  .get_databox<db::compute_databox_type<simple_tags>>();
  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto& expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeId(time_step.is_positive(), 8, start + 2 * time_step));
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
}

void check_abn(const Time& start, const TimeDelta& time_step) {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<component>{
                      db::create<typename component::simple_tags>(
                          TimeId(time_step.is_positive(), 8, start),
                          TimeId(time_step.is_positive(), 8, start + time_step),
                          time_step)});
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::AdamsBashforthN>(1)},
                           std::move(dist_objects)};

  for (const auto& step_start : {start, start + time_step}) {
    auto& box = runner.algorithms<component>()
                    .at(0)
                    .get_databox<typename component::initial_databox>();
    CHECK(db::get<Tags::TimeId>(box) ==
          TimeId(time_step.is_positive(), 8, step_start));
    CHECK(db::get<Tags::TimeStep>(box) == time_step);
    runner.next_action<component>(0);
  }

  auto& box = runner.algorithms<component>()
                  .at(0)
                  .get_databox<typename component::initial_databox>();
  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto& expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeId(time_step.is_positive(), 8, start + 2 * time_step));
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.AdvanceTime", "[Unit][Time][Actions]") {
  const Slab slab(0., 1.);
  check_rk3(slab.start(), slab.duration() / 2);
  check_rk3(slab.end(), -slab.duration() / 2);
  check_abn(slab.start(), slab.duration() / 2);
  check_abn(slab.end(), -slab.duration() / 2);
}
