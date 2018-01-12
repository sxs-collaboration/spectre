// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <catch.hpp>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Actions/FinalTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int,
                                      tmpl::list<CacheTags::FinalTime>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.FinalTime", "[Unit][Time][Actions]") {
  const Slab slab(3., 6.);
  ActionTesting::ActionRunner<Metavariables> runner{{5.}};

  auto box = db::create<db::AddTags<Tags::TimeId, Tags::TimeStep>,
                        db::AddComputeItemsTags<Tags::Time>>(
      TimeId{}, TimeDelta{});

  struct Test {
    Time time{};
    TimeDelta time_step{};
    bool expected_result{};
  };
  const std::array<Test, 4> tests{{
      {slab.start(), slab.duration(), false},
      {slab.start(), -slab.duration(), true},
      {slab.end(), slab.duration(), true},
      {slab.end(), -slab.duration(), false}}};

  for (const auto& test : tests) {
    db::mutate<Tags::TimeId, Tags::TimeStep>(
        box,
        [&test](auto& time_id, auto& time_step) {
          time_id.time = test.time;
          time_step = test.time_step;
        });

    bool terminate;
    std::tie(box, terminate) =
        runner.apply<component, Actions::FinalTime>(box, 0);
    CHECK(test.expected_result == terminate);
  }
}
