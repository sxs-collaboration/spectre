// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Actions/FinalTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int,
                                      tmpl::list<CacheTags::FinalTime>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.FinalTime", "[Unit][Time][Actions]") {
  const Slab slab(3., 6.);
  ActionTesting::ActionRunner<Metavariables> runner{{5.}};

  auto box = db::create<db::AddSimpleTags<Tags::TimeId, Tags::TimeStep>,
                        db::AddComputeTags<Tags::Time>>(TimeId{}, TimeDelta{});

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
        make_not_null(&box), [&test](const auto time_id, const auto time_step) {
          *time_id = TimeId(time_step->is_positive(), 0, test.time);
          *time_step = test.time_step;
        });

    bool terminate;
    std::tie(box, terminate) =
        runner.apply<component, Actions::FinalTime>(box, 0);
    CHECK(test.expected_result == terminate);
  }
}
