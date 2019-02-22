// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/Actions/UpdateU.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_include "Time/History.hpp"

class TimeStepper;

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = double;
};

struct System {
  using variables_tag = Var;
};

using variables_tag = Var;
using dt_variables_tag = Tags::dt<Var>;
using history_tag =
    Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>;

struct Metavariables;
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>>;
  using action_list = tmpl::list<Actions::UpdateU>;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStep, variables_tag, history_tag>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.UpdateU", "[Unit][Time][Actions]") {
  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  const auto rhs =
      [](const double t, const double y) { return 2. * t - 2. * (y - t * t); };

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<component>{
                      db::create<typename component::simple_tags>(
                          time_step, 1., history_tag::type{})});
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::RungeKutta3>()},
                           std::move(dist_objects)};

  const std::array<Time, 3> substep_times{
    {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10./3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    auto& before_box = runner.algorithms<component>()
                           .at(0)
                           .get_databox<typename component::initial_databox>();
    db::mutate<history_tag>(
        make_not_null(&before_box),
        [&rhs, &substep, &substep_times ](
            const gsl::not_null<db::item_type<history_tag>*> history,
            const double& vars) noexcept {
          const Time& time = gsl::at(substep_times, substep);
          history->insert(time, vars, rhs(time.value(), vars));
        },
        db::get<variables_tag>(before_box));

    runner.next_action<component>(0);
    auto& box = runner.algorithms<component>()
                    .at(0)
                    .get_databox<typename component::initial_databox>();

    CHECK(db::get<variables_tag>(box) ==
          approx(gsl::at(expected_values, substep)));
  }
}
