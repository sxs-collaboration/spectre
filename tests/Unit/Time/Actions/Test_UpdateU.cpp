// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>

#include "DataStructures/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBoxTag.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Var : db::DataBoxTag {
  static constexpr db::DataBoxString_t label = "Var";
  using type = double;
};

struct System {
  using variables_tag = Var;
  using dt_variables_tag = Tags::dt<Var>;
};

struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int,
                                      tmpl::list<CacheTags::TimeStepper>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.UpdateU", "[Unit][Time][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{
    {std::make_unique<TimeSteppers::RungeKutta3>()}};

  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;
  TimeId time_id{8, slab.start(), 0};

  using history_tag = Tags::HistoryEvolvedVariables<Var, Tags::dt<Var>>;

  const auto rhs =
      [](const double t, const double y) { return 2. * t - 2. * (y - t * t); };

  auto result_box = db::create<
    db::AddTags<Tags::TimeId, Tags::TimeStep, Var, history_tag>,
    db::AddComputeItemsTags<Tags::Time>>(
        time_id, time_step, 1., history_tag::type{});

  const std::array<Time, 3> substep_times{
    {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10./3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    time_id.time = gsl::at(substep_times, substep);
    time_id.substep = substep;

    auto box = db::create_from<db::RemoveTags<Tags::TimeId>,
                               db::AddTags<Tags::TimeId, Tags::dt<Var>>>(
        result_box, time_id,
        rhs(time_id.time.value(), db::get<Var>(result_box)));

    result_box = std::get<0>(runner.apply<component, Actions::UpdateU>(box, 0));

    CHECK(db::get<Var>(result_box) ==
          approx(gsl::at(expected_values, substep)));
  }
}
