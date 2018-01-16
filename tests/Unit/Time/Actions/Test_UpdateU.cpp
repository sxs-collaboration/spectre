// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
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
  static constexpr db::DataBoxString label = "Var";
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
  using variables_tag = Var;
  using dt_variables_tag = Tags::dt<Var>;

  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;
  TimeId time_id{8, slab.start(), 0};

  using history_tag =
      Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>;

  const auto rhs =
      [](const double t, const double y) { return 2. * t - 2. * (y - t * t); };

  auto box = db::create<
    db::AddTags<Tags::TimeId, Tags::TimeStep, variables_tag, dt_variables_tag,
                history_tag>,
    db::AddComputeItemsTags<Tags::Time>>(
        time_id, time_step, 1., 0., history_tag::type{});

  const std::array<Time, 3> substep_times{
    {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10./3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    db::mutate<dt_variables_tag, Tags::TimeId>(
        box,
        [&rhs, &substep, &substep_times ](double& dt_vars,
                                          TimeId& local_time_id,
                                          const double& vars) noexcept {
          local_time_id.time = gsl::at(substep_times, substep);
          local_time_id.substep = substep;

          dt_vars = rhs(local_time_id.time.value(), vars);
        },
        db::get<variables_tag>(box));

    box = std::get<0>(runner.apply<component, Actions::UpdateU>(box, 0));

    CHECK(db::get<variables_tag>(box) ==
          approx(gsl::at(expected_values, substep)));
  }
}
