// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Time/TimeSteppers/LtsHelpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>

#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeStepperTestUtils::lts {
void test_equal_rate(const LtsTimeStepper& stepper, const size_t order,
                     const size_t number_of_past_steps, const double epsilon,
                     const bool forward) {
  // This does an integral putting the entire derivative into the
  // boundary term.
  const double unused_local_deriv = 4444.;

  auto analytic = [](double t) { return sin(t); };
  auto driver = [](double t) { return cos(t); };
  auto coupling = [=](const double local, const double remote) {
    CHECK(local == unused_local_deriv);
    return remote;
  };

  Approx approx = Approx::custom().epsilon(epsilon);

  const uint64_t num_steps = 100;
  const Slab slab(0.875, 1.);
  const TimeDelta step_size = (forward ? 1 : -1) * slab.duration() / num_steps;

  TimeStepId time_id(forward, 0, forward ? slab.start() : slab.end());
  double y = analytic(time_id.substep_time());
  TimeSteppers::History<double> volume_history{order};
  TimeSteppers::BoundaryHistory<double, double, double> boundary_history{};

  {
    Time history_time = time_id.step_time();
    TimeDelta history_step_size = step_size;
    for (size_t j = 0; j < number_of_past_steps; ++j) {
      ASSERT(history_time.slab() == history_step_size.slab(), "Slab mismatch");
      if ((history_step_size.is_positive() and
           history_time.is_at_slab_start()) or
          (not history_step_size.is_positive() and
           history_time.is_at_slab_end())) {
        const Slab new_slab =
            history_time.slab().advance_towards(-history_step_size);
        history_time = history_time.with_slab(new_slab);
        history_step_size = history_step_size.with_slab(new_slab);
      }
      history_time -= history_step_size;
      const TimeStepId history_id(forward, 0, history_time);
      volume_history.insert_initial(history_id, analytic(history_time.value()),
                                    0.);
      boundary_history.local().insert_initial(history_id, order,
                                              unused_local_deriv);
      boundary_history.remote().insert_initial(history_id, order,
                                               driver(history_time.value()));
    }
  }

  for (uint64_t i = 0; i < num_steps; ++i) {
    for (uint64_t substep = 0;
         substep < stepper.number_of_substeps();
         ++substep) {
      volume_history.insert(time_id, y, 0.);
      boundary_history.local().insert(time_id, order, unused_local_deriv);
      boundary_history.remote().insert(time_id, order,
                                       driver(time_id.substep_time()));

      stepper.update_u(make_not_null(&y), volume_history, step_size);
      stepper.clean_history(make_not_null(&volume_history));
      stepper.add_boundary_delta(&y, make_not_null(&boundary_history),
                                 step_size, coupling);
      time_id = stepper.next_time_id(time_id, step_size);
    }
    CHECK(y == approx(analytic(time_id.substep_time())));
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(boundary_history.local().size() < 20);
  CHECK(boundary_history.remote().size() < 20);
}

void test_dense_output(const LtsTimeStepper& stepper) {
  // We only support variable time-step, multistep LTS integration.
  // Any multistep, variable time-step integrator must give the same
  // results from dense output as from just taking a short step
  // because we require dense output to be continuous.  A sufficient
  // test is therefore to run with an LTS pattern and check that the
  // dense output predicts the actual step result.
  const Slab slab(0., 1.);

  // We don't use any meaningful values.  We only care that the dense
  // output gives the same result as normal output.
  // NOLINTNEXTLINE(spectre-mutable)
  auto get_value = [value = 1.]() mutable { return value *= 1.1; };

  const auto coupling = [](const double a, const double b) { return a * b; };

  const auto make_time_id = [](const Time& t) {
    return TimeStepId(true, 0, t);
  };

  TimeSteppers::BoundaryHistory<double, double, double> history{};
  {
    const Slab init_slab = slab.retreat();
    for (size_t i = 0; i < stepper.number_of_past_steps(); ++i) {
      const Time init_time =
          init_slab.end() -
          init_slab.duration() * (i + 1) / stepper.number_of_past_steps();
      history.local().insert_initial(make_time_id(init_time), stepper.order(),
                                     get_value());
      history.remote().insert_initial(make_time_id(init_time), stepper.order(),
                                      get_value());
    }
  }

  std::array<std::deque<TimeDelta>, 2> dt{
      {{slab.duration() / 2, slab.duration() / 4, slab.duration() / 4},
       {slab.duration() / 6, slab.duration() / 6, slab.duration() * 2 / 9,
        slab.duration() * 4 / 9}}};

  Time t = slab.start();
  Time next_check = t + dt[0][0];
  std::array<Time, 2> next{{t, t}};
  for (;;) {
    const size_t side = next[0] <= next[1] ? 0 : 1;

    if (side == 0) {
      history.local().insert(make_time_id(next[0]), stepper.order(),
                             get_value());
    } else {
      history.remote().insert(make_time_id(next[1]), stepper.order(),
                              get_value());
    }

    const TimeDelta this_dt = gsl::at(dt, side).front();
    gsl::at(dt, side).pop_front();

    gsl::at(next, side) += this_dt;

    if (std::min(next[0], next[1]) == next_check) {
      double dense_result = 0.0;
      stepper.boundary_dense_output(&dense_result, history, next_check.value(),
                                    coupling);
      double delta = 0.0;
      stepper.add_boundary_delta(&delta, make_not_null(&history),
                                 next_check - t, coupling);
      CHECK(dense_result == approx(delta));
      if (next_check.is_at_slab_boundary()) {
        break;
      }
      t = next_check;
      next_check += dt[0].front();
    }
  }
}
}  // namespace TimeStepperTestUtils::lts
