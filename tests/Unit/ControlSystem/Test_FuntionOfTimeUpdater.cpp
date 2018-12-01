// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/FunctionOfTimeUpdater.hpp"
#include "ControlSystem/PiecewisePolynomial.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/ControlSystem/FoTUpdater_Helper.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.FunctionOfTimeUpdater.Translation",
                  "[ControlSystem][Unit]") {
  // A simple Translation `ControlError` is defined in FoTUpdater_Helper,
  // located in this test dir, in order to test the FunctionOfTimeUpdater.
  // The full Translation ControlError will be more complicated, since it will
  // need information from a DataBox.
  // Here, we utilize the simple 2-d case for testing a FunctionOfTime with
  // multiple components

  double t = 0.0;
  const double dt = 1.0e-3;
  const double final_time = 4.0;
  constexpr size_t deriv_order = 2;

  // This translation test is a simple 2-D test with time-dependent
  // `inertial` coordinates. We want to test that the f_of_t (denoted \lambda(t)
  // here) updates properly and continually maps the grid_coords to match the
  // inertial_coords: grid_coords -> grid_coords + \lambda(t)
  // The target map parameter is
  // \lambda_target(t) = inertial_coords(t) - grid_coords
  // The error in the mapping is defined as Q = \lambda_target - \lambda.
  // This is implemented in FoTUpdater_Helper, wehere we define the error as:
  // Q = inertial_coords(t) - grid_coords - f_of_t
  // where f_of_t is the current map parameter \lambda(t)
  DataVector grid_coords{{0.2, 0.4}};
  // Here we set up the inertial coords to have a sinusoidal time
  // dependence, which agrees with the grid_coords at t=0
  const double amp1 = 0.7;
  const double omega1 = 4.0 * M_PI;
  const double amp2 = 0.3;
  const double omega2 = 6.0 * M_PI;
  DataVector inertial_coords{grid_coords};

  // initialize our FunctionOfTime to agree at t=0
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {amp1 * omega1, amp2 * omega2}, {0.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(t, init_func);

  Averager<deriv_order> averager(0.25, false);
  Controller<deriv_order> control_signal;

  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const double initial_timescale = 2.0e-3;
  TimescaleTuner tst({initial_timescale, initial_timescale}, max_timescale,
                     min_timescale, decrease_timescale_threshold,
                     increase_timescale_threshold, increase_factor,
                     decrease_factor);

  FunctionOfTimeUpdater<deriv_order> updater(
      std::move(averager),
      std::move(control_signal),  // NOLINT
      std::move(tst));

  TestHelpers::ControlErrors::Translation<deriv_order> trans_error(grid_coords);

  while (t < final_time) {
    // make the error measurement
    trans_error(&updater, f_of_t, t, inertial_coords);
    // update the FunctionOfTime
    updater.modify(&f_of_t, t);
    // check that Q is within the specified tolerance
    CHECK(fabs(inertial_coords[0] - grid_coords[0] - f_of_t.func(t)[0][0]) <=
          decrease_timescale_threshold);
    CHECK(fabs(inertial_coords[1] - grid_coords[1] - f_of_t.func(t)[0][1]) <=
          decrease_timescale_threshold);

    // increase time and get inertial_coords(t)
    t += dt;
    inertial_coords[0] = grid_coords[0] + amp1 * sin(omega1 * t);
    inertial_coords[1] = grid_coords[1] + amp2 * sin(omega2 * t);
  }
}
