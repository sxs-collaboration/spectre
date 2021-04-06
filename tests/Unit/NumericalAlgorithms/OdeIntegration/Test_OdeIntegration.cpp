// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.OdeIntegration",
                  "[Unit][NumericalAlgorithms]") {
  // explicit stepper for an array of fundamental types
  // [explicit_fundamental_array_system]
  const auto oscillatory_array_system = [](const std::array<double, 2>& state,
                                           std::array<double, 2>& dt_state,
                                           const double /*time*/) noexcept {
    dt_state[0] = state[1];
    dt_state[1] = -state[0];
  };
  // [explicit_fundamental_array_system]
  // [explicit_fundamental_stepper_construction]
  boost::numeric::odeint::runge_kutta4<std::array<double, 2>>
      fixed_array_stepper;
  // [explicit_fundamental_stepper_construction]
  // [explicit_fundamental_stepper_use]
  const double fixed_step = 0.001;
  std::array<double, 2> fixed_step_array_state{{1.0, 0.0}};
  for (size_t i = 0; i < 1000; ++i) {
    CHECK(approx(fixed_step_array_state[0]) == cos(i * fixed_step));
    CHECK(approx(fixed_step_array_state[1]) == -sin(i * fixed_step));
    fixed_array_stepper.do_step(oscillatory_array_system,
                                fixed_step_array_state, i * fixed_step,
                                fixed_step);
  }
  // [explicit_fundamental_stepper_use]

  // dense output stepper for a fundamental type
  // [dense_output_fundamental_system]
  const auto quadratic_system = [](const double /*state*/, double& dt_state,
                                   const double time) noexcept {
    dt_state = 2.0 * time;
  };
  // [dense_output_fundamental_system]
  // [dense_output_fundamental_construction]
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<double>>>
      dense_fundamental_stepper = boost::numeric::odeint::make_dense_output(
          1.0e-12, 1.0e-12,
          boost::numeric::odeint::runge_kutta_dopri5<double>{});
  dense_fundamental_stepper.initialize(1.0, 1.0, 0.01);
  // [dense_output_fundamental_construction]
  // [dense_output_fundamental_stepper_use]
  std::pair<double, double> step_range =
      dense_fundamental_stepper.do_step(quadratic_system);
  double state_output;
  for(size_t i = 0; i < 50; ++i) {
    while(step_range.second < 1.0 + 0.01 * square(i) ) {
      step_range = dense_fundamental_stepper.do_step(quadratic_system);
    }
    dense_fundamental_stepper.calc_state(1.0 + 0.01 * square(i), state_output);
    CHECK(approx(state_output) == square(1.0 + 0.01 * square(i)));
  }
  // [dense_output_fundamental_stepper_use]
  // dense output stepper for an array of SpECTRE vectors
  // [dense_output_vector_stepper]
  const auto oscillatory_vector_system =
      [](const std::array<DataVector, 2>& state,
         std::array<DataVector, 2>& dt_state, const double /*time*/) noexcept {
        dt_state[0] = state[1];
        dt_state[1] = -state[0];
      };
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<DataVector, 2>>>>
      dense_stepper =
      boost::numeric::odeint::make_dense_output(1.0e-14, 1.0e-14,
                            boost::numeric::odeint::runge_kutta_dopri5<
                                std::array<DataVector, 2>>{});
  const DataVector initial_vector = {{0.1, 0.2, 0.3, 0.4, 0.5}};
  dense_stepper.initialize(
      std::array<DataVector, 2>{{initial_vector, DataVector{5, 0.0}}}, 0.0,
      0.01);
  std::array<DataVector, 2> dense_output_vector_state{
      {DataVector{5}, DataVector{5}}};
  step_range = dense_stepper.do_step(oscillatory_vector_system);
  for(size_t i = 0; i < 100; ++i) {
    while (step_range.second < 0.01 * i) {
      step_range = dense_stepper.do_step(oscillatory_vector_system);
    }
    dense_stepper.calc_state(0.01 * i, dense_output_vector_state);
    for(size_t j = 0; j < 5; ++j) {
      CHECK(approx(dense_output_vector_state[0][j]) ==
            (0.1 * j + 0.1) * cos(0.01 * i));
      CHECK(approx(dense_output_vector_state[1][j]) ==
            -(0.1 * j + 0.1) * sin(0.01 * i));
    }
  }
  // [dense_output_vector_stepper]
}
