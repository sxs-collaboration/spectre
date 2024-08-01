// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Time/TimeSteppers/LtsHelpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>

#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Rational.hpp"

namespace TimeStepperTestUtils::lts {
namespace {
VariableOrderChoice fixed_order_choice(const LtsTimeStepper& stepper) {
  REQUIRE(holds_alternative<TimeSteppers::Tags::FixedOrder>(stepper.order()));
  const size_t order = get<TimeSteppers::Tags::FixedOrder>(stepper.order());
  const size_t past_steps = stepper.number_of_past_steps();
  return {order, past_steps, order, past_steps};
}

VariableOrderChoice flip(const VariableOrderChoice& choice) {
  return {choice.remote_order, choice.remote_number_of_past_steps,
          choice.local_order, choice.local_number_of_past_steps};
}

void arbitrary_compare_to_volume(const LtsTimeStepper& stepper,
                                 const bool time_runs_forward,
                                 std::deque<Rational> step_pattern,
                                 std::deque<Rational> neighbor_step_pattern,
                                 const double tolerance,
                                 const VariableOrderChoice& order_choice) {
  const auto local_approx = Approx::custom().epsilon(tolerance);
  const auto step_sign = time_runs_forward ? 1 : -1;

  // Arbitrary function that gives "generic" results.  We don't
  // attempt to perform a meaningful integral.
  const auto arbitrary_rhs = [](const TimeStepId& t) {
    return 1.23 * static_cast<double>(t.substep()) + t.substep_time();
  };

  TimeSteppers::History<double> volume_history{order_choice.local_order};
  TimeSteppers::BoundaryHistory<double, double, double> boundary_history{};

  const Slab initial_slab(1.2, 3.4);
  TimeStepId id(time_runs_forward, 0,
                time_runs_forward ? initial_slab.start() : initial_slab.end());
  {
    auto init_step = step_sign * initial_slab.duration();
    auto init_time = id.step_time();
    int64_t slab_number = 0;
    while (boundary_history.local().size() <
               order_choice.local_number_of_past_steps or
           boundary_history.remote().size() <
               order_choice.remote_number_of_past_steps) {
      --slab_number;
      init_step =
          init_step.with_slab(init_step.slab().advance_towards(-init_step));
      init_time -= init_step;
      const TimeStepId init_id(time_runs_forward, slab_number, init_time);
      const double deriv = arbitrary_rhs(init_id);
      if (boundary_history.local().size() <
          order_choice.local_number_of_past_steps) {
        volume_history.insert_initial(init_id, 0.0, deriv);
        boundary_history.local().insert_initial(
            init_id, order_choice.local_order, deriv);
      }
      if (boundary_history.remote().size() <
          order_choice.remote_number_of_past_steps) {
        boundary_history.remote().insert_initial(
            init_id, order_choice.remote_order, deriv);
      }
    }
  }

  const bool equal_rate = step_pattern == neighbor_step_pattern;
  const auto coupling = [&](const double local, const double remote) {
    if (equal_rate) {
      CHECK(local == remote);
    }
    return local;
  };

  auto neighbor_id = id;

  const auto add_neighbor_entries = [&](const auto& limit) {
    while (stepper.neighbor_data_required(limit, neighbor_id)) {
      boundary_history.remote().insert(neighbor_id, order_choice.remote_order,
                                       arbitrary_rhs(neighbor_id));
      ASSERT(not neighbor_step_pattern.empty(), "Test logic error");
      const TimeDelta neighbor_step = step_sign *
                                      neighbor_step_pattern.front() *
                                      neighbor_id.step_time().slab().duration();
      neighbor_id = stepper.next_time_id(neighbor_id, neighbor_step);
      if (neighbor_id.substep() == 0) {
        neighbor_step_pattern.pop_front();
      }
    }
  };

  while (not step_pattern.empty()) {
    const TimeDelta step =
        step_sign * step_pattern.front() * id.step_time().slab().duration();
    step_pattern.pop_front();

    const double dense_test = (id.step_time() + step / 3).value();
    bool dense_succeeded = false;
    do {
      {
        const double deriv = arbitrary_rhs(id);
        volume_history.insert(id, 0.0, deriv);
        boundary_history.local().insert(id, order_choice.local_order, deriv);
      }

      // Check dense output
      {
        double volume_value = 0.0;
        if (stepper.dense_update_u(make_not_null(&volume_value), volume_history,
                                   dense_test)) {
          CHECK(not dense_succeeded);
          dense_succeeded = true;
          add_neighbor_entries(dense_test);
          double boundary_value = 0.0;
          stepper.boundary_dense_output(make_not_null(&boundary_value),
                                        boundary_history, dense_test, coupling);
          CHECK(volume_value == local_approx(boundary_value));
        }
      }

      id = stepper.next_time_id(id, step);
      add_neighbor_entries(id);

      // Check normal output
      {
        double volume_value = 0.0;
        stepper.update_u(make_not_null(&volume_value), volume_history, step);
        double boundary_value = 0.0;
        stepper.add_boundary_delta(make_not_null(&boundary_value),
                                   boundary_history, step, coupling);
        CHECK(volume_value == local_approx(boundary_value));
      }
      stepper.clean_history(make_not_null(&volume_history));
      stepper.clean_boundary_history(make_not_null(&boundary_history));
    } while (id.substep() != 0);
    CHECK(dense_succeeded);
  }
}
}  // namespace

void test_equal_rate(const LtsTimeStepper& stepper) {
  INFO("test_equal_rate");
  const auto order_choice = fixed_order_choice(stepper);
  const std::deque<Rational> step_pattern{{1}, {1, 4}, {1, 4}, {1, 2}};
  arbitrary_compare_to_volume(stepper, true, step_pattern, step_pattern,
                              1.0e-15, order_choice);
  arbitrary_compare_to_volume(stepper, false, step_pattern, step_pattern,
                              1.0e-15, order_choice);
}

void test_uncoupled(const LtsTimeStepper& stepper, const double tolerance,
                    std::optional<VariableOrderChoice> variable_order_choice) {
  INFO("test_uncoupled");
  if (not variable_order_choice.has_value()) {
    variable_order_choice.emplace(fixed_order_choice(stepper));
  }
  const std::deque<Rational> step_pattern{{1}, {1, 4}, {1, 4}, {1, 2}};
  const std::deque<Rational> neighbor_step_pattern{{1, 4}, {1, 4}, {1, 2}, {1}};
  arbitrary_compare_to_volume(stepper, true, step_pattern,
                              neighbor_step_pattern, tolerance,
                              *variable_order_choice);
  arbitrary_compare_to_volume(stepper, false, step_pattern,
                              neighbor_step_pattern, tolerance,
                              *variable_order_choice);
}

namespace {
struct StepResult {
  TimeStepId id;
  size_t order;
  double value;
};

template <typename Coupling>
class Element {
 public:
  template <typename AnalyticSelf, typename AnalyticNeighbor>
  Element(const LtsTimeStepper& stepper,
          const VariableOrderChoice& order_choice, Coupling coupling,
          std::deque<Rational> step_pattern, const TimeStepId& time_step_id,
          const Rational& neighbor_first_step,
          const AnalyticSelf& analytic_self,
          const AnalyticNeighbor& analytic_neighbor)
      : stepper_(&stepper),
        coupling_(std::move(coupling)),
        step_pattern_(std::move(step_pattern)),
        time_step_id_(time_step_id),
        value_(analytic_self(time_step_id_.substep_time())),
        step_size_(next_step_size()),
        next_time_step_id_(stepper_->next_time_id(time_step_id_, step_size_)),
        volume_history_(order_choice.local_order),
        next_message_(stepper_->next_time_id(
            time_step_id_, neighbor_first_step *
                               time_step_id_.step_time().slab().duration())) {
    auto init_step = (time_step_id_.time_runs_forward() ? 1 : -1) *
                     time_step_id_.step_time().slab().duration();
    auto init_time = time_step_id_.step_time();
    int64_t slab_number = 0;
    while (boundary_history_.local().size() <
               order_choice.local_number_of_past_steps + 1 or
           boundary_history_.remote().size() <
               order_choice.remote_number_of_past_steps + 1) {
      const TimeStepId init_id(time_step_id_.time_runs_forward(), slab_number,
                               init_time);
      if (boundary_history_.local().size() <
          order_choice.local_number_of_past_steps + 1) {
        volume_history_.insert_initial(init_id,
                                       analytic_self(init_time.value()), 0.0);
        boundary_history_.local().insert_initial(
            init_id, order_choice.local_order,
            analytic_self(init_time.value()));
      }
      if (boundary_history_.remote().size() <
          order_choice.remote_number_of_past_steps + 1) {
        boundary_history_.remote().insert_initial(
            init_id, order_choice.remote_order,
            analytic_neighbor(init_time.value()));
      }
      --slab_number;
      init_step =
          init_step.with_slab(init_step.slab().advance_towards(-init_step));
      init_time -= init_step;
    }
  }

  bool done() const { return step_pattern_.empty(); }
  TimeStepId next_time_step_id() const { return next_time_step_id_; }
  Time next_step_time() const { return time_step_id_.step_time() + step_size_; }

  bool local_dense_output_ready(const double time) const {
    const evolution_less<double> before{time_step_id_.time_runs_forward()};
    if (not before(time, next_step_time().value())) {
      return false;
    }
    double dummy_value = 0.0;
    return stepper_->dense_update_u(make_not_null(&dummy_value),
                                    volume_history_, time);
  }

  double dense_output(const double time) {
    REQUIRE(process_messages(time));
    double dense_result = *volume_history_.step_start(time).value;
    // Can skip the volume update because all derivatives are zero.
    stepper_->boundary_dense_output(make_not_null(&dense_result),
                                    boundary_history_, time, coupling_);
    return dense_result;
  }

  std::optional<StepResult> step() {
    if (not process_messages(next_time_step_id_)) {
      return std::nullopt;
    }
    stepper_->update_u(make_not_null(&value_), volume_history_, step_size_);
    stepper_->add_boundary_delta(make_not_null(&value_), boundary_history_,
                                 step_size_, coupling_);
    stepper_->clean_history(make_not_null(&volume_history_));
    stepper_->clean_boundary_history(make_not_null(&boundary_history_));

    time_step_id_ = next_time_step_id_;
    if (time_step_id_.substep() == 0) {
      step_pattern_.pop_front();
      if (not done()) {
        step_size_ = next_step_size();
      }
      step_size_ = step_size_.with_slab(time_step_id_.step_time().slab());
    }
    next_time_step_id_ = stepper_->next_time_id(time_step_id_, step_size_);
    volume_history_.insert(time_step_id_, value_, 0.0);
    boundary_history_.local().insert(
        time_step_id_, volume_history_.integration_order(), value_);
    return {{time_step_id_, volume_history_.integration_order(), value_}};
  }

  void receive_data(const StepResult& message, const TimeStepId& next_id) {
    ASSERT(message.id == next_message_, "Test logic error.");
    messages_.emplace_back(message);
    next_message_ = next_id;
  }

 private:
  TimeDelta next_step_size() const {
    return step_pattern_.front() * time_step_id_.step_time().slab().duration();
  }

  template <typename T>
  bool process_messages(const T& time) {
    while (not messages_.empty() and
           stepper_->neighbor_data_required(time, messages_.front().id)) {
      boundary_history_.remote().insert(messages_.front().id,
                                        messages_.front().order,
                                        messages_.front().value);
      messages_.pop_front();
    }
    return not(messages_.empty() and
               stepper_->neighbor_data_required(time, next_message_));
  }

  gsl::not_null<const LtsTimeStepper*> stepper_;
  Coupling coupling_;

  std::deque<Rational> step_pattern_;
  TimeStepId time_step_id_;
  double value_;
  TimeDelta step_size_;
  TimeStepId next_time_step_id_;
  TimeSteppers::History<double> volume_history_{};
  TimeSteppers::BoundaryHistory<double, double, double> boundary_history_{};
  std::deque<StepResult> messages_{};
  TimeStepId next_message_;
};

// Test system:
// dx/dt = x y
// dy/dt = - x y
//
// Solution:
// x = c / [1 + exp(-c (t - d))]
// y = c - x
namespace product_system {
double rhs_x(const double x, const double y) { return x * y; }
double rhs_y(const double x, const double y) { return -x * y; }

// Arbitrary
constexpr double conserved_sum = 0.7;  // c
constexpr double offset = 0.4;  // d

double analytic_x(const double t) {
  return conserved_sum / (1.0 + exp(-conserved_sum * (t - offset)));
}
double analytic_y(const double t) { return conserved_sum - analytic_x(t); }
}  // namespace product_system

void test_conservation_impl(const LtsTimeStepper& stepper,
                            std::deque<Rational> step_pattern_x,
                            std::deque<Rational> step_pattern_y,
                            const VariableOrderChoice& order_choice) {
  const bool time_runs_forward = step_pattern_x.front() > 0;
  const evolution_less<Time> before{time_runs_forward};

  const Slab initial_slab(1.2, 3.4);
  const TimeStepId initial_id(
      time_runs_forward, 0,
      time_runs_forward ? initial_slab.start() : initial_slab.end());

  const auto first_step_x = step_pattern_x.front();
  Element element_x(
      stepper, order_choice,
      [](const double local, const double remote) {
        return product_system::rhs_x(local, remote);
      },
      std::move(step_pattern_x), initial_id, step_pattern_y.front(),
      product_system::analytic_x, product_system::analytic_y);
  Element element_y(
      stepper, flip(order_choice),
      [](const double local, const double remote) {
        return product_system::rhs_y(remote, local);
      },
      std::move(step_pattern_y), initial_id, first_step_x,
      product_system::analytic_y, product_system::analytic_x);

  double test_time = initial_id.step_time().value();

  for (;;) {
    const bool need_more_x = not element_x.local_dense_output_ready(test_time);
    const bool need_more_y = not element_y.local_dense_output_ready(test_time);
    // If an element can produce dense output, it must have
    // transmitted enough data for its neighbor to do the same.
    if (not(need_more_x or need_more_y)) {
      const double dense_x = element_x.dense_output(test_time);
      const double dense_y = element_y.dense_output(test_time);
      CHECK(dense_x + dense_y == approx(product_system::conserved_sum));
      test_time = std::min(element_x.next_step_time(),
                           element_y.next_step_time(), before)
                      .value();
      continue;
    }

    if (element_x.done() and element_y.done()) {
      break;
    }

    if (need_more_x) {
      const auto step_result = element_x.step();
      if (step_result.has_value()) {
        element_y.receive_data(*step_result, element_x.next_time_step_id());
        continue;
      }
    }

    if (need_more_y) {
      const auto step_result = element_y.step();
      if (step_result.has_value()) {
        element_x.receive_data(*step_result, element_y.next_time_step_id());
        continue;
      }
    }

    INFO("Deadlocked");
    REQUIRE(false);
  }
}

double test_convergence_error(const LtsTimeStepper& stepper,
                              const std::deque<Rational>& step_pattern_x,
                              const std::deque<Rational>& step_pattern_y,
                              const int32_t repeats, const bool test_dense,
                              const VariableOrderChoice& order_choice) {
  const bool time_runs_forward = step_pattern_x.front() > 0;

  const auto initial_slab =
      time_runs_forward ? Slab::with_duration_from_start(-10.0, 5.0 / repeats)
                        : Slab::with_duration_to_end(-5.0, 5.0 / repeats);
  // Arbitrary but not a nice rational portion of a slab.
  const double dense_time = -7.91237;

  const TimeStepId initial_id(
      time_runs_forward, 0,
      time_runs_forward ? initial_slab.start() : initial_slab.end());

  std::deque<Rational> full_pattern_x{};
  std::deque<Rational> full_pattern_y{};
  for (int32_t i = 0; i < repeats; ++i) {
    full_pattern_x.insert(full_pattern_x.end(), step_pattern_x.begin(),
                          step_pattern_x.end());
    full_pattern_y.insert(full_pattern_y.end(), step_pattern_y.begin(),
                          step_pattern_y.end());
  }

  Element element_x(
      stepper, order_choice,
      [](const double local, const double remote) {
        return product_system::rhs_x(local, remote);
      },
      std::move(full_pattern_x), initial_id, step_pattern_y.front(),
      product_system::analytic_x, product_system::analytic_y);
  Element element_y(
      stepper, flip(order_choice),
      [](const double local, const double remote) {
        return product_system::rhs_y(remote, local);
      },
      std::move(full_pattern_y), initial_id, step_pattern_x.front(),
      product_system::analytic_y, product_system::analytic_x);

  for (;;) {
    // Step y as much as possible so we don't have to worry about
    // having enough data on the x side for dense output.
    while (not element_y.done()) {
      const auto step_result = element_y.step();
      if (not step_result.has_value()) {
        break;
      }
      element_x.receive_data(*step_result, element_y.next_time_step_id());
    }

    if (test_dense and element_x.local_dense_output_ready(dense_time)) {
      return element_x.dense_output(dense_time) -
             product_system::analytic_x(dense_time);
    }

    {
      REQUIRE(not element_x.done());
      const auto step_result = element_x.step();
      if (step_result.has_value()) {
        if (not test_dense and element_x.done()) {
          return step_result->value -
                 product_system::analytic_x(step_result->id.substep_time());
        }
        element_y.receive_data(*step_result, element_x.next_time_step_id());
      }
    }
  }
}

void test_convergence_impl(
    const LtsTimeStepper& stepper,
    const std::pair<int32_t, int32_t>& number_of_steps_range,
    const int32_t stride, const bool dense,
    const VariableOrderChoice& order_choice) {
  const size_t expected_order =
      std::min(order_choice.local_order, order_choice.remote_order);
  CAPTURE(dense);
  std::deque<Rational> step_pattern_x{{1, 2}, {1, 8}, {1, 8}, {1, 4}};
  std::deque<Rational> step_pattern_y{{1, 8}, {1, 8}, {1, 4}, {1, 2}};
  CHECK(convergence_rate(
            number_of_steps_range, stride, [&](const int32_t repeats) {
              return test_convergence_error(stepper, step_pattern_x,
                                            step_pattern_y, repeats, dense,
                                            order_choice);
            }) == approx(expected_order).margin(0.4));
  alg::for_each(step_pattern_x, [](Rational& x) { x *= -1; });
  alg::for_each(step_pattern_y, [](Rational& x) { x *= -1; });
  CHECK(convergence_rate(
            number_of_steps_range, stride, [&](const int32_t repeats) {
              return test_convergence_error(stepper, step_pattern_x,
                                            step_pattern_y, repeats, dense,
                                            order_choice);
            }) == approx(expected_order).margin(0.4));
}
}  // namespace

void test_conservation(
    const LtsTimeStepper& stepper,
    std::optional<VariableOrderChoice> variable_order_choice) {
  if (not variable_order_choice.has_value()) {
    variable_order_choice.emplace(fixed_order_choice(stepper));
  }
  std::deque<Rational> step_pattern_x{{1}, {1, 4}, {1, 4}, {1, 2}};
  std::deque<Rational> step_pattern_y{{1, 4}, {1, 4}, {1, 2}, {1}};
  test_conservation_impl(stepper, step_pattern_x, step_pattern_y,
                         *variable_order_choice);
  alg::for_each(step_pattern_x, [](Rational& x) { x *= -1; });
  alg::for_each(step_pattern_y, [](Rational& x) { x *= -1; });
  test_conservation_impl(stepper, std::move(step_pattern_x),
                         std::move(step_pattern_y), *variable_order_choice);
}

void test_convergence(
    const LtsTimeStepper& stepper,
    const std::pair<int32_t, int32_t>& number_of_steps_range,
    const int32_t stride,
    std::optional<VariableOrderChoice> variable_order_choice) {
  if (not variable_order_choice.has_value()) {
    variable_order_choice.emplace(fixed_order_choice(stepper));
  }
  test_convergence_impl(stepper, number_of_steps_range, stride, false,
                        *variable_order_choice);
}

void test_dense_convergence(
    const LtsTimeStepper& stepper,
    const std::pair<int32_t, int32_t>& number_of_steps_range,
    const int32_t stride,
    std::optional<VariableOrderChoice> variable_order_choice) {
  if (not variable_order_choice.has_value()) {
    variable_order_choice.emplace(fixed_order_choice(stepper));
  }
  test_convergence_impl(stepper, number_of_steps_range, stride, true,
                        *variable_order_choice);
}
}  // namespace TimeStepperTestUtils::lts
