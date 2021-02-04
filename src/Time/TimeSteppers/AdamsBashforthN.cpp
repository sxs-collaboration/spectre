// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsBashforthN.hpp"

#include <algorithm>

#include "Time/TimeStepId.hpp"
#include "Utilities/Math.hpp"

namespace TimeSteppers {

AdamsBashforthN::AdamsBashforthN(const size_t order) noexcept : order_(order) {
  if (order_ < 1 or order_ > maximum_order) {
    ERROR("The order for Adams-Bashforth Nth order must be 1 <= order <= "
          << maximum_order);
  }
}

size_t AdamsBashforthN::order() const noexcept { return order_; }

size_t AdamsBashforthN::number_of_past_steps() const noexcept {
  return order_ - 1;
}

double AdamsBashforthN::stable_step() const noexcept {
  if (order_ == 1) {
    return 1.;
  }

  // This is the condition that the characteristic polynomial of the
  // recurrence relation defined by the method has the correct sign at
  // -1.  It is not clear whether this is actually sufficient.
  const auto& coefficients = constant_coefficients(order_);
  double invstep = 0.;
  double sign = 1.;
  for (const auto coef : coefficients) {
    invstep += sign * coef;
    sign = -sign;
  }
  return 1. / invstep;
}

TimeStepId AdamsBashforthN::next_time_id(
    const TimeStepId& current_id,
    const TimeDelta& time_step) const noexcept {
  ASSERT(current_id.substep() == 0, "Adams-Bashforth should not have substeps");
  return {current_id.time_runs_forward(), current_id.slab_number(),
          current_id.step_time() + time_step};
}

std::vector<double> AdamsBashforthN::get_coefficients_impl(
    const std::vector<double>& steps) noexcept {
  const size_t order = steps.size();
  ASSERT(order >= 1 and order <= maximum_order, "Bad order" << order);
  if (std::all_of(steps.begin(), steps.end(),
                  [&steps](const double s) { return s == steps[0]; })) {
    return constant_coefficients(order);
  }

  return variable_coefficients(steps);
}

std::vector<double> AdamsBashforthN::variable_coefficients(
    const std::vector<double>& steps) noexcept {
  const size_t order = steps.size();  // "k" in below equations
  std::vector<double> result;
  result.reserve(order);

  // The `steps` vector contains the step sizes:
  //   steps = {dt_{n-k+1}, ..., dt_n}
  // Our goal is to calculate, for each j, the coefficient given by
  //   \int_0^1 dt ell_j(t dt_n; dt_n, dt_n + dt_{n-1}, ...,
  //                             dt_n + ... + dt_{n-k+1})
  // (Where the ell_j are the Lagrange interpolating polynomials.)

  std::vector<double> poly(order);
  double step_sum_j = 0.0;
  for (size_t j = 0; j < order; ++j) {
    // Calculate coefficients of the Lagrange interpolating polynomials,
    // in the standard a_0 + a_1 t + a_2 t^2 + ... form.
    std::fill(poly.begin(), poly.end(), 0.0);

    step_sum_j += steps[order - j - 1];
    poly[0] = 1.0;

    double step_sum_m = 0.0;
    for (size_t m = 0; m < order; ++m) {
      step_sum_m += steps[order - m - 1];
      if (m == j) {
        continue;
      }
      const double denom = 1.0 / (step_sum_j - step_sum_m);
      for (size_t i = m < j ? m + 1 : m; i > 0; --i) {
        poly[i] = (poly[i - 1] - poly[i] * step_sum_m) * denom;
      }
      poly[0] *= -step_sum_m * denom;
    }

    // Integrate p(t dt_n), term by term.
    for (size_t m = 0; m < order; ++m) {
      poly[m] /= m + 1.0;
    }
    result.push_back(evaluate_polynomial(poly, steps.back()));
  }
  return result;
}

std::vector<double> AdamsBashforthN::constant_coefficients(
    const size_t order) noexcept {
  switch (order) {
    case 1: return {1.};
    case 2: return {1.5, -0.5};
    case 3: return {23.0 / 12.0, -4.0 / 3.0, 5.0 / 12.0};
    case 4: return {55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -3.0 / 8.0};
    case 5: return {1901.0 / 720.0, -1387.0 / 360.0, 109.0 / 30.0,
          -637.0 / 360.0, 251.0 / 720.0};
    case 6: return {4277.0 / 1440.0, -2641.0 / 480.0, 4991.0 / 720.0,
          -3649.0 / 720.0, 959.0 / 480.0, -95.0 / 288.0};
    case 7: return {198721.0 / 60480.0, -18637.0 / 2520.0, 235183.0 / 20160.0,
          -10754.0 / 945.0, 135713.0 / 20160.0, -5603.0 / 2520.0,
          19087.0 / 60480.0};
    case 8: return {16083.0 / 4480.0, -1152169.0 / 120960.0, 242653.0 / 13440.0,
          -296053.0 / 13440.0, 2102243.0 / 120960.0, -115747.0 / 13440.0,
          32863.0 / 13440.0, -5257.0 / 17280.0};
    default:
      ERROR("Bad order: " << order);
  }
}

void AdamsBashforthN::pup(PUP::er& p) noexcept {
  LtsTimeStepper::Inherit::pup(p);
  p | order_;
}

bool operator==(const AdamsBashforthN& lhs,
                const AdamsBashforthN& rhs) noexcept {
  return lhs.order_ == rhs.order_;
}

bool operator!=(const AdamsBashforthN& lhs,
                const AdamsBashforthN& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace TimeSteppers

/// \cond
PUP::able::PUP_ID TimeSteppers::AdamsBashforthN::my_PUP_ID =  // NOLINT
    0;
/// \endcond
