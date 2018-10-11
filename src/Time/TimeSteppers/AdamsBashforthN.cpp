// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsBashforthN.hpp"

#include <algorithm>

#include "Time/TimeId.hpp"

namespace TimeSteppers {

AdamsBashforthN::AdamsBashforthN(const size_t order) noexcept : order_(order) {
  if (order_ < 1 or order_ > maximum_order) {
    ERROR("The order for Adams-Bashforth Nth order must be 1 <= order <= "
          << maximum_order);
  }
}

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

TimeId AdamsBashforthN::next_time_id(
    const TimeId& current_id,
    const TimeDelta& time_step) const noexcept {
  ASSERT(current_id.substep() == 0, "Adams-Bashforth should not have substeps");
  return {current_id.time_runs_forward(), current_id.slab_number(),
          current_id.time() + time_step};
}

std::vector<double> AdamsBashforthN::get_coefficients_impl(
    const std::vector<double>& steps) noexcept {
  const size_t order = steps.size();
  ASSERT(order >= 1 and order <= maximum_order, "Bad order" << order);
  if (std::all_of(steps.begin(), steps.end(),
                  [=](const double& s) { return s == 1.; })) {
    return constant_coefficients(order);
  }

  return variable_coefficients(steps);
}

std::vector<double> AdamsBashforthN::variable_coefficients(
    const std::vector<double>& steps) noexcept {
  const size_t order = steps.size();  // "k" in below equations

  // The `steps` vector contains the relative step sizes:
  //   steps = {dt_{n-k+1}/dt_n, ..., dt_n/dt_n}
  // Our goal is to calculate, for each j, the coefficient given by
  //   \int_0^1 dt ell_j(t; 1, (dt_n + dt_{n-1})/dt_n, ...,
  //                        (dt_n + ... + dt_{n-k+1})/dt_n)
  // (Where the ell_j are the Lagrange interpolating polynomials.)

  // Calculate coefficients of the numerators of the Lagrange interpolating
  // polynomial, in the standard form.
  std::vector<std::vector<double>> polynomials(order,
                                               std::vector<double>(order, 0.));
  for (auto& poly : polynomials) {
    poly[0] = 1.;
  }
  {
    double step_sum = 0.;
    for (size_t m = 0; m < order; ++m) {
      const double step = steps[order - m - 1];
      step_sum += step;
      for (size_t j = 0; j < order; ++j) {
        if (m == j) {
          continue;
        }
        auto& poly = polynomials[j];
        for (size_t i = m + (m > j ? 0 : 1); i > 0; --i) {
          poly[i] = poly[i - 1] - step_sum * poly[i];
        }
        poly[0] *= -step_sum;
      }
    }
  }

  // Calculate the denominators of the Lagrange interpolating polynomials.
  std::vector<double> denominators;
  denominators.reserve(order);
  for (size_t j = 0; j < order; ++j) {
    double denom = 1.;
    double step_sum = 0.;
    for (size_t m = 0; m < j; ++m) {
      const double step = steps[order - j + m - 1];
      step_sum += step;
      denom *= step_sum;
    }
    step_sum = 0.;
    for (size_t m = 0; m < order - j - 1; ++m) {
      const double step = steps[order - j - m - 2];
      step_sum += step;
      denom *= step_sum;
    }
    denominators.push_back(denom);
  }

  // At this point, the Lagrange interpolating polynomials are given by:
  //   ell_j(t; ...) = +/- sum_m t^m polynomials[j][m] / denominators[j]

  // Integrate, term by term.
  std::vector<double> result;
  result.reserve(order);
  double overall_sign = order % 2 == 0 ? -1. : 1.;
  for (size_t j = 0; j < order; ++j) {
    const auto& poly = polynomials[j];
    double integral = 0.;
    for (size_t i = 0; i < order; ++i) {
      integral += poly[i] / (i + 1);
    }
    result.push_back(overall_sign * integral / denominators[j]);
    overall_sign = -overall_sign;
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
