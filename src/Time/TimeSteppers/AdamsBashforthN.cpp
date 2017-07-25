// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsBashforthN.hpp"

#include <algorithm>

#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

AdamsBashforthN::AdamsBashforthN(size_t target_order, bool self_start) noexcept
    : target_order_(target_order), is_self_starting_(self_start) {
  if (target_order_ < 1 or target_order_ > maximum_order) {
    ERROR("The order for Adams-Bashforth Nth order must be 1 <= order <= "
          << maximum_order);
  }
}

size_t AdamsBashforthN::number_of_substeps() const noexcept {
  return 1;
}

size_t AdamsBashforthN::number_of_past_steps() const noexcept {
  return target_order_ - 1;
}

bool AdamsBashforthN::is_self_starting() const noexcept {
  return is_self_starting_;
}

double AdamsBashforthN::stable_step() const noexcept {
  if (target_order_ == 1) {
    return 1.;
  }

  // This is the condition that the characteristic polynomial of the
  // recurrence relation defined by the method has the correct sign at
  // -1.  It is not clear whether this is actually sufficient.
  const auto& coefficients = gsl::at(coefficients_, target_order_ - 2);
  double invstep = 0.;
  double sign = 1.;
  for (const auto coef : coefficients) {
    invstep += sign * coef;
    sign = -sign;
  }
  return 1. / invstep;
}

std::vector<double> AdamsBashforthN::get_coefficients_impl(
    const std::vector<double>& steps) noexcept {
  const size_t order = steps.size();
  ASSERT(order >= 1 and order <= maximum_order, "Bad order" << order);
  if (order == 1) {
    return {1.};
  }
  if (std::all_of(steps.begin(), steps.end(),
                  [=](const double& s) { return s == 1.; })) {
    return gsl::at(coefficients_, order - 2);
  }

  return variable_coefficients(steps);
}

std::vector<double> AdamsBashforthN::variable_coefficients(
    const std::vector<double>& steps) noexcept {
  const size_t order = steps.size();

  // polynomials[coefficient][term exponent]
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

const std::array<std::vector<double>, AdamsBashforthN::maximum_order>
AdamsBashforthN::coefficients_{{
    {1.5, -0.5},                                           // 2nd order
    {23.0 / 12.0, -4.0 / 3.0, 5.0 / 12.0},                 // 3rd order
    {55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -3.0 / 8.0},  // 4th order
    {1901.0 / 720.0, -1387.0 / 360.0, 109.0 / 30.0, -637.0 / 360.0,
     251.0 / 720.0},  // 5th order
    {4277.0 / 1440.0, -2641.0 / 480.0, 4991.0 / 720.0, -3649.0 / 720.0,
     959.0 / 480.0, -95.0 / 288.0},  // 6th order
    {198721.0 / 60480.0, -18637.0 / 2520.0, 235183.0 / 20160.0,
     -10754.0 / 945.0, 135713.0 / 20160.0, -5603.0 / 2520.0,
     19087.0 / 60480.0},  // 7th order
    {16083.0 / 4480.0, -1152169.0 / 120960.0, 242653.0 / 13440.0,
     -296053.0 / 13440.0, 2102243.0 / 120960.0, -115747.0 / 13440.0,
     32863.0 / 13440.0, -5257.0 / 17280.0},  // 8th order
}};

}  // namespace TimeSteppers
