// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Cerk3.hpp"

#include <cmath>
#include <limits>

#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

size_t Cerk3::order() const { return 3; }

size_t Cerk3::error_estimate_order() const { return 2; }

uint64_t Cerk3::number_of_substeps() const { return 3; }

uint64_t Cerk3::number_of_substeps_for_error() const { return 3; }

size_t Cerk3::number_of_past_steps() const { return 0; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1).
double Cerk3::stable_step() const { return 1.2563726633091645; }

TimeStepId Cerk3::next_time_id(const TimeStepId& current_id,
                               const TimeDelta& time_step) const {
  const auto& step = current_id.substep();
  const auto& t0 = current_id.step_time();
  const auto& t = current_id.substep_time();
  if (step < number_of_substeps()) {
    if (step == 0) {
      ASSERT(t == t0, "In Cerk3 substep 0, the substep time ("
                          << t << ") should equal t0 (" << t0 << ")");
    } else {
      ASSERT(t == t0 + gsl::at(c_, step - 1) * time_step,
             "In Cerk3 substep "
                 << step << ", the substep time (" << t
                 << ") should equal t0+c[" << step - 1 << "]*dt ("
                 << t0 + gsl::at(c_, step - 1) * time_step << ")");
    }
    if (step < number_of_substeps() - 1) {
      return {current_id.time_runs_forward(), current_id.slab_number(), t0,
              step + 1, t0 + gsl::at(c_, step) * time_step};
    } else {
      return {current_id.time_runs_forward(), current_id.slab_number(),
              t0 + time_step};
    }
  } else {
    ERROR("In Cerk3 substep should be one of 0,1,2,3, not "
          << current_id.substep());
  }
}

TimeStepId Cerk3::next_time_id_for_error(const TimeStepId& current_id,
                                         const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr double Cerk3::a2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 2> Cerk3::a3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 3> Cerk3::a4_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 4> Cerk3::b1_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 4> Cerk3::b3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 4> Cerk3::b4_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 4> Cerk3::e_;
const std::array<Time::rational_t, 2> Cerk3::c_ = {{{12, 23}, {4, 5}}};
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Cerk3::my_PUP_ID =  // NOLINT
    0;
