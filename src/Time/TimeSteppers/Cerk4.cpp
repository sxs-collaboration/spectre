// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Cerk4.hpp"

#include <cmath>
#include <limits>

#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {
Cerk4::Cerk4(CkMigrateMessage* /*msg*/) {}

size_t Cerk4::order() const { return 4; }

size_t Cerk4::error_estimate_order() const { return 3; }

uint64_t Cerk4::number_of_substeps() const { return 5; }

uint64_t Cerk4::number_of_substeps_for_error() const { return 5; }

size_t Cerk4::number_of_past_steps() const { return 0; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1). For the fourth order method:
//  alpha_5 = 5 (1 - 2 c_3) c_4
// The stability limit as compared to a forward Euler method is given by finding
// the root for |p(-2 z)|-1=0. For forward Euler this is 1.0.
double Cerk4::stable_step() const { return 1.4367588951002057; }

TimeStepId Cerk4::next_time_id(const TimeStepId& current_id,
                               const TimeDelta& time_step) const {
  const auto& step = current_id.substep();
  const auto& t0 = current_id.step_time();
  const auto& t = current_id.substep_time();
  if (step < number_of_substeps()) {
    if (step == 0) {
      ASSERT(t == t0, "In Cerk4 substep 0, the substep time ("
                          << t << ") should equal t0 (" << t0 << ")");
    } else {
      ASSERT(t == t0 + gsl::at(c_, step - 1) * time_step,
             "In Cerk4 substep "
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
    ERROR("In Cerk4 substep should be one of 0,1,2,3,4,5, not "
          << current_id.substep());
  }
}

TimeStepId Cerk4::next_time_id_for_error(const TimeStepId& current_id,
                                         const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr double Cerk4::a2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 2> Cerk4::a3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 3> Cerk4::a4_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 4> Cerk4::a5_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk4::a6_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk4::b1_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr double Cerk4::b2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk4::b3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk4::b4_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk4::b5_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk4::b6_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk4::e_;
const std::array<Time::rational_t, 4> Cerk4::c_ = {
    {{1, 6}, {11, 37}, {11, 17}, {13, 15}}};
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Cerk4::my_PUP_ID =  // NOLINT
    0;
