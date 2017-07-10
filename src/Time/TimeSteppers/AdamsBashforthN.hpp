// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class AdamsBashforthN

#pragma once

#include <array>
#include <boost/iterator/transform_iterator.hpp>
#include <deque>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// An Nth Adams-Bashforth time stepper.
class AdamsBashforthN : public TimeStepper::Inherit {
 public:
  static constexpr const size_t maximum_order = 8;

  // struct TargetOrder {
  //   using type = size_t;
  //   static constexpr OptionString_t label = {"TargetOrder"};
  //   static constexpr OptionString_t help = {
  //       "Target order of Adams-Bashforth method."};
  //   static type lower_bound() { return 1; }
  //   static type upper_bound() { return maximum_order; }
  // };

  // struct SelfStart {
  //   using type = bool;
  //   static constexpr OptionString_t label = {"SelfStart"};
  //   static constexpr OptionString_t help = {
  //       "If true then the method starts at first order and increases."};
  //   static type default_value() { return false; }
  // };
  // using OptionsList = tmpl::list<TargetOrder, SelfStart>;
  // static std::string class_id() { return "AdamsBashforthN"; }
  // static constexpr OptionString_t help = {
  //     "An Adams-Bashforth Nth order time-stepper. The target order is the "
  //     "order of the method. If a self-starting approach is chosen then the "
  //     "method starts at first order and increases the step-size until the "
  //     "desired order is reached."};

  AdamsBashforthN(size_t target_order, bool self_start) noexcept;
  AdamsBashforthN(const AdamsBashforthN&) noexcept = default;
  AdamsBashforthN& operator=(const AdamsBashforthN&) noexcept = default;
  AdamsBashforthN(AdamsBashforthN&&) noexcept = default;
  AdamsBashforthN& operator=(AdamsBashforthN&&) noexcept = default;
  ~AdamsBashforthN() noexcept override = default;

  template <typename Vars, typename DerivVars>
  TimeDelta update_u(
      Vars& u,
      const std::deque<std::tuple<Time, Vars, DerivVars>>& history,
      const TimeDelta& time_step) const noexcept;

  size_t number_of_substeps() const noexcept override;

  size_t number_of_past_steps() const noexcept override;

  bool is_self_starting() const noexcept override;

  double stable_step() const noexcept override;

 private:
  /// Get coefficients for a time step.  Arguments are an iterator
  /// pair to past times, oldest to newest, and the time step to take.
  template <typename Iterator>
  static std::vector<double> get_coefficients(const Iterator& times_begin,
                                              const Iterator& times_end,
                                              const TimeDelta& step) noexcept;

  static std::vector<double> get_coefficients_impl(
      const std::vector<double>& steps) noexcept;

  static std::vector<double> variable_coefficients(
      const std::vector<double>& steps) noexcept;

  static const std::array<std::vector<double>, maximum_order> coefficients_;

  size_t target_order_ = 3;
  bool is_self_starting_ = true;
};

template <typename Vars, typename DerivVars>
TimeDelta AdamsBashforthN::update_u(
    Vars& u,
    const std::deque<std::tuple<Time, Vars, DerivVars>>& history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(is_self_starting_ or target_order_ == history.size(),
         "Length of history should be the order, so "
         << target_order_ << ", but is: " << history.size());
  ASSERT(history.size() <= target_order_,
         "Length of history (" << history.size() << ") "
         << "should not exceed target order (" << target_order_ << ")");

  // Cast the overloaded get function to a function pointer taking a
  // history entry to a time reference.  This is needed to prevent the
  // boost::make_transform_iterator calls from being ambiguous.
  const Time& (*const get_time)(const decltype(history[0])&) = std::get<0>;
  const auto& coefficients = get_coefficients(
      boost::make_transform_iterator(history.cbegin(), get_time),
      boost::make_transform_iterator(history.cend(), get_time), time_step);

  switch (history.size()) {
    case 1: {
      u += time_step.value() * std::get<2>(history[0]);
      return time_step;
    }
    case 2: {
      u += time_step.value() * (coefficients[1] * std::get<2>(history[0]) +
                                coefficients[0] * std::get<2>(history[1]));
      return time_step;
    }
    case 3: {
      u += time_step.value() * (coefficients[2] * std::get<2>(history[0]) +
                                coefficients[1] * std::get<2>(history[1]) +
                                coefficients[0] * std::get<2>(history[2]));
      return time_step;
    }
    case 4: {
      u += time_step.value() * (coefficients[3] * std::get<2>(history[0]) +
                                coefficients[2] * std::get<2>(history[1]) +
                                coefficients[1] * std::get<2>(history[2]) +
                                coefficients[0] * std::get<2>(history[3]));
      return time_step;
    }
    case 5: {
      u += time_step.value() * (coefficients[4] * std::get<2>(history[0]) +
                                coefficients[3] * std::get<2>(history[1]) +
                                coefficients[2] * std::get<2>(history[2]) +
                                coefficients[1] * std::get<2>(history[3]) +
                                coefficients[0] * std::get<2>(history[4]));
      return time_step;
    }
    case 6: {
      u += time_step.value() * (coefficients[5] * std::get<2>(history[0]) +
                                coefficients[4] * std::get<2>(history[1]) +
                                coefficients[3] * std::get<2>(history[2]) +
                                coefficients[2] * std::get<2>(history[3]) +
                                coefficients[1] * std::get<2>(history[4]) +
                                coefficients[0] * std::get<2>(history[5]));
      return time_step;
    }
    case 7: {
      u += time_step.value() * (coefficients[6] * std::get<2>(history[0]) +
                                coefficients[5] * std::get<2>(history[1]) +
                                coefficients[4] * std::get<2>(history[2]) +
                                coefficients[3] * std::get<2>(history[3]) +
                                coefficients[2] * std::get<2>(history[4]) +
                                coefficients[1] * std::get<2>(history[5]) +
                                coefficients[0] * std::get<2>(history[6]));
      return time_step;
    }
    case 8: {
      u += time_step.value() * (coefficients[7] * std::get<2>(history[0]) +
                                coefficients[6] * std::get<2>(history[1]) +
                                coefficients[5] * std::get<2>(history[2]) +
                                coefficients[4] * std::get<2>(history[3]) +
                                coefficients[3] * std::get<2>(history[4]) +
                                coefficients[2] * std::get<2>(history[5]) +
                                coefficients[1] * std::get<2>(history[6]) +
                                coefficients[0] * std::get<2>(history[7]));
      return time_step;
    }
    default:
      ERROR("Bad amount of history data: " << history.size());
  }
}

template <typename Iterator>
std::vector<double> AdamsBashforthN::get_coefficients(
    const Iterator& times_begin, const Iterator& times_end,
    const TimeDelta& step) noexcept {
  std::vector<double> steps;
  // This may be slightly more space than we need, but we can't get
  // the exact amount without iterating through the iterators, which
  // is not necessarily cheap depending on the iterator type.
  steps.reserve(maximum_order);
  for (auto t = times_begin; std::next(t) != times_end; ++t) {
    steps.push_back((*std::next(t) - *t) / step);
  }
  steps.push_back(1.);
  return get_coefficients_impl(steps);
}

}  // namespace TimeSteppers
