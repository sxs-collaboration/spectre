// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeStepRequestProcessor.hpp"

#include <algorithm>
#include <optional>

#include "Time/EvolutionOrdering.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

TimeStepRequestProcessor::TimeStepRequestProcessor(const bool time_runs_forward)
    : time_runs_forward_(time_runs_forward) {
  const double inf = evolution_less<double>{time_runs_forward}.infinity();
  step_size_request_ = inf;
  step_end_request_ = inf;
  next_step_size_request_ = inf;
  next_step_end_request_ = inf;
  step_size_hard_limit_ = inf;
  step_end_hard_limit_ = inf;
}

std::optional<double> TimeStepRequestProcessor::new_step_size_goal() const {
  return step_size_goal_;
}

double TimeStepRequestProcessor::step_size(const double step_start,
                                           const double step_size_goal) const {
  const evolution_less<double> smaller{time_runs_forward_};

  const double goal = step_size_goal_.value_or(step_size_goal);
  const double size_from_end = step_end_request_ - step_start;

  if (can_use_larger_limit_) {
    // Both limits are from the same request
    const double request = std::max(step_size_request_, size_from_end, smaller);
    if (not(smaller(goal, request) or
            smaller(next_step_size_request_, request) or
            smaller(next_step_end_request_ - step_start, request))) {
      return request;
    }
  }

  return std::min({goal, step_size_request_, size_from_end}, smaller);
}

double TimeStepRequestProcessor::step_end(const double step_start,
                                          const double step_size_goal) const {
  const evolution_less<double> before{time_runs_forward_};

  const double goal = step_start + step_size_goal_.value_or(step_size_goal);
  const double end_from_size = step_start + step_size_request_;

  if (can_use_larger_limit_) {
    // Both limits are from the same request
    const double request = std::max(step_end_request_, end_from_size, before);
    if (not(before(goal, request) or
            before(step_start + next_step_size_request_, request) or
            before(next_step_end_request_, request))) {
      return request;
    }
  }

  return std::min({goal, step_end_request_, end_from_size}, before);
}

void TimeStepRequestProcessor::process(const TimeStepRequest& request) {
  // Earlier-time and smaller-step are the same operation on doubles,
  // but it's easier to read if they have different names.
  const evolution_less<double> before{time_runs_forward_};
  const evolution_less<double> smaller{time_runs_forward_};

  ASSERT(not(request.size_goal.has_value() and
             request.size.has_value() and
             smaller(*request.size_goal, *request.size)),
         "Requested a step size goal of " << *request.size_goal
         << ", but a particular step size larger than that goal: "
         << *request.size);

  TimeStepRequestProcessor request_postprocessor(time_runs_forward_);
  request_postprocessor.step_size_goal_ = request.size_goal;

  request_postprocessor.step_size_request_ =
      request.size.value_or(smaller.infinity());
  request_postprocessor.step_end_request_ =
      request.end.value_or(before.infinity());

  request_postprocessor.can_use_larger_limit_ =
      request.size.has_value() and request.end.has_value();

  request_postprocessor.step_size_hard_limit_ =
      request.size_hard_limit.value_or(smaller.infinity());
  request_postprocessor.step_end_hard_limit_ =
      request.end_hard_limit.value_or(before.infinity());

  *this += request_postprocessor;
}

TimeStepRequestProcessor& TimeStepRequestProcessor::operator+=(
    const TimeStepRequestProcessor& other) {
  ASSERT(time_runs_forward_ == other.time_runs_forward_,
         "Inconsistent time directions.");

  // Earlier-time and smaller-step are the same operation on doubles,
  // but it's easier to read if they have different names.
  const evolution_less<double> before{time_runs_forward_};
  const evolution_less<double> smaller{time_runs_forward_};

  if (smaller(other.step_size_goal_.value_or(smaller.infinity()),
              step_size_goal_.value_or(smaller.infinity()))) {
    step_size_goal_ = other.step_size_goal_;
  }

  if (smaller(step_size_request_, other.step_size_request_)) {
    next_step_size_request_ =
        std::min(next_step_size_request_, other.step_size_request_, smaller);
  } else if (smaller(other.step_size_request_, step_size_request_)) {
    next_step_size_request_ =
        std::min(step_size_request_, other.next_step_size_request_, smaller);
  } else {
    next_step_size_request_ = std::min(next_step_size_request_,
                                       other.next_step_size_request_, smaller);
  }
  if (before(step_end_request_, other.step_end_request_)) {
    next_step_end_request_ =
        std::min(next_step_end_request_, other.step_end_request_, before);
  } else if (before(other.step_end_request_, step_end_request_)) {
    next_step_end_request_ =
        std::min(step_end_request_, other.next_step_end_request_, before);
  } else {
    next_step_end_request_ =
        std::min(next_step_end_request_, other.next_step_end_request_, before);
  }

  const double new_size_request =
      std::min(step_size_request_, other.step_size_request_, smaller);
  const double new_end_request =
      std::min(step_end_request_, other.step_end_request_, before);
  if ((new_size_request == step_size_request_) ==
          (new_end_request == step_end_request_) and
      (new_size_request == other.step_size_request_) ==
          (new_end_request == other.step_end_request_)) {
    if (new_size_request == step_size_request_) {
      if (new_size_request == other.step_size_request_) {
        can_use_larger_limit_ =
            can_use_larger_limit_ and other.can_use_larger_limit_;
      } else {
        // use existing value
      }
    } else {
      if (new_size_request == other.step_size_request_) {
        can_use_larger_limit_ = other.can_use_larger_limit_;
      } else {
        can_use_larger_limit_ = false;
      }
    }
  } else {
    can_use_larger_limit_ = false;
  }
  step_size_request_ = new_size_request;
  step_end_request_ = new_end_request;

  step_size_hard_limit_ =
      std::min(step_size_hard_limit_, other.step_size_hard_limit_, smaller);
  step_end_hard_limit_ =
      std::min(step_end_hard_limit_, other.step_end_hard_limit_, before);

  return *this;
}

void TimeStepRequestProcessor::error_on_hard_limit(const double size,
                                                   const double end) const {
  // Earlier-time and smaller-step are the same operation on doubles,
  // but it's easier to read if they have different names.
  const evolution_less<double> before{time_runs_forward_};
  const evolution_less<double> smaller{time_runs_forward_};

  if (smaller(step_size_hard_limit_, size)) {
    ERROR("Could not adjust step below "
          << size << " to meet maximum step size " << step_size_hard_limit_);
  }
  if (before(step_end_hard_limit_, end)) {
    ERROR("Could not adjust step to before "
          << end << " to avoid exceeding time " << step_end_hard_limit_);
  }
}

void TimeStepRequestProcessor::pup(PUP::er& p) {
  p | time_runs_forward_;
  p | step_size_goal_;
  p | step_size_request_;
  p | step_end_request_;
  p | can_use_larger_limit_;
  p | next_step_size_request_;
  p | next_step_end_request_;
  p | step_size_hard_limit_;
  p | step_end_hard_limit_;
}

bool operator==(const TimeStepRequestProcessor& a,
                const TimeStepRequestProcessor& b) {
  return a.time_runs_forward_ == b.time_runs_forward_ and
         a.step_size_goal_ == b.step_size_goal_ and
         a.step_size_request_ == b.step_size_request_ and
         a.step_end_request_ == b.step_end_request_ and
         a.can_use_larger_limit_ == b.can_use_larger_limit_ and
         a.next_step_size_request_ == b.next_step_size_request_ and
         a.next_step_end_request_ == b.next_step_end_request_ and
         a.step_size_hard_limit_ == b.step_size_hard_limit_ and
         a.step_end_hard_limit_ == b.step_end_hard_limit_;
}

bool operator!=(const TimeStepRequestProcessor& a,
                const TimeStepRequestProcessor& b) {
  return not(a == b);
}

TimeStepRequestProcessor operator+(const TimeStepRequestProcessor& a,
                                   const TimeStepRequestProcessor& b) {
  auto result = a;
  result += b;
  return result;
}
