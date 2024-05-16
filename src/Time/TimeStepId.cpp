// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeStepId.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <pup.h>

#include "Time/EvolutionOrdering.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

TimeStepId::TimeStepId(const bool time_runs_forward, const int64_t slab_number,
                       const Time& time)
    : slab_number_(slab_number),
      step_time_(time),
      step_size_(time_runs_forward ? 1 : -1),
      substep_time_(time.value()) {
  canonicalize();
}

TimeStepId::TimeStepId(const bool time_runs_forward, const int64_t slab_number,
                       const Time& step_time, const uint64_t substep,
                       const TimeDelta& step_size, const double substep_time)
    : slab_number_(slab_number),
      step_time_(step_time),
      substep_(substep),
      step_size_(substep == 0 ? (time_runs_forward ? 1 : -1)
                              : step_size.fraction()),
      substep_time_(substep_time) {
  ASSERT(substep <= std::numeric_limits<decltype(substep_)>::max(),
         "Overflow in substep: " << substep);
  ASSERT(substep_ != 0 or step_time_.value() == substep_time_,
         "Initial substep must align with the step.");
  ASSERT(substep_ == 0 or step_time.slab() == step_size.slab(),
         "Time and step have different slabs.");
  ASSERT(substep == 0 or time_runs_forward == step_size.is_positive(),
         "Step size has wrong sign: " << step_size);
  if (time_runs_forward) {
    ASSERT(substep_time_ >= step_time_.value(),
           "Substep must be within the step.");
  } else {
    ASSERT(substep_time_ <= step_time_.value(),
           "Substep must be within the step.");
  }
  canonicalize();
}

bool TimeStepId::time_runs_forward() const { return step_size_ > 0; }

TimeDelta TimeStepId::step_size() const {
  ASSERT(substep_ != 0, "Step size not available at substep 0.");
  return {step_time_.slab(), step_size_};
}

bool TimeStepId::is_at_slab_boundary() const {
  return substep_ == 0 and step_time_.is_at_slab_boundary();
}

TimeStepId TimeStepId::next_step(const TimeDelta& step_size) const {
  ASSERT(substep_ == 0 or step_size == this->step_size(),
         "Step size inconsistent: " << this->step_size() << " " << step_size);
  return {time_runs_forward(), slab_number_, step_time_ + step_size};
}

TimeStepId TimeStepId::next_substep(const TimeDelta& step_size,
                                    const double step_fraction) const {
  ASSERT(substep_ == 0 or step_size == this->step_size(),
         "Step size inconsistent: " << this->step_size() << " " << step_size);
  ASSERT(step_fraction >= 0.0 and step_fraction <= 1.0,
         "Substep must be within the step.");
  const double new_time = (1.0 - step_fraction) * step_time_.value() +
                          step_fraction * (step_time_ + step_size).value();
  return {time_runs_forward(), slab_number_, step_time_, substep() + 1,
          step_size, new_time};
}

void TimeStepId::canonicalize() {
  if (time_runs_forward() ? step_time_.is_at_slab_end()
                          : step_time_.is_at_slab_start()) {
    ASSERT(substep_ == 0,
           "Time needs to be advanced, but step already started");
    const Slab new_slab = time_runs_forward() ? step_time_.slab().advance()
                                              : step_time_.slab().retreat();
    ++slab_number_;
    step_time_ = step_time_.with_slab(new_slab);
    ASSERT(substep_time_ == step_time_.value(),
           "Time changed during canonicalization");
  }
}

void TimeStepId::pup(PUP::er& p) {
  p | slab_number_;
  p | step_time_;
  p | substep_;
  p | step_size_;
  p | substep_time_;
}

bool operator==(const TimeStepId& a, const TimeStepId& b) {
  ASSERT(a.time_runs_forward() == b.time_runs_forward(),
         "Time is not running in a consistent direction");
  bool equal = a.slab_number() == b.slab_number() and
               a.step_time() == b.step_time() and a.substep() == b.substep();
  if (equal and a.substep() != 0) {
    equal = a.step_size() == b.step_size();
  }
  ASSERT(not equal or a.substep_time() == b.substep_time(),
         "IDs at same step and substep but different times");
  return equal;
}

bool operator!=(const TimeStepId& a, const TimeStepId& b) {
  return not(a == b);
}

bool operator<(const TimeStepId& a, const TimeStepId& b) {
  ASSERT(a.time_runs_forward() == b.time_runs_forward(),
         "Time is not running in a consistent direction");
  if (a.slab_number() != b.slab_number()) {
    return a.slab_number() < b.slab_number();
  }
  if (a.step_time() != b.step_time()) {
    return evolution_less<Time>{a.time_runs_forward()}(a.step_time(),
                                                       b.step_time());
  }
  if (a.substep() != b.substep()) {
    return a.substep() < b.substep();
  }
  if (a.substep() == 0) {
    // Objects are equal.
    return false;
  }
  // Arbitrary, but need a total ordering of TimeStepId to use it as a
  // std::map key.
  return a.step_size() < b.step_size();
}
bool operator<=(const TimeStepId& a, const TimeStepId& b) { return not(b < a); }
bool operator>(const TimeStepId& a, const TimeStepId& b) { return b < a; }
bool operator>=(const TimeStepId& a, const TimeStepId& b) { return not(a < b); }

std::ostream& operator<<(std::ostream& s, const TimeStepId& id) {
  return s << id.slab_number() << ':' << id.step_time() << ':' << id.substep()
           << ':' << id.substep_time();
}

size_t hash_value(const TimeStepId& id) {
  size_t h = 0;
  boost::hash_combine(h, id.slab_number());
  boost::hash_combine(h, id.step_time());
  boost::hash_combine(h, id.substep());
  if (id.substep() != 0) {
    boost::hash_combine(h, id.step_size());
    boost::hash_combine(h, id.substep_time());
  }
  return h;
}

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<TimeStepId>::operator()(const TimeStepId& id) const {
  return boost::hash<TimeStepId>{}(id);
}
}  // namespace std
