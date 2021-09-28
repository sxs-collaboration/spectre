// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeStepId.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

#include "Time/EvolutionOrdering.hpp"
#include "Time/Slab.hpp"

void TimeStepId::canonicalize() {
  if (time_runs_forward_ ? step_time_.is_at_slab_end()
                         : step_time_.is_at_slab_start()) {
    ASSERT(substep_ == 0,
           "Time needs to be advanced, but step already started");
    const Slab new_slab = time_runs_forward_ ? substep_time_.slab().advance()
                                             : substep_time_.slab().retreat();
    ++slab_number_;
    substep_time_ = substep_time_.with_slab(new_slab);
    step_time_ = substep_time_;
  }
}

void TimeStepId::pup(PUP::er& p) {
  p | time_runs_forward_;
  p | slab_number_;
  p | step_time_;
  p | substep_;
  p | substep_time_;
}

bool operator==(const TimeStepId& a, const TimeStepId& b) {
  ASSERT(a.time_runs_forward() == b.time_runs_forward(),
         "Time is not running in a consistent direction");
  const bool equal = a.slab_number() == b.slab_number() and
                     a.step_time() == b.step_time() and
                     a.substep() == b.substep();
  // This could happen if we have a local-time-stepping substep
  // method.  If we implement any of those the comparison operators
  // for TimeStepId will have to be revisited.
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
  return a.slab_number() < b.slab_number() or
         (a.slab_number() == b.slab_number() and
          (evolution_less<Time>{a.time_runs_forward()}(a.step_time(),
                                                       b.step_time()) or
           (a.step_time() == b.step_time() and a.substep() < b.substep())));
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
  boost::hash_combine(h, id.substep_time());
  return h;
}

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<TimeStepId>::operator()(const TimeStepId& id) const {
  return boost::hash<TimeStepId>{}(id);
}
}  // namespace std
