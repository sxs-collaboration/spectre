// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "Time/Time.hpp"

/// \ingroup TimeGroup
/// Time-like interface to a double for use with dense output
struct ApproximateTime {
  double time = std::numeric_limits<double>::signaling_NaN();
  double value() const { return time; }
};

/// \ingroup TimeGroup
/// TimeDelta-like interface to a double for use with dense output
struct ApproximateTimeDelta {
  double delta = std::numeric_limits<double>::signaling_NaN();
  double value() const { return delta; }
  bool is_positive() const { return delta > 0.; }
};

// Duplicate most of the interface for Time and TimeDelta.  Leave out
// the mutation functions, as these are only supposed to be used for
// temporary calculations.

#define DECLARE_OP(lprefix, rprefix, ret, op, lhs, rhs) \
  ret operator op(const lprefix##lhs& a, const rprefix##rhs& b);

#define DECLARE_ALL_OPS(lprefix, rprefix)                                     \
  DECLARE_OP(lprefix, rprefix, bool, ==, Time, Time)                          \
  DECLARE_OP(lprefix, rprefix, bool, !=, Time, Time)                          \
  DECLARE_OP(lprefix, rprefix, bool, <, Time, Time)                           \
  DECLARE_OP(lprefix, rprefix, bool, >, Time, Time)                           \
  DECLARE_OP(lprefix, rprefix, bool, <=, Time, Time)                          \
  DECLARE_OP(lprefix, rprefix, bool, >=, Time, Time)                          \
  DECLARE_OP(lprefix, rprefix, bool, ==, TimeDelta, TimeDelta)                \
  DECLARE_OP(lprefix, rprefix, bool, !=, TimeDelta, TimeDelta)                \
  DECLARE_OP(lprefix, rprefix, bool, <, TimeDelta, TimeDelta)                 \
  DECLARE_OP(lprefix, rprefix, bool, >, TimeDelta, TimeDelta)                 \
  DECLARE_OP(lprefix, rprefix, bool, <=, TimeDelta, TimeDelta)                \
  DECLARE_OP(lprefix, rprefix, bool, >=, TimeDelta, TimeDelta)                \
  DECLARE_OP(lprefix, rprefix, ApproximateTimeDelta, -, Time, Time)           \
  DECLARE_OP(lprefix, rprefix, ApproximateTime, +, Time, TimeDelta)           \
  DECLARE_OP(lprefix, rprefix, ApproximateTime, +, TimeDelta, Time)           \
  DECLARE_OP(lprefix, rprefix, ApproximateTime, -, Time, TimeDelta)           \
  DECLARE_OP(lprefix, rprefix, ApproximateTimeDelta, +, TimeDelta, TimeDelta) \
  DECLARE_OP(lprefix, rprefix, ApproximateTimeDelta, -, TimeDelta, TimeDelta) \
  DECLARE_OP(lprefix, rprefix, double, /, TimeDelta, TimeDelta)

DECLARE_ALL_OPS(Approximate, )
DECLARE_ALL_OPS(, Approximate)
DECLARE_ALL_OPS(Approximate, Approximate)

#undef DECLARE_ALL_OPS
#undef DECLARE_OP

ApproximateTimeDelta operator+(const ApproximateTimeDelta& a);
ApproximateTimeDelta operator-(const ApproximateTimeDelta& a);

ApproximateTimeDelta operator*(const ApproximateTimeDelta& a,
                               const TimeDelta::rational_t& b);
ApproximateTimeDelta operator*(const TimeDelta::rational_t& a,
                               const ApproximateTimeDelta& b);
ApproximateTimeDelta operator/(const ApproximateTimeDelta& a,
                               const TimeDelta::rational_t& b);

std::ostream& operator<<(std::ostream& os, const ApproximateTime& t);

ApproximateTimeDelta abs(const ApproximateTimeDelta& t);

std::ostream& operator<<(std::ostream& os, const ApproximateTimeDelta& dt);
