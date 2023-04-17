// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>

#include "Time/ApproximateTime.hpp"

#define DEFINE_OP(lprefix, rprefix, ret, op, lhs, rhs)            \
  ret operator op(const lprefix##lhs& a, const rprefix##rhs& b) { \
    return {a.value() op b.value()};                              \
  }

#define DEFINE_ALL_OPS(lprefix, rprefix)                                     \
  DEFINE_OP(lprefix, rprefix, bool, ==, Time, Time)                          \
  DEFINE_OP(lprefix, rprefix, bool, !=, Time, Time)                          \
  DEFINE_OP(lprefix, rprefix, bool, <, Time, Time)                           \
  DEFINE_OP(lprefix, rprefix, bool, >, Time, Time)                           \
  DEFINE_OP(lprefix, rprefix, bool, <=, Time, Time)                          \
  DEFINE_OP(lprefix, rprefix, bool, >=, Time, Time)                          \
  DEFINE_OP(lprefix, rprefix, bool, ==, TimeDelta, TimeDelta)                \
  DEFINE_OP(lprefix, rprefix, bool, !=, TimeDelta, TimeDelta)                \
  DEFINE_OP(lprefix, rprefix, bool, <, TimeDelta, TimeDelta)                 \
  DEFINE_OP(lprefix, rprefix, bool, >, TimeDelta, TimeDelta)                 \
  DEFINE_OP(lprefix, rprefix, bool, <=, TimeDelta, TimeDelta)                \
  DEFINE_OP(lprefix, rprefix, bool, >=, TimeDelta, TimeDelta)                \
  DEFINE_OP(lprefix, rprefix, ApproximateTimeDelta, -, Time, Time)           \
  DEFINE_OP(lprefix, rprefix, ApproximateTime, +, Time, TimeDelta)           \
  DEFINE_OP(lprefix, rprefix, ApproximateTime, +, TimeDelta, Time)           \
  DEFINE_OP(lprefix, rprefix, ApproximateTime, -, Time, TimeDelta)           \
  DEFINE_OP(lprefix, rprefix, ApproximateTimeDelta, +, TimeDelta, TimeDelta) \
  DEFINE_OP(lprefix, rprefix, ApproximateTimeDelta, -, TimeDelta, TimeDelta) \
  DEFINE_OP(lprefix, rprefix, double, /, TimeDelta, TimeDelta)

#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbraced-scalar-init"
#endif  /* __clang__ */
DEFINE_ALL_OPS(Approximate, )
DEFINE_ALL_OPS(, Approximate)
DEFINE_ALL_OPS(Approximate, Approximate)
#ifdef __clang__
#pragma GCC diagnostic pop
#endif  /* __clang__ */

#undef DEFINE_ALL_OPS
#undef DEFINE_OP

ApproximateTimeDelta operator+(const ApproximateTimeDelta& a) { return a; }
ApproximateTimeDelta operator-(const ApproximateTimeDelta& a) {
  return {-a.value()};
}

ApproximateTimeDelta operator*(const ApproximateTimeDelta& a,
                               const TimeDelta::rational_t& b) {
  return {a.value() * b.value()};
}
ApproximateTimeDelta operator*(const TimeDelta::rational_t& a,
                               const ApproximateTimeDelta& b) {
  return {a.value() * b.value()};
}
ApproximateTimeDelta operator/(const ApproximateTimeDelta& a,
                               const TimeDelta::rational_t& b) {
  return {a.value() * b.inverse().value()};
}

std::ostream& operator<<(std::ostream& os, const ApproximateTime& t) {
  return os << t.value();
}

ApproximateTimeDelta abs(const ApproximateTimeDelta& t) {
  return {std::abs(t.value())};
}

std::ostream& operator<<(std::ostream& os, const ApproximateTimeDelta& dt) {
  return os << dt.value();
}
