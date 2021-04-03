// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/IsCallable.hpp"

namespace {
class A {
 public:
  int operator()() { return 1; }
};

class B {
 public:
  double operator()(const int /* unused */, const double /* unused */) {
    return 0.;
  }
};
}  // namespace

/// [is_callable_example]
static_assert(not tt::is_callable<A, int, double>::value,
              "Failed testing type trait is_callable");
static_assert(not tt::is_callable<A, int>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable<A>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable<B, int, double>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable_t<B, int, double>::value,
              "Failed testing type trait is_callable");
static_assert(tt::is_callable_v<B, int, double>,
              "Failed testing type trait is_callable");
static_assert(not tt::is_callable<B>::value,
              "Failed testing type trait is_callable");
/// [is_callable_example]
