// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace {
/// [CREATE_IS_CALLABLE_EXAMPLE]
CREATE_IS_CALLABLE(foo)
CREATE_IS_CALLABLE_V(foo)
CREATE_IS_CALLABLE_R_V(foo)
CREATE_IS_CALLABLE(foobar)
CREATE_IS_CALLABLE_V(foobar)
CREATE_IS_CALLABLE_R_V(foobar)
struct bar {
  size_t foo(int /*unused*/, double /*unused*/) { return size_t{0}; }
};

static_assert(is_foo_callable_v<bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(is_foo_callable_r_v<size_t, bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foo_callable_v<bar, int>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foo_callable_v<bar>, "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foobar_callable_r_v<size_t, bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foo_callable_r_v<int, bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
static_assert(not is_foobar_callable_v<bar, int, double>,
              "Failed testing CREATE_IS_CALLABLE");
/// [CREATE_IS_CALLABLE_EXAMPLE]
}  // namespace
