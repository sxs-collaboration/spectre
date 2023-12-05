// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace db::detail {
CREATE_IS_CALLABLE(apply)
CREATE_IS_CALLABLE_V(apply)

template <typename Func, typename... Args>
constexpr void error_function_not_callable() {
  static_assert(
      std::is_same_v<Func, void>,
      "The function is not callable with the expected arguments.  "
      "See the first template parameter of "
      "error_function_not_callable for the function or object type and "
      "the remaining arguments for the parameters that cannot be "
      "passed. If all the argument types match, it could be that you "
      "have a template parameter that cannot be deduced."
      "Note that for most DataBox functions, you must pass either "
      "a function pointer, a lambda, or a class with a call operator "
      "or static apply function, and this error will also arise if "
      "the provided entity does not satisfy that requirement (e.g. "
      "if the provided class defines a function with the incorrect name).");
}
}  // namespace db::detail
