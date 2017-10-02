// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Utilities/TMPL.hpp"

namespace MakeVectorImpl {
/// \cond NEVER
template <class T>
struct tag {
  using type = T;
};
template <class T, size_t n>
struct get_tag : get_tag<typename T::type, n - 1> {};
template <class T>
struct get_tag<T, 0> : tag<T> {};

template <class Tag, size_t n = 1>
using type_t = typename get_tag<Tag, n>::type;
/// \endcond

// build the return type:
template <class T0, class... Ts>
using vector_types =
    type_t<std::conditional<std::is_same<T0, void>::value,
                            tmpl::front<tmpl::list<typename std::decay<Ts>...>>,
                            tag<T0>>,
           2>;
template <class T0, class... Ts>
using vector_return_type = std::vector<vector_types<T0, Ts...>>;
}  // namespace MakeVectorImpl

/*!
 * \brief Constructs a `std::vector` containing each of the arguments in 'ts'.
 *
 * This is useful as it allows in place construction of a vector of non-copyable
 * objects.
 *
 * \example
 * \snippet Utilities/Test_MakeVector.cpp make_vector_example
 */
template <class ValueType = void, class Arg1, class... Args>
auto make_vector(Arg1&& arg_1, Args&&... remaining_args) {
  std::vector<
      std::conditional_t<cpp17::is_same_v<ValueType, void>, Arg1, ValueType>>
      return_vector;
  return_vector.reserve(sizeof...(Args) + 1);
  return_vector.emplace_back(std::forward<Arg1>(arg_1));
  (void)std::initializer_list<int>{
      (((void)retval.emplace_back(std::forward<Ts>(ts))), 0)...};
  return retval;
}

template <class T>
std::vector<T> make_vector() {
  return {};
}
