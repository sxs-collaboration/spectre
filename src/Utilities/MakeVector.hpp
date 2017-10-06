// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \brief Constructs a `std::vector` containing arguments passed in.
 *
 * This is useful as it allows in-place construction of a vector of non-copyable
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
      (((void)return_vector.emplace_back(std::forward<Args>(remaining_args))),
       0)...};
  return return_vector;
}

template <class T>
std::vector<T> make_vector() {
  return {};
}
