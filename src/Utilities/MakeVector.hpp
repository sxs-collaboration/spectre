// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <initializer_list>
#include <utility>
#include <vector>

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
template <class ValueType = void, class Arg0, class... Args>
auto make_vector(Arg0&& arg_0, Args&&... remaining_args) {
  std::vector<
      std::conditional_t<cpp17::is_same_v<ValueType, void>, Arg0, ValueType>>
      return_vector;
  return_vector.reserve(sizeof...(Args) + 1);
  return_vector.emplace_back(std::forward<Arg0>(arg_0));
  (void)std::initializer_list<int>{
      (((void)return_vector.emplace_back(std::forward<Args>(remaining_args))),
       0)...};
  return return_vector;
}

template <class T>
std::vector<T> make_vector() {
  return {};
}
