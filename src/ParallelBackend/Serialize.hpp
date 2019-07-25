// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the serialize and deserialize functions.

#pragma once

#include <vector>

#include <pup.h>

/*!
 * \ingroup ParallelGroup
 * \brief Serialize an object using PUP.
 *
 * The type to serialize as must be explicitly specified.  We require
 * this because a mismatch between the serialize and deserialize calls
 * causes undefined behavior and we do not want this to depend on
 * inferred types for safety.
 *
 * \tparam T type to serialize
 */
template <typename T, typename U>
std::vector<char> serialize(const U& obj) {
  static_assert(std::is_same<T, U>::value,
                "Explicit type for serialization differs from deduced type");
  const T& typed_obj = obj;
  // pup routine is non-const, but shouldn't modify anything in serialization
  // mode.
  // clang-tidy: do not use const_cast
  auto& mut_obj = const_cast<T&>(typed_obj);  // NOLINT

  PUP::sizer sizer;
  sizer | mut_obj;
  std::vector<char> data(sizer.size());
  PUP::toMem writer(data.data());
  writer | mut_obj;

  return data;
}

/*!
 * \ingroup ParallelGroup
 * \brief Deserialize an object using PUP.
 *
 * \tparam T the type to deserialize to
 */
template <typename T>
T deserialize(const void* const data) {
  PUP::fromMem reader(data);
  T result{};
  reader | result;
  return result;
}
