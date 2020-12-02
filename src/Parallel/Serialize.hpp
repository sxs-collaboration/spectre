// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the serialize and deserialize functions.

#pragma once

#include <pup.h>
#include <vector>

#include "Utilities/Gsl.hpp"

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
template <typename T>
std::vector<char> serialize(const T& obj) noexcept {
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
T deserialize(const void* const data) noexcept {  // NOLINT
  // clang-tidy: no const in forward decl (this is a definition)
  PUP::fromMem reader(data);
  T result{};
  reader | result;
  return result;
}

/*!
 * \ingroup ParallelGroup
 * \brief Deserialize an object using PUP.
 *
 * \tparam T the type to deserialize to.
 */
template <typename T>
void deserialize(const gsl::not_null<T*> result,
                 const void* const data) noexcept {  // NOLINT
  // clang-tidy: no const in forward decl (this is a definition)
  PUP::fromMem reader(data);
  reader | *result;
}
