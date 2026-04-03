// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the serialize and deserialize functions.

#pragma once

#include <pup.h>
#include <type_traits>
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
std::vector<char> serialize(const T& obj) {
  const T& typed_obj = obj;
  // pup routine is non-const, but shouldn't modify anything in serialization
  // mode.
  // clang-tidy: do not use const_cast
  auto& mut_obj = const_cast<T&>(typed_obj);  // NOLINT

#if defined(SPECTRE_USE_CHARM)
  PUP::sizer sizer;
#elif defined(SPECTRE_USE_FINDUS)
  findus::serialize::Serializer sizer{findus::serialize::Serializer::Sizing};
#endif  // SPECTRE_USE_CHARM
  sizer | mut_obj;
#if defined(SPECTRE_USE_CHARM)
  std::vector<char> data(sizer.size());
  PUP::toMem writer(data.data());
#elif defined(SPECTRE_USE_FINDUS)
  std::vector<char> data(sizer.number_of_bytes());
  findus::serialize::Serializer writer{
      findus::serialize::Serializer::Packing,
      reinterpret_cast<std::byte*>(data.data()), sizer.number_of_bytes()};
#endif  // SPECTRE_USE_CHARM
  writer | mut_obj;

  return data;
}

/*!
 * \ingroup ParallelGroup
 * \brief Deserialize an object using PUP.
 *
 * \tparam T the type to deserialize to.
 */
template <typename T>
void deserialize(const gsl::not_null<T*> result,
                 const void* const data) {  // NOLINT
  // clang-tidy: no const in forward decl (this is a definition)
#if defined(SPECTRE_USE_CHARM)
  PUP::fromMem reader(data);
#elif defined(SPECTRE_USE_FINDUS)
  // Note: because Charm++ didn't bounds check we can't do so here right now. We
  // would need to pass in the number of points that data points to. That would
  // be a good future improvement.
  findus::serialize::Serializer reader{
      findus::serialize::Serializer::Unpacking,
      reinterpret_cast<const std::byte* const>(data), 0};
#endif  // SPECTRE_USE_CHARM
  reader | *result;
}

/*!
 * \ingroup ParallelGroup
 * \brief Deserialize an object using PUP.
 *
 * \tparam T the type to deserialize to
 */
template <typename T>
T deserialize(const void* const data) {  // NOLINT
  // clang-tidy: no const in forward decl (this is a definition)
  T result{};
  deserialize(make_not_null(&result), data);
  return result;
}

/*!
 * \ingroup ParallelGroup
 * \brief Returns the size of an object in bytes
 */
template <typename T>
size_t size_of_object_in_bytes(const T& obj) {
#if defined(SPECTRE_USE_CHARM)
  PUP::sizer sizer;
  sizer | const_cast<T&>(obj);  // NOLINT
  return sizer.size();
#elif defined(SPECTRE_USE_FINDUS)
  findus::serialize::Serializer sizer{findus::serialize::Serializer::Sizing};
  sizer | const_cast<T&>(obj);  // NOLINT
  return sizer.number_of_bytes();
#endif  // SPECTRE_USE_CHARM
}

/*!
 * \ingroup ParallelGroup
 * \brief Serializes and deserializes an object `t` of type `T`
 *
 * Performs a PUP round-trip on `t`, returning the deserialized copy.
 * Useful for testing that a class is correctly serializable, and for
 * copying objects that store state in a way only accessible via PUP.
 *
 * \requires `T` to be default constructible.
 */
template <typename T>
T serialize_and_deserialize(const T& t) {
  static_assert(
      std::is_default_constructible_v<T>,
      "Cannot use serialize_and_deserialize if a class is not default "
      "constructible.");
  return deserialize<T>(serialize<T>(t).data());
}

/*!
 * \ingroup ParallelGroup
 * \brief Serializes and deserializes an object `t` of type `T` into `result`
 *
 * Performs a PUP round-trip on `t`, writing the deserialized value into
 * the pre-existing object pointed to by `result`.
 *
 * Unlike the value-returning overload, `T` does not need to be default
 * constructible, since the result is written into the pre-existing `*result`.
 */
template <typename T>
void serialize_and_deserialize(const gsl::not_null<T*> result, const T& t) {
  deserialize<T>(result, serialize<T>(t).data());
}
