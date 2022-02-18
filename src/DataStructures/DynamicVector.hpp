// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// `blaze::DynamicVector` is a general-purpose dynamically sized vector type.
/// This file implements interoperability of `blaze::DynamicVector` with our
/// data structures.

#pragma once

#include <blaze/math/DynamicVector.h>
#include <pup.h>
#include <type_traits>
#include <vector>

#include "Options/ParseOptions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace PUP {
/// @{
/// Serialization of blaze::DynamicVector
template <typename T, bool TF, typename Tag>
void pup(er& p, blaze::DynamicVector<T, TF, Tag>& t) {
  size_t size = t.size();
  p | size;
  if (p.isUnpacking()) {
    t.resize(size);
  }
  if (std::is_fundamental_v<T>) {
    PUParray(p, t.data(), size);
  } else {
    for (T& element : t) {
      p | element;
    }
  }
}
template <typename T, bool TF, typename Tag>
void operator|(er& p, blaze::DynamicVector<T, TF, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP

namespace MakeWithValueImpls {
template <typename T, bool TF, typename Tag>
struct NumberOfPoints<blaze::DynamicVector<T, TF, Tag>> {
  static SPECTRE_ALWAYS_INLINE size_t
  apply(const blaze::DynamicVector<T, TF, Tag>& input) {
    return input.size();
  }
};

template <typename T, bool TF, typename Tag>
struct MakeWithSize<blaze::DynamicVector<T, TF, Tag>> {
  static SPECTRE_ALWAYS_INLINE blaze::DynamicVector<T, TF, Tag> apply(
      const size_t size, const T& value) {
    return blaze::DynamicVector<T, TF, Tag>(size, value);
  }
};
}  // namespace MakeWithValueImpls

template <typename T, bool TF, typename Tag>
struct Options::create_from_yaml<blaze::DynamicVector<T, TF, Tag>> {
  template <typename Metavariables>
  static blaze::DynamicVector<T, TF, Tag> create(
      const Options::Option& options) {
    const auto data = options.parse_as<std::vector<T>>();
    blaze::DynamicVector<T, TF, Tag> result(data.size());
    std::copy(std::begin(data), std::end(data), result.begin());
    return result;
  }
};
