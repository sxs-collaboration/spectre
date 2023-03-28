// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// `blaze::CompressedVector` is a general-purpose sparse vector type. This file
/// implements interoperability of `blaze::CompressedVector` with our data
/// structures.

#pragma once

#include <blaze/math/CompressedVector.h>
#include <cstddef>
#include <pup.h>
#include <vector>

#include "Utilities/Gsl.hpp"

/// \cond
namespace Options {
struct Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace PUP {
/// @{
/// Serialization of blaze::CompressedVector
template <typename T, bool TF, typename Tag>
void pup(er& p, blaze::CompressedVector<T, TF, Tag>& t) {
  size_t size = t.size();
  p | size;
  size_t num_non_zeros = t.nonZeros();
  p | num_non_zeros;
  // blaze::CompressedVector has no `.data()` access, so we use the low-level
  // `append` mechanism for serialization instead of `PUParray`. Maybe there's
  // an even faster way using PUPbytes.
  size_t index;
  if (p.isUnpacking()) {
    t.resize(size);
    t.reserve(num_non_zeros);
    T value;
    for (size_t i = 0; i < num_non_zeros; ++i) {
      p | index;
      p | value;
      t.append(index, value);
    }
  } else {
    for (auto it = t.begin(); it != t.end(); ++it) {
      index = it->index();
      p | index;
      p | it->value();
    }
  }
}
template <typename T, bool TF, typename Tag>
void operator|(er& p, blaze::CompressedVector<T, TF, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP

namespace CompressedVector_detail {
template <typename T>
std::vector<T> parse_to_vector(const Options::Option& options);
}  // namespace CompressedVector_detail

template <typename T, bool TF, typename Tag>
struct Options::create_from_yaml<blaze::CompressedVector<T, TF, Tag>> {
  template <typename Metavariables>
  static blaze::CompressedVector<T, TF, Tag> create(
      const Options::Option& options) {
    const auto data = CompressedVector_detail::parse_to_vector<T>(options);
    blaze::CompressedVector<T, TF, Tag> result(data.size());
    // Insert only non-zero elements. Can't use iterators and `std::copy`
    // because for sparse types the iterators only run over non-zero elements.
    // There's probably a faster way to do this construction using the low-level
    // `append` function, but it's probably not worth the effort for small
    // matrices created in input files.
    for (size_t i = 0; i < data.size(); ++i) {
      if (gsl::at(data, i) != 0.) {
        result[i] = gsl::at(data, i);
      }
    }
    return result;
  }
};
