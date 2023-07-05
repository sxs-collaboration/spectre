// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// `blaze::StaticVector` is a general-purpose fixed size vector type. This file
/// implements interoperability of `blaze::StaticVector` with our data
/// structures.

#pragma once

#include <array>
#include <blaze/math/StaticVector.h>
#include <pup.h>
#include <type_traits>

#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace PUP {
/// @{
/// Serialization of blaze::StaticVector
template <typename T, size_t N, bool TF, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
void pup(er& p, blaze::StaticVector<T, N, TF, AF, PF, Tag>& t) {
  if (std::is_fundamental_v<T>) {
    PUParray(p, t.data(), N);
  } else {
    for (T& element : t) {
      p | element;
    }
  }
}
template <typename T, size_t N, bool TF, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
void operator|(er& p, blaze::StaticVector<T, N, TF, AF, PF, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP

namespace MakeWithValueImpls {
template <typename T, size_t N, bool TF, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
struct NumberOfPoints<blaze::StaticVector<T, N, TF, AF, PF, Tag>> {
  static constexpr size_t apply(
      const blaze::StaticVector<T, N, TF, AF, PF, Tag>& /*input*/) {
    return N;
  }
};

template <typename T, size_t N, bool TF, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
struct MakeWithSize<blaze::StaticVector<T, N, TF, AF, PF, Tag>> {
  static SPECTRE_ALWAYS_INLINE blaze::StaticVector<T, N, TF, AF, PF, Tag> apply(
      const size_t size, const T& value) {
    ASSERT(size == N, "Size mismatch for StaticVector: Expected "
                          << N << ", got " << size << ".");
    return blaze::StaticVector<T, N, TF, AF, PF, Tag>(value);
  }
};
}  // namespace MakeWithValueImpls

template <typename T, size_t N, bool TF, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<
    blaze::StaticVector<T, N, TF, AF, PF, Tag>> {
  static constexpr bool is_trivial = false;
  static SPECTRE_ALWAYS_INLINE void apply(
      const gsl::not_null<blaze::StaticVector<T, N, TF, AF, PF, Tag>*>
      /*result*/,
      const size_t size) {
    ERROR("Tried to resize a StaticVector to " << size);
  }
};

template <typename T, size_t N, bool TF, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
struct Options::create_from_yaml<blaze::StaticVector<T, N, TF, AF, PF, Tag>> {
  template <typename Metavariables>
  static blaze::StaticVector<T, N, TF, AF, PF, Tag> create(
      const Options::Option& options) {
    return {options.parse_as<std::array<T, N>>()};
  }
};
