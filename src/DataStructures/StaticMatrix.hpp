// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// `blaze::StaticMatrix` is a general-purpose fixed size matrix type. This file
/// implements interoperability of `blaze::StaticMatrix` with our data
/// structures.

#pragma once

#include <array>
#include <blaze/math/StaticMatrix.h>
#include <cstddef>
#include <pup.h>
#include <type_traits>

#include "Options/ParseOptions.hpp"

namespace PUP {
/// @{
/// Serialization of blaze::StaticMatrix
template <typename Type, size_t M, size_t N, bool SO, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
void pup(er& p, blaze::StaticMatrix<Type, M, N, SO, AF, PF, Tag>& t) {
  const size_t spacing = t.spacing();
  const size_t data_size = spacing * (SO == blaze::rowMajor ? M : N);
  if (std::is_fundamental_v<Type>) {
    PUParray(p, t.data(), data_size);
  } else {
    for (size_t i = 0; i < data_size; ++i) {
      p | t.data()[i];
    }
  }
}
template <typename Type, size_t M, size_t N, bool SO, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
void operator|(er& p, blaze::StaticMatrix<Type, M, N, SO, AF, PF, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP

template <typename Type, size_t M, size_t N, bool SO, blaze::AlignmentFlag AF,
          blaze::PaddingFlag PF, typename Tag>
struct Options::create_from_yaml<
    blaze::StaticMatrix<Type, M, N, SO, AF, PF, Tag>> {
  template <typename Metavariables>
  static blaze::StaticMatrix<Type, M, N, SO, AF, PF, Tag> create(
      const Options::Option& options) {
    return {options.parse_as<std::array<std::array<Type, N>, M>>()};
  }
};
