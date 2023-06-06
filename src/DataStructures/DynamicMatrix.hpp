// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// `blaze::DynamicMatrix` is a general-purpose dynamically sized matrix type.
/// This file implements interoperability of `blaze::DynamicMatrix` with our
/// data structures.

#pragma once

#include <blaze/math/DynamicMatrix.h>
#include <cstddef>
#include <pup.h>
#include <type_traits>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Gsl.hpp"

namespace PUP {
/// @{
/// Serialization of blaze::DynamicMatrix
template <typename Type, bool SO, typename Alloc, typename Tag>
void pup(er& p, blaze::DynamicMatrix<Type, SO, Alloc, Tag>& t) {
  size_t rows = t.rows();
  size_t columns = t.columns();
  p | rows;
  p | columns;
  if (p.isUnpacking()) {
    t.resize(rows, columns);
  }
  const size_t spacing = t.spacing();
  const size_t data_size = spacing * (SO == blaze::rowMajor ? rows : columns);
  if (std::is_fundamental_v<Type>) {
    PUParray(p, t.data(), data_size);
  } else {
    for (size_t i = 0; i < data_size; ++i) {
      p | t.data()[i];
    }
  }
}
template <typename Type, bool SO, typename Alloc, typename Tag>
void operator|(er& p, blaze::DynamicMatrix<Type, SO, Alloc, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP

namespace DynamicMatrix_detail {
template <typename Type>
std::vector<std::vector<Type>> parse_to_vectors(const Options::Option& options);
}  // namespace DynamicMatrix_detail

template <typename Type, bool SO, typename Alloc, typename Tag>
struct Options::create_from_yaml<blaze::DynamicMatrix<Type, SO, Alloc, Tag>> {
  template <typename Metavariables>
  static blaze::DynamicMatrix<Type, SO, Alloc, Tag> create(
      const Options::Option& options) {
    const auto data = DynamicMatrix_detail::parse_to_vectors<Type>(options);
    const size_t num_rows = data.size();
    size_t num_cols = 0;
    if (num_rows > 0) {
      num_cols = data[0].size();
    }
    blaze::DynamicMatrix<Type, SO, Alloc, Tag> result(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; i++) {
      const auto& row = gsl::at(data, i);
      if (row.size() != num_cols) {
        PARSE_ERROR(options.context(),
                    "All matrix rows must have the same size.");
      }
      std::copy(row.begin(), row.end(), blaze::row(result, i).begin());
    }
    return result;
  }
};
