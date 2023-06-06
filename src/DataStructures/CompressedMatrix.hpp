// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// `blaze::CompressedMatrix` is a general-purpose sparse matrix type. This file
/// implements interoperability of `blaze::CompressedMatrix` with our data
/// structures.

#pragma once

#include <blaze/math/CompressedMatrix.h>
#include <cstddef>
#include <pup.h>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Gsl.hpp"

namespace PUP {
/// @{
/// Serialization of blaze::CompressedMatrix
template <typename Type, bool SO, typename Tag>
void pup(er& p, blaze::CompressedMatrix<Type, SO, Tag>& t) {
  size_t rows = t.rows();
  size_t columns = t.columns();
  p | rows;
  p | columns;
  const size_t first_dimension = (SO == blaze::rowMajor) ? rows : columns;
  size_t num_non_zeros = t.nonZeros();
  p | num_non_zeros;
  // blaze::CompressedMatrix has no `.data()` access, so we use the low-level
  // `append` mechanism for serialization instead of `PUParray`. Maybe there's
  // an even faster way using PUPbytes.
  size_t index;
  if (p.isUnpacking()) {
    t.resize(rows, columns);
    t.reserve(num_non_zeros);
    Type value;
    for (size_t i = 0; i < first_dimension; ++i) {
      p | num_non_zeros;
      for (size_t j = 0; j < num_non_zeros; ++j) {
        p | index;
        p | value;
        if constexpr (SO == blaze::rowMajor) {
          t.append(i, index, value);
        } else {
          t.append(index, i, value);
        }
      }
      t.finalize(i);
    }
  } else {
    for (size_t i = 0; i < first_dimension; ++i) {
      num_non_zeros = t.nonZeros(i);
      p | num_non_zeros;
      for (auto it = t.begin(i); it != t.end(i); ++it) {
        index = it->index();
        p | index;
        p | it->value();
      }
    }
  }
}
template <typename Type, bool SO, typename Tag>
void operator|(er& p, blaze::CompressedMatrix<Type, SO, Tag>& t) {
  pup(p, t);
}
/// @}
}  // namespace PUP

namespace CompressedMatrix_detail {
template <typename Type>
std::vector<std::vector<Type>> parse_to_vectors(const Options::Option& options);
}  // namespace CompressedMatrix_detail

template <typename Type, bool SO, typename Tag>
struct Options::create_from_yaml<blaze::CompressedMatrix<Type, SO, Tag>> {
  template <typename Metavariables>
  static blaze::CompressedMatrix<Type, SO, Tag> create(
      const Options::Option& options) {
    const auto data = CompressedMatrix_detail::parse_to_vectors<Type>(options);
    const size_t num_rows = data.size();
    size_t num_cols = 0;
    if (num_rows > 0) {
      num_cols = data[0].size();
    }
    blaze::CompressedMatrix<Type, SO, Tag> result(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; i++) {
      const auto& row = gsl::at(data, i);
      if (row.size() != num_cols) {
        PARSE_ERROR(options.context(),
                    "All matrix rows must have the same size.");
      }
      for (size_t j = 0; j < num_cols; j++) {
        if (gsl::at(row, j) != 0.) {
          result(i, j) = gsl::at(row, j);
        }
      }
    }
    return result;
  }
};
