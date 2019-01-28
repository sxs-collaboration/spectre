// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function transpose

#pragma once

#include <cstddef>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \ingroup NumericalAlgorithmsGroup
/// \brief Function to compute transposed data.
///
/// Transpose the data pointed to by `data`, writing the result to the
/// location pointed to by `result`.  See the ::transpose function for
/// a safer interface and for the meaning of the other arguments.
// This is not an overload of transpose because overload resolution
// would make non-intuitive choices of the return-by-pointer version
// below over this function.
template <typename T>
void raw_transpose(const gsl::not_null<T*> result, const T* const data,
                   const size_t chunk_size,
                   const size_t number_of_chunks) noexcept {
  // The i outside loop order is faster, but that could be architecture
  // dependent and so may need updating in the future. Changing this made the
  // logical derivatives in 3D with 50 variables 20% faster.
  for (size_t i = 0; i < chunk_size; ++i) {
    for (size_t j = 0; j < number_of_chunks; ++j) {
      // clang-tidy: pointer arithmetic
      result.get()[j + number_of_chunks * i] =  // NOLINT
          data[i + chunk_size * j];             // NOLINT
    }
  }
}

// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Function to compute transposed data.
///
/// The primary use of this function is to rearrange the memory layout so that
/// another function can operate on contiguous chunks of memory.
///
/// \requires `result.size()` to be the product of `number_of_chunks` and
/// `chunk_size`, `u.size()` to be equal or greater than `result.size()`,
/// and that both `result` and `u` have a `data()` member function.
///
/// \details The container `u` holds a contiguous array of data,
/// treated as a sequence of `number_of_chunks` contiguous sets of
/// entries of size `chunk_size`.  The output `result` has its data
/// arranged such that the first `number_of_chunks` elements in
/// `result` will be the first element of each chunk of `u`.  The
/// last `number_of_chunks` elements in `result` will be the last
/// (i.e.  `chunk_size`-th) element of each chunk of `u`.  If
/// `u.size()` is greater than `result.size()` the extra elements
/// of `u` are ignored.
///
/// \note This is equivalent to treating the first part of `u` as a matrix and
/// filling `result` (or the returned object) with the transpose of that
/// matrix.
///
/// If `u` represents a block of data indexed by
/// \f$(x, y, z, \ldots)\f$ with the first index varying fastest,
/// transpose serves to rotate the indices.  If the extents are
/// \f$(X, Y, Z, \ldots)\f$, with product \f$N\f$, `transpose(u, X,
/// N/X)` reorders the data to be indexed \f$(y, z, \ldots, x)\f$,
/// `transpose(u, X*Y, N/X/Y)` reorders the data to be indexed
/// \f$(z, \ldots, x, y)\f$, etc.
///
/// \example
/// \snippet Test_Transpose.cpp transpose_matrix
/// \snippet Test_Transpose.cpp transpose_by_not_null_example
/// \snippet Test_Transpose.cpp return_transpose_example
/// \snippet Test_Transpose.cpp partial_transpose_example
///
/// \tparam U the type of data to be transposed
/// \tparam T the type of the transposed data
template <typename U, typename T>
void transpose(const gsl::not_null<T*> result, const U& u,
               const size_t chunk_size,
               const size_t number_of_chunks) noexcept {
  ASSERT(chunk_size * number_of_chunks == result->size(),
         "chunk_size = " << chunk_size << ", number_of_chunks = "
                         << number_of_chunks << ", size = " << result->size());
  ASSERT(result->size() <= u.size(),
         "result.size = " << result->size() << ", u.size = " << u.size());
  raw_transpose(make_not_null(result->data()), u.data(), chunk_size,
                number_of_chunks);
}

template <typename U, typename T = U>
T transpose(const U& u, const size_t chunk_size,
            const size_t number_of_chunks) noexcept {
  T t = make_with_value<T>(u, 0.0);
  transpose(make_not_null(&t), u, chunk_size, number_of_chunks);
  return t;
}
// @}
