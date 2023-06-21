// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Transpose.hpp"
#include <type_traits>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace {
// We assume matrix points to the start of the sub matrix.
//
// Streaming writes didn't improve things with AVX2 on Zen2
// architecture. For AVX-512 streaming might be more useful, but AVX-512 is
// usually not recommended as of 2023 because the CPUs down clock so much
// that all non-math work also suffers.
template <size_t RowsInBlock, size_t ColumnsInBlock>
void transpose_block(double* matrix_transpose, const double* matrix,
                     int32_t columns, int32_t rows);

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
#if defined(__AVX__)
template <>
void transpose_block<4, 4>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256d row0 = _mm256_loadu_pd(matrix + 0 * columns);
  const __m256d row1 = _mm256_loadu_pd(matrix + 1 * columns);
  const __m256d row2 = _mm256_loadu_pd(matrix + 2 * columns);
  const __m256d row3 = _mm256_loadu_pd(matrix + 3 * columns);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row3), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row3), 0b1111);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_storeu_pd(matrix_transpose + 1 * rows,
                   _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_storeu_pd(matrix_transpose + 2 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
  _mm256_storeu_pd(matrix_transpose + 3 * rows,
                   _mm256_permute2f128_pd(tmp2, tmp3, 0x31));
}

template <>
void transpose_block<3, 4>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_loadu_pd(matrix + 0 * columns);
  const __m256d row1 = _mm256_loadu_pd(matrix + 1 * columns);
  const __m256d row2 = _mm256_loadu_pd(matrix + 2 * columns);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row2), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row2), 0b1111);

  _mm256_maskstore_pd(matrix_transpose + 0 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 1 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 2 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
  _mm256_maskstore_pd(matrix_transpose + 3 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x31));
}

template <>
void transpose_block<2, 4>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256d row0 = _mm256_loadu_pd(matrix + 0 * columns);
  const __m256d row1 = _mm256_loadu_pd(matrix + 1 * columns);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd((row0), (row1), 0b1111);

  _mm_storeu_pd(matrix_transpose + 0 * rows, _mm256_extractf128_pd(tmp0, 0));
  _mm_storeu_pd(matrix_transpose + 1 * rows, _mm256_extractf128_pd(tmp1, 0));
  _mm_storeu_pd(matrix_transpose + 2 * rows, _mm256_extractf128_pd(tmp0, 1));
  _mm_storeu_pd(matrix_transpose + 3 * rows, _mm256_extractf128_pd(tmp1, 1));
}

template <>
void transpose_block<1, 4>(double* matrix_transpose, const double* const matrix,
                           const int32_t /*columns*/, const int32_t rows) {
  const __m256d row0 = _mm256_loadu_pd(matrix);

  const __m256d tmp0 = _mm256_shuffle_pd(row0, row0, 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd(row0, row0, 0b1111);

  const __m128i store_mask = _mm_set_epi64x(0, -1);
  _mm_maskstore_pd(matrix_transpose + 0 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp0));
  _mm_maskstore_pd(matrix_transpose + 1 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp1));
  _mm_maskstore_pd(matrix_transpose + 2 * rows, store_mask,
                   _mm256_extractf128_pd(tmp0, 1));
  _mm_maskstore_pd(matrix_transpose + 3 * rows, store_mask,
                   _mm256_extractf128_pd(tmp1, 1));
}

template <>
void transpose_block<4, 3>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);
  const __m256d row1 = _mm256_maskload_pd(matrix + 1 * columns, mask);
  const __m256d row2 = _mm256_maskload_pd(matrix + 2 * columns, mask);
  const __m256d row3 = _mm256_maskload_pd(matrix + 3 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row3), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row3), 0b1111);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_storeu_pd(matrix_transpose + 1 * rows,
                   _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_storeu_pd(matrix_transpose + 2 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
}

template <>
void transpose_block<4, 2>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256d rows0_1 = _mm256_permute2f128_pd(
      _mm256_castpd128_pd256(_mm_load_pd(matrix + 0 * columns)),
      _mm256_castpd128_pd256(_mm_load_pd(matrix + 2 * columns)), 0b00100000);
  const __m256d rows2_3 = _mm256_permute2f128_pd(
      _mm256_castpd128_pd256(_mm_load_pd(matrix + 1 * columns)),
      _mm256_castpd128_pd256(_mm_load_pd(matrix + 3 * columns)), 0b00100000);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_unpacklo_pd(rows0_1, rows2_3));
  _mm256_storeu_pd(matrix_transpose + 1 * rows,
                   _mm256_unpackhi_pd(rows0_1, rows2_3));
}

template <>
void transpose_block<4, 1>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  // We load the 4 rows into SSE registers, and then combine them into a
  // single AVX register for write.
  const __m128d row0 = _mm_load_pd1(matrix + 0 * columns);
  const __m128d row1 = _mm_load_pd1(matrix + 1 * columns);
  const __m128d row2 = _mm_load_pd1(matrix + 2 * columns);
  const __m128d row3 = _mm_load_pd1(matrix + 3 * columns);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_insertf128_pd(
                       _mm256_castpd128_pd256(_mm_shuffle_pd(row0, row1, 0b00)),
                       _mm_shuffle_pd(row2, row3, 0b00), 1));
}

template <>
void transpose_block<3, 3>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);
  const __m256d row1 = _mm256_maskload_pd(matrix + 1 * columns, mask);
  const __m256d row2 = _mm256_maskload_pd(matrix + 2 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row2), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row2), 0b1111);

  _mm256_maskstore_pd(matrix_transpose + 0 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 1 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 2 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
}

template <>
void transpose_block<3, 2>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 =
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 0 * columns));
  const __m256d row1 =
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 1 * columns));
  const __m256d row2 =
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 2 * columns));

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row2), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row2), 0b1111);

  _mm256_maskstore_pd(matrix_transpose + 0 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 1 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
}

template <>
void transpose_block<3, 1>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  // We load the 3 rows into SSE registers, and then combine them into a
  // single AVX register for write.
  const __m128d row0 = _mm_load_pd1(matrix + 0 * columns);
  const __m128d row1 = _mm_load_pd1(matrix + 1 * columns);
  const __m128d row2 = _mm_load_pd1(matrix + 2 * columns);

  _mm256_maskstore_pd(
      matrix_transpose + 0 * rows, mask,
      _mm256_insertf128_pd(
          _mm256_castpd128_pd256(_mm_shuffle_pd(row0, row1, 0b00)), row2, 1));
}

template <>
void transpose_block<2, 3>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);
  const __m256d row1 = _mm256_maskload_pd(matrix + 1 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd((row0), (row1), 0b1111);

  _mm_storeu_pd(matrix_transpose + 0 * rows, _mm256_castpd256_pd128(tmp0));
  _mm_storeu_pd(matrix_transpose + 1 * rows, _mm256_castpd256_pd128(tmp1));
  _mm_storeu_pd(matrix_transpose + 2 * rows, _mm256_extractf128_pd(tmp0, 1));
}

template <>
void transpose_block<1, 3>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd(row0, row0, 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd(row0, row0, 0b1111);

  const __m128i store_mask = _mm_set_epi64x(0, -1);
  _mm_maskstore_pd(matrix_transpose + 0 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp0));
  _mm_maskstore_pd(matrix_transpose + 1 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp1));
  _mm_maskstore_pd(matrix_transpose + 2 * rows, store_mask,
                   _mm256_extractf128_pd(tmp0, 1));
}
#endif

#if defined(__SSE2__)
template <>
void transpose_block<2, 2>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m128d row0 = _mm_loadu_pd(matrix);
  const __m128d row1 = _mm_loadu_pd(matrix + columns);

  const __m128d tmp0 = _mm_shuffle_pd(row0, row1, 0b00);
  const __m128d tmp1 = _mm_shuffle_pd(row0, row1, 0b11);

  _mm_storeu_pd(matrix_transpose, tmp0);
  _mm_storeu_pd(matrix_transpose + rows, tmp1);
}

template <>
void transpose_block<2, 1>(double* matrix_transpose, const double* const matrix,
                           const int32_t columns, const int32_t /*rows*/) {
  const __m128d row0 = _mm_load_pd1(matrix + 0 * columns);
  const __m128d row1 = _mm_load_pd1(matrix + 1 * columns);

  const __m128d tmp0 = _mm_shuffle_pd(row0, row1, 0b00);

  _mm_storeu_pd(matrix_transpose, tmp0);
}

template <>
void transpose_block<1, 2>(double* matrix_transpose, const double* const matrix,
                           const int32_t /*columns*/, const int32_t rows) {
#if defined(__AVX__)
  const __m128d row = _mm_loadu_pd(matrix);
  _mm_maskstore_pd(matrix_transpose + 0 * rows, _mm_set_epi64x(0, -1), row);
  _mm_maskstore_pd(matrix_transpose + 1 * rows - 1, _mm_set_epi64x(-1, 0), row);
#else
  matrix_transpose[0] = matrix[0];
  matrix_transpose[rows] = matrix[1];
#endif
}
#endif
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

template <>
void transpose_block<1, 1>(double* matrix_transpose, const double* const matrix,
                           const int32_t /*columns*/, const int32_t /*rows*/) {
  *matrix_transpose = *matrix;
}

template <int32_t BlockSize, int32_t RowExcess, int32_t ColumnExcess>
void transpose_impl(double* matrix_transpose,  //
                    const double* const matrix, const int32_t in_number_of_rows,
                    const int32_t in_number_of_columns) {
  const int32_t bound_on_rows = in_number_of_rows - RowExcess;
  const int32_t bound_on_columns = in_number_of_columns - ColumnExcess;

  for (int32_t row_index = 0UL; row_index < bound_on_rows;
       row_index += BlockSize) {
    for (int32_t column_index = 0UL; column_index < bound_on_columns;
         column_index += BlockSize) {
      if constexpr (BlockSize != 1) {
        transpose_block<BlockSize, BlockSize>(
            matrix_transpose + row_index + in_number_of_rows * column_index,
            matrix + column_index + in_number_of_columns * row_index,
            in_number_of_columns, in_number_of_rows);
      } else {
        static_assert(BlockSize == 1);
        static_assert(RowExcess == 0);
        static_assert(ColumnExcess == 0);
        transpose_block<1, 1>(
            matrix_transpose + row_index + in_number_of_rows * column_index,
            matrix + column_index + in_number_of_columns * row_index,
            in_number_of_columns, in_number_of_rows);
      }
    }
    // Handle remainder in row, that is, deal with extra columns.
    if constexpr (BlockSize > 1 and ColumnExcess != 0) {
      const int32_t column_index = bound_on_columns;
      transpose_block<BlockSize, ColumnExcess>(
          matrix_transpose + row_index + in_number_of_rows * column_index,
          matrix + column_index + in_number_of_columns * row_index,
          in_number_of_columns, in_number_of_rows);
    }
  }

  // Now deal with excess in either the columns or rows.
  //
  // We have the choice of either having the extra loops of the inner index
  // (currently row_index)  inside the main loop above or down below. This is a
  // tradeoff between data cache and instruction cache.
  if constexpr (BlockSize > 1 and RowExcess != 0) {
    const int32_t row_index = bound_on_rows;
    for (int32_t column_index = 0UL; column_index < bound_on_columns;
         column_index += BlockSize) {
      transpose_block<RowExcess, BlockSize>(
          matrix_transpose + row_index + in_number_of_rows * column_index,
          matrix + column_index + in_number_of_columns * row_index,
          in_number_of_columns, in_number_of_rows);
    }
    if constexpr (ColumnExcess != 0) {
      const int32_t column_index = bound_on_columns;
      transpose_block<RowExcess, ColumnExcess>(
          matrix_transpose + row_index + in_number_of_rows * column_index,
          matrix + column_index + in_number_of_columns * row_index,
          in_number_of_columns, in_number_of_rows);
    }
  }
}
}  // namespace

namespace detail {
void transpose_impl(double* matrix_transpose, const double* const matrix,
                    const int32_t number_of_rows,
                    const int32_t number_of_columns) {
  constexpr size_t block_size =
#if defined(__AVX__)
      4
#elif defined(__SSE2__)
      2
#else
      1
#endif
      ;
  const auto forward_to_impl = [&](auto row_excess_v) {
    constexpr size_t row_excess = decltype(row_excess_v)::value;
    switch (number_of_columns % static_cast<int32_t>(block_size)) {
#if defined(__AVX__)
      case 3:
        ::transpose_impl<block_size, row_excess, 3>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      case 2:
        ::transpose_impl<block_size, row_excess, 2>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
#endif
#if defined(__SSE2__) or defined(__AVX__)
      case 1:
        ::transpose_impl<block_size, row_excess, 1>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
#endif
      case 0:
        ::transpose_impl<block_size, row_excess, 0>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      default:
        ERROR("Can't determine the excess number of columns.");
    };
  };
  switch (number_of_rows % static_cast<int32_t>(block_size)) {
#if defined(__AVX__)
    case 3:
      forward_to_impl(std::integral_constant<uint32_t, 3>{});
      break;
    case 2:
      forward_to_impl(std::integral_constant<uint32_t, 2>{});
      break;
#endif
#if defined(__SSE2__) or defined(__AVX__)
    case 1:
      forward_to_impl(std::integral_constant<uint32_t, 1>{});
      break;
#endif
    case 0:
      forward_to_impl(std::integral_constant<uint32_t, 0>{});
      break;
    default:
      ERROR("Can't determine the excess number of rows.");
  };
}
}  // namespace detail
