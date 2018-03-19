// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions that have warnings that aren't always ignored even with
/// a warning pop. This header must only be included in the Test_Blas.cpp
// file.

#pragma once

#ifdef __GNUC__
#pragma GCC system_header
#endif

#include <cstddef>

#include "Utilities/Blas.hpp"

namespace test_blas_asserts_for_bad_char {
inline void dgemm_error_transa_false() {
  size_t size = 0;
  double value = 10;
  dgemm_('this_bad', 'N', size, size, size, value, &value, size, &value, size,
         value, &value, size);
}

inline void dgemm_error_transb_false() {
  size_t size = 0;
  double value = 10;
  dgemm_('n', 'this_bad', size, size, size, value, &value, size, &value, size,
         value, &value, size);
}

inline void dgemm_error_transa_true() {
  size_t size = 0;
  double value = 10;
  dgemm_<true>('this_bad', 'N', size, size, size, value, &value, size, &value,
               size, value, &value, size);
}

inline void dgemm_error_transb_true() {
  size_t size = 0;
  double value = 10;
  dgemm_<true>('n', 'this_bad', size, size, size, value, &value, size, &value,
               size, value, &value, size);
}

inline void dgemv_error_trans() {
  size_t size = 0;
  double value = 10;
  dgemv_('this_bad', size, size, value, &value, size, &value, size, value,
         &value, size);
}
}  // namespace test_blas_asserts_for_bad_char
