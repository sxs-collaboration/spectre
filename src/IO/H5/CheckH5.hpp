// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro CHECK_H5

#pragma once

#include "ErrorHandling/Error.hpp"

/*!
 * \ingroup HDF5Group
 * \brief Check if an HDF5 operation was successful
 *
 * \requires `h5_status` is a valid HDF5 return type, `m` is a stringstream
 * syntax error message.
 *
 * \effects if `h5_status` is set to error then aborts execution else none
 */
#define CHECK_H5(h5_status, m)                        \
  if (h5_status < 0) {                     /*NOLINT*/ \
    ERROR("Failed HDF5 operation: " << m); /*NOLINT*/ \
  } else                                              \
    static_cast<void>(0)
