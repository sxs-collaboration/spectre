// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <hdf5.h>

#include "Utilities/ForceInline.hpp"

// H5F wrappers
namespace h5 {
/// \ingroup HDF5Group
SPECTRE_ALWAYS_INLINE auto h5f_acc_rdonly() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5F_ACC_RDONLY;
#pragma GCC diagnostic pop
}

/// \ingroup HDF5Group
SPECTRE_ALWAYS_INLINE auto h5f_acc_rdwr() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5F_ACC_RDWR;
#pragma GCC diagnostic pop
}

SPECTRE_ALWAYS_INLINE auto h5f_acc_trunc() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5F_ACC_TRUNC;
#pragma GCC diagnostic pop
}
}  // namespace h5

// H5P wrappers
namespace h5 {
/// \ingroup HDF5Group
SPECTRE_ALWAYS_INLINE auto h5p_default() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5P_DEFAULT;
#pragma GCC diagnostic pop
}
}  // namespace h5

// H5S wrappers
namespace h5 {
/// \ingroup HDF5Group
SPECTRE_ALWAYS_INLINE auto h5s_all() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5S_ALL;
#pragma GCC diagnostic pop
}

/// \ingroup HDF5Group
SPECTRE_ALWAYS_INLINE auto h5s_unlimited() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5S_UNLIMITED;
#pragma GCC diagnostic pop
}

/// \ingroup HDF5Group
SPECTRE_ALWAYS_INLINE auto h5s_scalar() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return H5S_SCALAR;
#pragma GCC diagnostic pop
}
}  // namespace h5
