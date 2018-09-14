// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function for retrieving the HDF5 type for a template parameter

#pragma once

#include <hdf5.h>
#include <string>
#include <type_traits>

#include "IO/H5/CheckH5.hpp"
#include "Utilities/ForceInline.hpp"

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Returns the HDF5 datatype for the corresponding type `T`
 *
 * \requires `T` is a valid type with a corresponding HDF5 datatype, i.e.
 * a fundamental or a std::string
 * \returns the HDF5 datatype for the corresponding type `T`
 *
 * \note For strings, the returned type must be released with H5Tclose
 */
template <typename T>
SPECTRE_ALWAYS_INLINE hid_t h5_type();
/// \cond HIDDEN_SYMBOLS
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<double>() {
  return H5T_NATIVE_DOUBLE;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<int>() {
  return H5T_NATIVE_INT;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<unsigned int>() {
  return H5T_NATIVE_UINT;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<long>() {
  return H5T_NATIVE_LONG;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<unsigned long>() {
  return H5T_NATIVE_ULONG;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<long long>() {
  return H5T_NATIVE_LLONG;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<unsigned long long>() {
  return H5T_NATIVE_ULLONG;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<char>() {
  return H5T_NATIVE_CHAR;  // LCOV_EXCL_LINE
}
template <>
SPECTRE_ALWAYS_INLINE hid_t h5_type<std::string>() {
  hid_t datatype = H5Tcopy(H5T_C_S1);
  CHECK_H5(datatype, "Failed to allocate string.");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CHECK_H5(H5Tset_size(datatype, H5T_VARIABLE),
           "Failed to set size of string.");
#pragma GCC diagnostic pop
  return datatype;
}
/// \endcond

/*!
 * \ingroup HDF5Group
 * \brief Create an H5 FORTRAN string.
 */
SPECTRE_ALWAYS_INLINE hid_t fortran_string() noexcept {
  hid_t datatype = H5Tcopy(H5T_FORTRAN_S1);
  CHECK_H5(datatype, "Failed to allocate string.");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CHECK_H5(H5Tset_size(datatype, H5T_VARIABLE),
           "Failed to set size of string.");
#pragma GCC diagnostic pop
  return datatype;
}
}  // namespace h5
