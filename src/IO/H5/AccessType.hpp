// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines enum for specifying whether the H5 file is ReadWrite or ReadOnly

#pragma once

#include <iosfwd>

/*!
 * \ingroup HDF5Group
 * \brief Contains functions and classes for manipulating HDF5 files
 *
 * Wraps many underlying C H5 routines making them easier to use and easier to
 * manipulate H5 files.
 */
namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Set the access type to the H5File
 */
enum class AccessType {
  /// Allow read-write access to the file
  ReadWrite,
  /// Allow only read access to the file
  ReadOnly
};

std::ostream& operator<<(std::ostream& os, AccessType t) noexcept;
}  // namespace h5
