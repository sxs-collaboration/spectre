// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class h5::Version for storing version history of files

#pragma once

#include <cstdint>
#include <hdf5.h>
#include <string>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Used to store the version of the file
 *
 * A Version object should be stored inside each H5 object that is to represent
 * a file, e.g. Dat, or Text.
 *
 * \example
 * To write the version use:
 * \snippet Test_H5.cpp h5file_write_version
 * To read the version use:
 * \snippet Test_H5.cpp h5file_read_version
 *
 */
class Version : public h5::Object {
 public:
  /// \cond HIDDEN_SYMOLS
  static std::string extension() { return ".ver"; }

  Version(bool exists, detail::OpenGroup&& group, hid_t location,
          std::string name, uint32_t version = 1);

  Version(const Version& /*rhs*/) = delete;
  Version& operator=(const Version& /*rhs*/) = delete;

  Version(Version&& /*rhs*/) noexcept = default;             // NOLINT
  Version& operator=(Version&& /*rhs*/) noexcept = default;  // NOLINT
  ~Version() override = default;
  /// \endcond

  uint32_t get_version() const noexcept { return version_; }

 private:
  /// \cond HIDDEN_SYMBOLS
  uint32_t version_;
  // group_ is necessary since the when the h5::Object is destroyed it closes
  // all groups that were opened to get to it.
  detail::OpenGroup group_;
  /// \endcond
};
}  // namespace h5
