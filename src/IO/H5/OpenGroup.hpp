// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class OpenGroup for opening groups in HDF5

#pragma once

#include <hdf5.h>
#include <string>
#include <vector>

#include "IO/H5/AccessType.hpp"

namespace h5 {
namespace detail {
/*!
 * \ingroup HDF5Group
 * \brief Open an H5 group
 *
 * Opens a group recursively on creation and closes the groups when destructed.
 */
class OpenGroup {
 public:
  OpenGroup() = default;

  /*!
   * \param file_id the root/base file/group id where to start opening
   * \param group_name the full path to the group to open
   * \param access_type either AccessType::ReadOnly or AccessType::ReadWrite
   */
  OpenGroup(hid_t file_id, const std::string& group_name,
            h5::AccessType access_type);

  /// \cond HIDDEN_SYMBOLS
  ~OpenGroup();
  /// \endcond

  // @{
  /// \cond HIDDEN_SYMBOLS
  /// Copying does not make sense since the group will then be closed twice.
  OpenGroup(const OpenGroup& /*rhs*/) = delete;
  OpenGroup& operator=(const OpenGroup& /*rhs*/) = delete;
  /// \endcond
  // @}

  OpenGroup(OpenGroup&& rhs) noexcept;             // NOLINT
  OpenGroup& operator=(OpenGroup&& rhs) noexcept;  // NOLINT

  const hid_t& id() const noexcept { return group_id_; }

 private:
  /// \cond HIDDEN_SYMBOLS
  std::vector<hid_t> group_path_;
  hid_t group_id_{-1};
  /// \endcond
};
}  // namespace detail
}  // namespace h5
