// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <hdf5.h>
#include <string>
#include <vector>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Writes an archive of the source tree into a dataset.
 */
class SourceArchive : public h5::Object {
 public:
  /// \cond HIDDEN_SYMOLS
  static std::string extension() noexcept { return ".tar.gz"; }

  SourceArchive(bool exists, detail::OpenGroup&& group, hid_t location,
                const std::string& name) noexcept;

  SourceArchive(const SourceArchive&) = delete;
  SourceArchive& operator=(const SourceArchive&) = delete;

  SourceArchive(SourceArchive&&) noexcept = delete;             // NOLINT
  SourceArchive& operator=(SourceArchive&&) noexcept = delete;  // NOLINT

  ~SourceArchive() override = default;
  /// \endcond

  const std::vector<char>& get_archive() const noexcept {
    return source_archive_;
  }

 private:
  /// \cond HIDDEN_SYMBOLS
  detail::OpenGroup group_;
  std::vector<char> source_archive_;
  /// \endcond
};
}  // namespace h5
