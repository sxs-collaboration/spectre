// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/File.hpp"

#include <algorithm>
#include <hdf5.h>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Header.hpp"  // IWYU pragma: keep
#include "IO/H5/Object.hpp"
#include "IO/H5/SourceArchive.hpp"  // IWYU pragma: keep
#include "IO/H5/Wrappers.hpp"
#include "Utilities/FileSystem.hpp"

namespace h5 {
template <AccessType Access_t>
H5File<Access_t>::H5File(std::string file_name, bool append_to_file)
    : file_name_(std::move(file_name)) {
  if (file_name_.size() - 3 != file_name_.find(".h5")) {
    ERROR("All HDF5 file names must end in '.h5'. The path and file name '"
          << file_name_ << "' does not satisfy this");
  }
  const bool file_exists = file_system::check_if_file_exists(file_name_);
  if (not file_exists and AccessType::ReadOnly == Access_t) {
    ERROR("Cannot create a file in ReadOnly mode, '" << file_name_ << "'");
  }
  if (append_to_file and AccessType::ReadOnly == Access_t) {
    ERROR("Cannot append to a file opened in read-only mode. File name is: "
          << file_name_);
  }
  if (file_exists and not append_to_file and
      h5::AccessType::ReadWrite == Access_t) {
    ERROR("File '" << file_name_
                   << "' already exists and we are not allowed to append. To "
                      "reduce the risk of accidental deletion you must "
                      "explicitly delete the file first using the file_system "
                      "library in SpECTRE or through your shell.");
  }
  file_id_ = file_exists
                 ? H5Fopen(file_name_.c_str(),
                           AccessType::ReadOnly == Access_t ? h5f_acc_rdonly()
                                                            : h5f_acc_rdwr(),
                           h5p_default())
                 : H5Fcreate(file_name_.c_str(), h5f_acc_trunc(), h5p_default(),
                             h5p_default());
  CHECK_H5(file_id_, "Failed to open file '"
                         << file_name_ << "'. The file exists status is: "
                         << std::boolalpha << file_exists
                         << ". Writing from directory: " << file_system::cwd()
                         << ". Trying to open in mode: " << Access_t);
  if (not file_exists) {
    insert_header();
    insert_source_archive();
  }
}

/// \cond
template <AccessType Access_t>
H5File<Access_t>::H5File(H5File&& rhs) noexcept {
  file_name_ = std::move(rhs.file_name_);
  file_id_ = std::move(rhs.file_id_);
  current_object_ = std::move(rhs.current_object_);
  h5_groups_ = std::move(rhs.h5_groups_);
  rhs.file_id_ = -1;
}

template <AccessType Access_t>
H5File<Access_t>& H5File<Access_t>::operator=(H5File&& rhs) noexcept {
  if (file_id_ != -1) {
    CHECK_H5(H5Fclose(file_id_),
             "Failed to close file: '" << file_name_ << "'");
  }

  file_name_ = std::move(rhs.file_name_);
  file_id_ = std::move(rhs.file_id_);
  current_object_ = std::move(rhs.current_object_);
  h5_groups_ = std::move(rhs.h5_groups_);
  rhs.file_id_ = -1;
  return *this;
}

template <AccessType Access_t>
H5File<Access_t>::~H5File() {
  if (file_id_ != -1) {
    CHECK_H5(H5Fclose(file_id_),
             "Failed to close file: '" << file_name_ << "'");
  }
}

template <>
void H5File<AccessType::ReadWrite>::insert_source_archive() noexcept {
  insert<h5::SourceArchive>("/src");
}
template <>
void H5File<AccessType::ReadOnly>::insert_source_archive() noexcept {}
/// \endcond
}  // namespace h5

template class h5::H5File<h5::AccessType::ReadOnly>;
template class h5::H5File<h5::AccessType::ReadWrite>;
