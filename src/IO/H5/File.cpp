// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/File.hpp"

#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Helpers.hpp"
#include "Utilities/Literals.hpp"

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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  file_id_ = file_exists
                 ? H5Fopen(file_name_.c_str(),
                           AccessType::ReadOnly == Access_t ? H5F_ACC_RDONLY
                                                            : H5F_ACC_RDWR,
                           H5P_DEFAULT)
                 : H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(file_id_, "Failed to open file '" << file_name_ << "'");
  if (not file_exists) {
    insert_header();
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
/// \endcond
}  // namespace h5

template class h5::H5File<h5::AccessType::ReadOnly>;
template class h5::H5File<h5::AccessType::ReadWrite>;
