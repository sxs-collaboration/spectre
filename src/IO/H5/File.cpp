// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/File.hpp"

#include <algorithm>
#include <charm++.h>
#include <exception>
#include <hdf5.h>
#include <string>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Header.hpp"  // IWYU pragma: keep
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Object.hpp"
#include "IO/H5/SourceArchive.hpp"  // IWYU pragma: keep
#include "IO/H5/Wrappers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/System/Abort.hpp"

namespace h5 {
template <AccessType Access_t>
H5File<Access_t>::H5File(std::string file_name, bool append_to_file,
                         const std::string& input_source, bool use_file_locking)
    : file_name_(std::move(file_name)) {
  if (file_name_.size() - 3 != file_name_.find(".h5")) {
    ERROR("All HDF5 file names must end in '.h5'. The path and file name '"
          << file_name_ << "' does not satisfy this");
  }
  const bool file_exists = file_system::check_if_file_exists(file_name_);
  if (not file_exists and AccessType::ReadOnly == Access_t) {
    ERROR("Trying to open the file '"
          << file_name_
          << "' in ReadOnly mode but the file does not exist. If you want to "
             "create the file you must switch to ReadWrite mode.");
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

  const hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  CHECK_H5(fapl_id, "Failed to create file access property list.");
#ifdef HDF5_SUPPORTS_SET_FILE_LOCKING
  CHECK_H5(H5Pset_file_locking(
               fapl_id, use_file_locking,
               // Ignore file locks when they are disabled on the file system
               true),
           "Failed to configure file locking.");
#else
  (void)use_file_locking;
#endif

  file_id_ = file_exists
                 ? H5Fopen(file_name_.c_str(),
                           AccessType::ReadOnly == Access_t ? h5f_acc_rdonly()
                                                            : h5f_acc_rdwr(),
                           fapl_id)
                 : H5Fcreate(file_name_.c_str(), h5f_acc_trunc(), h5p_default(),
                             fapl_id);
  CHECK_H5(file_id_, "Failed to open file '"
                         << file_name_ << "'. The file exists status is: "
                         << std::boolalpha << file_exists
                         << ". Writing from directory: " << file_system::cwd()
                         << ". Trying to open in mode: " << Access_t);
  if (not file_exists) {
    close_current_object();
    insert_header();
    close_current_object();
    insert_source_archive();
    write_to_attribute<std::string>(file_id_, "InputSource.yaml"s,
                                    {{input_source}});
  }
  close_current_object();
}

template <AccessType Access_t>
H5File<Access_t>::H5File(H5File&& rhs) {
  file_name_ = std::move(rhs.file_name_);
  file_id_ = std::move(rhs.file_id_);
  current_object_ = std::move(rhs.current_object_);
  h5_groups_ = std::move(rhs.h5_groups_);
  rhs.file_id_ = -1;
}

template <AccessType Access_t>
H5File<Access_t>& H5File<Access_t>::operator=(H5File&& rhs) {
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
    const auto h5_status = H5Fclose(file_id_);
    if (h5_status < 0) {
      // Could not close the H5 file.  Since we cannot throw an exception
      // from the destructor (without modifying the destructor of the Charm++
      // class PUP::able), we just abort.  But before aborting we attempt to
      // see if another exception is propagating, and if so print its error
      // message.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      CkError("Failed to close H5 File '%s'\n", file_name_.c_str());
      std::exception_ptr exception = std::current_exception();
      if (exception) {
        try {
          std::rethrow_exception(exception);
        } catch (std::exception& ex) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
          CkError("while handling the following exception:\n\n%s\n", ex.what());
        }
      }
      sys::abort("");
    }
  }
}

template <AccessType Access_t>
void H5File<Access_t>::close() const {
  // Need to close current object because `H5Fclose` keeps the file open
  // internally until all objects are closed
  close_current_object();
  if (file_id_ != -1) {
    CHECK_H5(H5Fclose(file_id_), "Failed to close the file.");
    file_id_ = -1;
  }
}

template <AccessType Access_t>
std::string H5File<Access_t>::input_source() const {
  return h5::read_value_attribute<std::string>(file_id_, "InputSource.yaml"s);
}

template <>
void H5File<AccessType::ReadWrite>::insert_source_archive() {
  insert<h5::SourceArchive>("/src");
}
template <>
void H5File<AccessType::ReadOnly>::insert_source_archive() {}
}  // namespace h5

template class h5::H5File<h5::AccessType::ReadOnly>;
template class h5::H5File<h5::AccessType::ReadWrite>;
