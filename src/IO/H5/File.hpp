// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class h5::H5File

#pragma once

#include <algorithm>
#include <exception>
#include <hdf5.h>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Header.hpp"  // IWYU pragma: keep
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/PrettyType.hpp"

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Opens an HDF5 file for access and allows manipulation of data
 *
 * Opens an HDF5 file either in ReadOnly or ReadWrite mode depending on the
 * template parameter `Access_t`. In ReadWrite mode h5::Object's can be inserted
 * into the file, and objects can be retrieved to have their data manipulated.
 * Example objects are dat files, text files, and volume data files. A single
 * H5File can contain many different objects so that the number of files stored
 * during a simulation is reduced.
 *
 * When an h5::object inside an H5File is opened or created the H5File object
 * holds a copy of the h5::object.
 *
 * \warning Only one object can be open at a time, which means if a reference to
 * the object is kept around after the H5File's current object is closed there
 * is a dangling reference. Also, this means that after you insert an object,
 * you must close that object before you can insert/open another.
 *
 * \example
 * To open a file for read-write access:
 * \snippet Test_H5File.cpp h5file_readwrite_file
 *
 * \note The dangling reference issue could be fixed by having a function in
 * addition to `get` that takes a lambda. The lambda takes exactly one parameter
 * of the type of the h5::Object it will be operating on. While this approach is
 * likely to be syntactically strange for many users it will most likely be more
 * performant than the `shared_ptr` solution.
 *
 * @tparam Access_t either h5::AccessType::ReadWrite or h5::AccessType::ReadOnly
 */
template <AccessType Access_t>
class H5File {
 public:
  // empty constructor for classes which store an H5File and need to be
  // charm-compatible.
  H5File() = default;

  /*!
   * \requires `file_name` is a valid path and ends in `.h5`.
   * \effects On object creation opens the HDF5 file at `file_name`
   *
   * @param file_name the path to the file to open or create
   * @param append_to_file if true allow appending to the file, otherwise abort
   * the simulation if the file exists
   * @param input_source a string containing the input source options (yaml
   * formatted). Defaults to an empty string; when writing, specify the provided
   * yaml input options (if any) to write them to the output file's
   * `InputSource.yaml` attribute.
   * @param use_file_locking Toggle file locking (default false).
   * HDF5 file locking is explained here:
   * https://github.com/HDFGroup/hdf5/blob/develop/doc/file-locking.md.
   * This toggle only has an effect if the HDF5 library supports
   * 'H5Pset_file_locking'. Otherwise, file locking is enabled if the HDF5
   * library was built with it, which it probably was. If file locking is
   * enabled, simulations may crash when the file they try to access is being
   * read by another process (like an analysis tool). We could make this more
   * resilient in the future by waiting to acquire the file lock with a timeout,
   * and/or retrying IO operations after progressively longer wait times (e.g.
   * first try again right away, then also print to terminal after some retries,
   * then eventually abort to avoid wasting compute time on a run that can't do
   * IO).
   */
  explicit H5File(std::string file_name, bool append_to_file = false,
                  const std::string& input_source = ""s,
                  bool use_file_locking = false);

  /// \cond HIDDEN_SYMBOLS
  ~H5File();
  /// \endcond

  /// @{
  /*!
   * \brief It does not make sense to copy an object referring to a file, only
   * to move it.
   */
  H5File(const H5File& /*rhs*/) = delete;
  H5File& operator=(const H5File& /*rhs*/) = delete;
  /// @}

  /// \cond HIDDEN_SYMBOLS
  H5File(H5File&& rhs);             // NOLINT
  H5File& operator=(H5File&& rhs);  // NOLINT
  /// \endcond

  /// Get name of the H5 file
  const std::string& name() const { return file_name_; }

  /// Get a std::vector of the names of all immediate subgroups of the file
  const std::vector<std::string> groups() const {
    return h5::get_group_names(file_id_, "/");
  }

  /// \brief Return a vector of all filenames in the H5 file
  /// \tparam ObjectType Only return a vector that contains this type of file.
  /// Default is `void` which returns all files.
  template <typename ObjectType = void>
  const std::vector<std::string> all_files(const std::string& group_name) const;

  /// Get the InputSource.yaml string embedded in the file
  std::string input_source() const;

  /// @{
  /*!
   * \requires `ObjectType` is a valid h5::Object derived class, `path`
   * is a valid path in the HDF5 file
   * \return a reference to the object inside the HDF5 file.
   *
   * @tparam ObjectType the type of the h5::Object to be retrieved, e.g. Dat
   * @param path the path of the retrieved object
   * @param args arguments forwarded to the ObjectType constructor
   */
  template <
      typename ObjectType, typename... Args,
      typename std::enable_if_t<((void)sizeof(ObjectType),
                                 Access_t == AccessType::ReadWrite)>* = nullptr>
  ObjectType& get(const std::string& path, Args&&... args);

  template <typename ObjectType, typename... Args>
  const ObjectType& get(const std::string& path, Args&&... args) const;
  /// @}

  /*!
   * \brief Insert an object into an H5 file.
   *
   * \requires `ObjectType` is a valid h5::Object derived class, `path` is a
   * valid path in the HDF5 file, and `args` are valid arguments to be forwarded
   * to the constructor of `ObjectType`.
   * \effects Creates a new H5 object of type `ObjectType` at the location
   * `path` in the HDF5 file.
   *
   * \return a reference the created object.
   *
   * @tparam ObjectType the type of the h5::Object to be inserted, e.g. Dat
   * @param path the path of the inserted object
   * @param args additional arguments to be passed to the constructor of the
   * object
   */
  template <typename ObjectType, typename... Args>
  ObjectType& insert(const std::string& path, Args&&... args);

  /*!
   * \brief Inserts an object like `insert` if it does not exist, returns the
   * object if it does.
   */
  template <typename ObjectType, typename... Args>
  ObjectType& try_insert(const std::string& path, Args&&... args);

  /*!
   * \effects Closes the current object, if there is none then has no effect
   */
  void close_current_object() const { current_object_ = nullptr; }

  /*!
   * \effects Closes the H5 file. No H5 operations are permitted after this
   * operation.
   */
  void close() const;

  template <typename ObjectType>
  bool exists(const std::string& path) const {
    auto exists_group_name = check_if_object_exists<ObjectType>(path);
    return std::get<0>(exists_group_name);
  }

 private:
  /// \cond HIDDEN_SYMBOLS
  template <typename ObjectType,
            std::enable_if_t<((void)sizeof(ObjectType),
                              Access_t == AccessType::ReadWrite)>* = nullptr>
  ObjectType& convert_to_derived(
      std::unique_ptr<h5::Object>& current_object);  // NOLINT
  template <typename ObjectType>
  const ObjectType& convert_to_derived(
      const std::unique_ptr<h5::Object>& current_object) const;

  void insert_header();
  void insert_source_archive();

  template <typename ObjectType>
  std::tuple<bool, detail::OpenGroup, std::string> check_if_object_exists(
      const std::string& path) const;

  std::string file_name_;
  // NOLINTNEXTLINE(spectre-mutable)
  mutable hid_t file_id_{-1};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::unique_ptr<h5::Object> current_object_{nullptr};
  std::vector<std::string> h5_groups_;
  /// \endcond HIDDEN_SYMBOLS
};

// ======================================================================
// H5File Definitions
// ======================================================================

template <AccessType Access_t>
template <typename ObjectType>
const std::vector<std::string> H5File<Access_t>::all_files(
    const std::string& group_name) const {
  std::vector<std::string> groups = h5::get_group_names(file_id_, group_name);

  // Loop through the initial files and groups and get all subfiles and groups
  std::vector<std::string> all_files_and_groups{};
  for (auto it = groups.begin(); it != groups.end(); ++it) {
    // Full group name
    const std::string prefix =
        group_name == "/" ? group_name : (group_name + "/");

    // If this is a file, there aren't any subfiles so add it to the overall
    // list and continue. Most extensions follow the ".XYZ" rule. Headers are
    // special though because they end in ".tar.gz"
    const auto extension_pos = it->find_last_of(".");
    if (not(extension_pos == std::string::npos) and
        (it->size() - extension_pos == 4 or
         it->substr(extension_pos) == ".gz")) {
      all_files_and_groups.insert(all_files_and_groups.end(), prefix + *it);
      continue;
    }

    // Get all sub files
    auto extra_files_and_groups = all_files(prefix + *it);

    // Insert the files to the overall list
    all_files_and_groups.insert(all_files_and_groups.end(),
                                extra_files_and_groups.begin(),
                                extra_files_and_groups.end());
  }

  // Filter out the ones we don't want
  if constexpr (not std::is_same_v<ObjectType, void>) {
    const auto range_end = std::remove_if(
        all_files_and_groups.begin(), all_files_and_groups.end(),
        [](const std::string& t) {
          return t.find(ObjectType::extension()) == std::string::npos;
        });
    // Shrink down the vector
    all_files_and_groups.erase(range_end, all_files_and_groups.end());
    all_files_and_groups.shrink_to_fit();
  }

  return all_files_and_groups;
}

template <AccessType Access_t>
template <typename ObjectType, typename... Args,
          typename std::enable_if_t<((void)sizeof(ObjectType),
                                     Access_t == AccessType::ReadWrite)>*>
ObjectType& H5File<Access_t>::get(const std::string& path, Args&&... args) {
  // Ensure we call the const version of the get function to avoid infinite
  // recursion. The reason this is implemented in this manner is to avoid code
  // duplication.
  // clang-tidy: do not use const_cast
  return const_cast<ObjectType&>(  // NOLINT
      static_cast<H5File<Access_t> const*>(this)->get<ObjectType>(
          path, std::forward<Args>(args)...));
}

template <AccessType Access_t>
template <typename ObjectType, typename... Args>
const ObjectType& H5File<Access_t>::get(const std::string& path,
                                        Args&&... args) const {
  if (current_object_ != nullptr) {
    ERROR("Object " << current_object_->subfile_path()
                    << " already open. Cannot open object " << path << ".");
  }
  // C++17: structured bindings
  auto exists_group_name = check_if_object_exists<ObjectType>(path);
  hid_t group_id = std::get<1>(exists_group_name).id();
  if (not std::get<0>(exists_group_name)) {
    ERROR("Cannot open the object '" << path + ObjectType::extension()
                                     << "' because it does not exist.");
  }
  current_object_ = std::make_unique<ObjectType>(
      std::get<0>(exists_group_name), std::move(std::get<1>(exists_group_name)),
      group_id, std::move(std::get<2>(exists_group_name)),
      std::forward<Args>(args)...);
  return dynamic_cast<const ObjectType&>(*current_object_);
}

template <AccessType Access_t>
template <typename ObjectType, typename... Args>
ObjectType& H5File<Access_t>::insert(const std::string& path, Args&&... args) {
  static_assert(AccessType::ReadWrite == Access_t,
                "Can only insert into ReadWrite access H5 files.");
  if (current_object_ != nullptr) {
    ERROR("Object " << current_object_->subfile_path()
                    << " already open. Cannot insert object " << path << ".");
  }
  // C++17: structured bindings
  auto exists_group_name = check_if_object_exists<ObjectType>(path);
  if (std::get<0>(exists_group_name)) {
    ERROR(
        "Cannot insert an Object that already exists. Failed to add Object "
        "named: "
        << path);
  }

  hid_t group_id = std::get<1>(exists_group_name).id();
  return convert_to_derived<ObjectType>(
      current_object_ = std::make_unique<ObjectType>(
          std::get<0>(exists_group_name),
          std::move(std::get<1>(exists_group_name)), group_id,
          std::move(std::get<2>(exists_group_name)),
          std::forward<Args>(args)...));
}

template <AccessType Access_t>
template <typename ObjectType, typename... Args>
ObjectType& H5File<Access_t>::try_insert(const std::string& path,
                                         Args&&... args) {
  static_assert(AccessType::ReadWrite == Access_t,
                "Can only insert into ReadWrite access H5 files.");
  if (current_object_ != nullptr) {
    ERROR("Object " << current_object_->subfile_path()
                    << " already open. Cannot try to insert object " << path
                    << ".");
  }
  // C++17: structured bindings
  auto exists_group_name = check_if_object_exists<ObjectType>(path);
  hid_t group_id = std::get<1>(exists_group_name).id();
  return convert_to_derived<ObjectType>(
      current_object_ = std::make_unique<ObjectType>(
          std::get<0>(exists_group_name),
          std::move(std::get<1>(exists_group_name)), group_id,
          std::move(std::get<2>(exists_group_name)),
          std::forward<Args>(args)...));
}

/// \cond HIDDEN_SYMBOLS
template <AccessType Access_t>
template <typename ObjectType,
          typename std::enable_if_t<((void)sizeof(ObjectType),
                                     Access_t == AccessType::ReadWrite)>*>
ObjectType& H5File<Access_t>::convert_to_derived(
    std::unique_ptr<h5::Object>& current_object) {
  if (nullptr == current_object) {
    ERROR("No object to convert.");  // LCOV_EXCL_LINE
  }
  try {
    return dynamic_cast<ObjectType&>(*current_object);
    // LCOV_EXCL_START
  } catch (const std::bad_cast& e) {
    ERROR("Failed to cast to object.\nCast error: " << e.what());
    // LCOV_EXCL_STOP
  }
}
template <AccessType Access_t>
template <typename ObjectType>
const ObjectType& H5File<Access_t>::convert_to_derived(
    const std::unique_ptr<h5::Object>& current_object) const {
  if (nullptr == current_object) {
    ERROR("No object to convert.");
  }
  try {
    return dynamic_cast<const ObjectType&>(*current_object);
  } catch (const std::bad_cast& e) {
    ERROR("Failed to cast to object.\nCast error: " << e.what());
  }
}

template <AccessType Access_t>
template <typename ObjectType>
std::tuple<bool, detail::OpenGroup, std::string>
H5File<Access_t>::check_if_object_exists(const std::string& path) const {
  std::string name_only = "/";
  if (path != "/") {
    name_only = file_system::get_file_name(path);
  }
  const std::string name_with_extension = name_only + ObjectType::extension();
  detail::OpenGroup group(file_id_, file_system::get_parent_path(path),
                          Access_t);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const bool object_exists =
      name_with_extension == "/" or
      H5Lexists(group.id(), name_with_extension.c_str(), H5P_DEFAULT) or
      H5Aexists(group.id(), name_with_extension.c_str());
#pragma GCC diagnostic pop
  return std::make_tuple(object_exists, std::move(group), std::move(name_only));
}

template <>
inline void H5File<AccessType::ReadWrite>::insert_header() {
  insert<h5::Header>("/header");
}
// Not tested because it is only required to get code to compile, if statement
// in constructor prevents call.
template <>
inline void H5File<AccessType::ReadOnly>::insert_header() {}  // LCOV_EXCL_LINE

/// \endcond
}  // namespace h5
