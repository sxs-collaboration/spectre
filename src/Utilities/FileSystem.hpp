// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares functions to do file system manipulations

#pragma once

#include <cstddef>
#include <string>
#include <vector>

/*!
 * \ingroup FileSystemGroup
 * \brief A light-weight file system library based on POSIX.
 *
 * We use this library instead of a subprocess based library because OpenMPI
 * does not support forking of processes on all systems. Since the
 * parallelization library we use may be implemented on top of OpenMPI we
 * take the safe route and use POSIX.
 */
namespace file_system {

/*!
 * \ingroup FileSystemGroup
 * \brief Wraps the dirname function to get the pathname of the parent directory
 *
 * See the opengroup documentation:
 * http://pubs.opengroup.org/onlinepubs/9699919799/functions/dirname.html
 *
 * \example
 * \snippet Test_FileSystem.cpp get_parent_path
 *
 * \requires `path` is a valid path on the filesystem
 * \returns the path to the parent directory
 */
std::string get_parent_path(const std::string& path);

/*!
 * \ingroup FileSystemGroup
 * \brief Given a path to a file returns the file name
 *
 * \example
 * \snippet Test_FileSystem.cpp get_file_name
 *
 * \requires `file_path` is a valid path on the filesystem
 * \returns the file name
 */
std::string get_file_name(const std::string& file_path);

/*!
 * \ingroup FileSystemGroup
 * \brief Get the absolute path, resolving symlinks
 *
 * \requires `rel_path` is a valid path on the filesystem
 * \returns the absolute path
 */
std::string get_absolute_path(const std::string& rel_path);

/*!
 * \ingroup FileSystemGroup
 * \brief Creates a directory, including any parents that don't exist. If the
 * directory exists `create_directory` does nothing.
 *
 * \requires permissions to create `dir` on the filesystem
 * \effects creates the directory `dir` on the filesystem
 *
 * \param dir the path where to create the directory
 * \param wait_time time to wait in seconds between failures
 * \param num_tries number of attempts to create directory (for slow
 * filesystems)
 */
void create_directory(const std::string& dir, double wait_time = 1,
                      size_t num_tries = 40);

/*!
 * \ingroup FileSystemGroup
 * \brief Returns true if the directory exists
 *
 * \returns `true` if the directory exists
 */
bool check_if_dir_exists(const std::string& dir);

/*!
 * \ingroup FileSystemGroup
 * \brief Returns true if the regular file or link to the regular file exists.
 *
 * \note See the stat(2) documentation, e.g. at
 * http://man7.org/linux/man-pages/man2/stat.2.html for details.
 *
 * \returns `true` if the file exists
 */
bool check_if_file_exists(const std::string& file);

/*!
 * \ingroup FileSystemGroup
 * \brief Returns true if the path points to a regular file or a link to a
 * regular file.
 *
 * \note See the stat(2) documentation, e.g. at
 * http://man7.org/linux/man-pages/man2/stat.2.html for details.
 *
 * \requires `path` is a valid path on the filesystem
 * \returns `true` if `file` is a file, not a directory
 */
bool is_file(const std::string& path);

/*!
 * \ingroup FileSystemGroup
 * \brief Returns the file size in bytes
 *
 * \requires `file` is a valid file on the filesystem
 * \returns size of `file` in bytes
 */
size_t file_size(const std::string& file);

/*!
 * \ingroup FileSystemGroup
 * \brief Returns the current working directory, resolving symlinks
 */
std::string cwd();

/*!
 * \ingroup FileSystemGroup
 * \brief Gets a list of files in a directory
 *
 * \returns vector of all files and directories inside `dir_name`
 */
std::vector<std::string> ls(const std::string& dir_name = "./");

/*!
 * \ingroup FileSystemGroup
 * \brief Deletes a file or directory.
 *
 * \requires `path` be a valid path on the filesystem
 * \effects deletes `path` from the filesystem, if `recursive` is `true` then
 * behaves like `rm -r`, otherwise like `rm` but will delete an empty directory
 */
void rm(const std::string& path, bool recursive);
}  // namespace file_system
