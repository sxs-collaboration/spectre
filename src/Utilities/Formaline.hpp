// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

/*!
 * \ingroup UtilitiesGroup
 * \brief Functions for retrieving system and source tree information
 */
namespace formaline {
/*!
 * \brief Returns a byte stream of the source tree at the time the executable
 * was compiled.
 */
std::vector<char> get_archive() noexcept;

/*!
 * \brief Returns the environment variables at link time.
 */
std::string get_environment_variables() noexcept;

/*!
 * \brief Returns the contents of SpECTRE's LibraryVersions.txt file.
 */
std::string get_library_versions() noexcept;

/*!
 * \brief Returns the PATH, CPATH, LD_LIBRARY_PATH, LIBRARY_PATH, and
 * CMAKE_PREFIX_PATH at time of compilation.
 */
std::string get_paths() noexcept;

/*!
 * \brief Write the source tree archive to the file
 * `filename_without_extension.tar.gz`
 */
void write_to_file(const std::string& filename_without_extension) noexcept;
}  // namespace formaline
