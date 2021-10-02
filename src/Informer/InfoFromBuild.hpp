// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function info_from_build.

#pragma once

#include <string>

/*!
 * \ingroup LoggingGroup
 * \brief Information about the version, date, host, git commit, and link time
 *
 * The information returned by this function is invaluable for identifying
 * the version of the code used in a simulation, as well as which host, the
 * date the code was compiled, and the time of linkage.
 */
std::string info_from_build();

// We declare these functions `extern "C"` so their symbols are not mangled and
// we can ignore that they are undefined until link time (see `CMakeLists.txt`).
// Note that the `std::string` return type is not compatible with C, but that's
// OK as long as we're not calling or defining the functions in C. See
// Microsoft's C4190:
// https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4190
#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif  // defined(__clang__)
/*!
 * \ingroup LoggingGroup
 * \brief The time and date the executable was linked
 */
extern "C" std::string link_date();
/*!
 * \ingroup LoggingGroup
 * \brief The git description at the time the executable was linked
 */
extern "C" std::string git_description();
/*!
 * \ingroup LoggingGroup
 * \brief The git branch at the time the executable was linked
 */
extern "C" std::string git_branch();
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__clang__)

/*!
 * \ingroup LoggingGroup
 * \brief Retrieve a string containing the current version of SpECTRE
 */
std::string spectre_version();

/*!
 * \ingroup LoggingGroup
 * \brief Returns the path to the Unit test directory.
 */
std::string unit_test_src_path();

/*!
 * \ingroup LoggingGroup
 * \brief Returns the path to the Unit test directory in the build directory.
 */
std::string unit_test_build_path();
