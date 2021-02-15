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
 *
 * We declare the function `extern "C"` so its symbol is not mangled and we can
 * ignore that it is undefined until link time (see `CMakeLists.txt`). Note that
 * the `std::string` return type is not compatible with C, but that's OK as long
 * as we're not calling or defining the function in C. See Microsoft's C4190:
 * https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4190
 */
#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif  // defined(__clang__)
extern "C" std::string info_from_build();
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
std::string unit_test_src_path() noexcept;

/*!
 * \ingroup LoggingGroup
 * \brief Returns the path to the Unit test directory in the build directory.
 */
std::string unit_test_build_path() noexcept;
