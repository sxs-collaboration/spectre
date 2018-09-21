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

/*!
 * \ingroup LoggingGroup
 * \brief Retrieve a string containing the current version of SpECTRE
 */
std::string spectre_version();

/*!
 * \ingroup LoggingGroup
 * \brief Returns major version
 */
int spectre_major_version();

/*!
 * \ingroup LoggingGroup
 * \brief Returns minor version
 */
int spectre_minor_version();

/*!
 * \ingroup LoggingGroup
 * \brief Returns patch version
 */
int spectre_patch_version();

/*!
 * \ingroup LoggingGroup
 * \brief Returns the path to the Unit test directory.
 */
std::string unit_test_path() noexcept;
