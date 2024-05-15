// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>

namespace Cce {
/// @{
/*!
 * \brief Retrieves the extraction radius from the specified file name.
 *
 * \details We assume that the filename has the extraction radius encoded as an
 * integer between the last occurrence of 'R' and the last occurrence of '.'.
 * This is the format provided by SpEC.
 */
std::string get_text_radius(const std::string& cce_data_filename);

std::optional<double> get_extraction_radius(
    const std::string& cce_data_filename,
    const std::optional<double>& extraction_radius, bool error = true);
/// @}
}  // namespace Cce
