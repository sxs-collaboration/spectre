// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace h5 {
/*!
 * \brief Combine a volume subfile across different HDF5 files.
 *
 * The argument `blocks_to_combine` can list block names and block groups that
 * should be combined. We ignore other blocks when combining the HDF5
 * files. This provides a way to filter volume data for easier visualization.
 */
void combine_h5(const std::vector<std::string>& file_names,
                const std::string& subfile_name, const std::string& output,
                std::optional<double> start_value = std::nullopt,
                std::optional<double> stop_value = std::nullopt,
                const std::optional<std::vector<std::string>>&
                    blocks_to_combine = std::nullopt,
                bool check_src = true);

}  // namespace h5
