// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Check if all files within `input_filenames` have the same source
 * archive.
 */
bool check_src_files_match(const std::vector<std::string>& input_filenames);

/*!
 * \ingroup HDF5Group
 * \brief Check if all files within `input_filenames` with volume subfile
 * `subfile_name` have the same set of observation ids.
 */
bool check_observation_ids_match(
    const std::vector<std::string>& input_filenames,
    const std::string& subfile_name);

}  // namespace h5
