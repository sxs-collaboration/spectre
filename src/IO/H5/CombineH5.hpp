// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "IO/Logging/Verbosity.hpp"

namespace h5 {
/*!
 * \brief Combine a volume subfile across different HDF5 files.
 *
 * The argument `blocks_to_combine` can list block names and block groups that
 * should be combined. We ignore other blocks when combining the HDF5
 * files. This provides a way to filter volume data for easier visualization.
 */
void combine_h5_vol(const std::vector<std::string>& file_names,
                const std::string& subfile_name, const std::string& output,
                std::optional<double> start_value = std::nullopt,
                std::optional<double> stop_value = std::nullopt,
                const std::optional<std::vector<std::string>>&
                    blocks_to_combine = std::nullopt,
                bool check_src = true);

/*!
 * \brief Combine the `h5::Dat` subfiles of multiple `h5::H5File`s into a single
 * H5 file.
 *
 * \details The times in each `h5::Dat` subfile can be unordered. The necessary
 * sorting will be handled in this function. However, the \p h5_files_to_combine
 * must be mononitcally increasing in time; meaning the earliest time in
 * `File1.h5` must come before the earliest time in `File2.h5`.
 *
 * If there are overlapping times, the "latest" one is always used;
 * meaning if you have data in `File1.h5` and `File2.h5` and if the earliest
 * time in `File2.h5` is before some times in `File1.h5`, those times in
 * `File1.h5` will be discarded and won't appear in the combined H5 file.
 *
 * If the H5 files in \p h5_files_to_combine have other types of subfiles, those
 * will be ignored and will not appear in \p output_h5_filename.
 *
 * If \p h5_files_to_combine is empty, an error will occur.
 *
 * If there are no `h5::Dat` files in the \p h5_files_to_combine, an error will
 * occur.
 *
 * If the legend or version of an `h5::Dat` is not the same in all of
 * \p h5_files_to_combine, an error will occur.
 *
 * \param h5_files_to_combine Vector of H5 files to combine. They must all have
 * the same `h5::Dat` filenames, and those `h5::Dat` subfiles must have the same
 * legends and versions. If not, an error will occur.
 * \param output_h5_filename Name of the combined H5 file. The `h5::Dat` subfile
 * structure will be identical to the ones in \p h5_files_to_combine.
 * \param verbosity Controls how much is printed to stdout. Defaults to no
 * `Verbosity::Silent` or no output.
 */
void combine_h5_dat(const std::vector<std::string>& h5_files_to_combine,
                    const std::string& output_h5_filename,
                    Verbosity verbosity = Verbosity::Silent);
}  // namespace h5
