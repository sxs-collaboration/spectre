// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/CheckH5PropertiesMatch.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/VolumeData.hpp"

namespace h5 {
bool check_src_files_match(const std::vector<std::string>& input_filenames) {
  // Holds source archive data of first file
  std::vector<char> src_tar_initial;

  // Braces to specify scope for H5 file
  {
    // Reads the 0th volume file to compare against
    const h5::H5File<h5::AccessType::ReadOnly> initial_file(input_filenames[0],
                                                            false);

    // Obtains the source archive of the 0th volume file to compare against
    const auto& src_archive_object_initial =
        initial_file.get<h5::SourceArchive>("/src");
    src_tar_initial = src_archive_object_initial.get_archive();
  }  // End of scope for H5 file

  // Iterates through each of the other files and compares source archives
  for (size_t i = 1; i < input_filenames.size(); ++i) {
    // Reads the i-th volume file to compare with
    const h5::H5File<h5::AccessType::ReadOnly> comparison_file(
        input_filenames[i], false);

    // Obtains the source archive of the i-th volume file to compare with
    const auto& src_archive_object_compare =
        comparison_file.get<h5::SourceArchive>("/src");
    const std::vector<char>& src_tar_compare =
        src_archive_object_compare.get_archive();

    // Returns false if the source archive data does not match
    if (src_tar_initial != src_tar_compare) {
      return false;
    }
  }
  return true;
}

bool check_observation_ids_match(
    const std::vector<std::string>& input_filenames,
    const std::string& subfile_name) {
  // Holds observation id data of first file
  std::vector<size_t> obs_id_initial;

  // Braces to specify scope for H5 file
  {
    // Reads the 0th volume file to compare against
    const h5::H5File<h5::AccessType::ReadOnly> initial_file(input_filenames[0],
                                                            false);

    // Obtains the list of observation ids of the 0th volume file to compare
    // against
    const auto& volume_data_initial =
        initial_file.get<h5::VolumeData>("/" + subfile_name);
    obs_id_initial = volume_data_initial.list_observation_ids();
  }  // End of scope for H5 file

  // Iterates through each of the other files and compares observation id lists
  for (size_t i = 1; i < input_filenames.size(); ++i) {
    // Reads the i-th volume file to compare against
    const h5::H5File<h5::AccessType::ReadOnly> comparison_file(
        input_filenames[i], false);

    // Obtains the observation id list of the i-th volume file to compare with
    const auto& volume_data_compare =
        comparison_file.get<h5::VolumeData>("/" + subfile_name);
    const std::vector<size_t>& obs_id_compare =
        volume_data_compare.list_observation_ids();

    // Returns false if an observation id does not match
    if (obs_id_initial != obs_id_compare) {
      return false;
    }
  }
  return true;
}

}  // namespace h5
