// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5PropertiesMatch.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/FileSystem.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// Returns all the observation_ids stored in the volume files. Assumes all
// volume files have the same observation ids
std::vector<size_t> get_observation_ids(const std::string& file_prefix,
                                        const std::string& subfile_name) {
  const h5::H5File<h5::AccessType::ReadOnly> initial_file(file_prefix + "0.h5",
                                                          false);
  const auto& initial_volume_file =
      initial_file.get<h5::VolumeData>("/" + subfile_name);
  return initial_volume_file.list_observation_ids();
}

// Returns total number of elements for an observation id across all volume data
// files
size_t get_number_of_elements(const std::vector<std::string>& input_filenames,
                              const std::string& subfile_name,
                              const size_t& observation_id) {
  size_t total_elements = 0;
  for (const auto& input_filename : input_filenames) {
    const h5::H5File<h5::AccessType::ReadOnly> original_file(input_filename,
                                                             false);
    const auto& original_volume_file =
        original_file.get<h5::VolumeData>("/" + subfile_name);
    total_elements += original_volume_file.get_extents(observation_id).size();
  }
  return total_elements;
}

void combine_h5(const std::string& file_prefix, const std::string& subfile_name,
                const std::string& output) {
  // Parses for and stores all input files to be looped over
  const std::vector<std::string>& file_names =
      file_system::glob(file_prefix + "*.h5");

  // Checks that volume data was generated with identical versions of SpECTRE
  if (!h5::check_src_files_match(file_names)) {
    ERROR(
        "One or more of your files were found to have differing src.tar.gz "
        "files, meaning that they may be from differing versions of SpECTRE.");
  }

  // Checks that volume data files contain the same observation ids
  if (!h5::check_observation_ids_match(file_names, subfile_name)) {
    ERROR(
        "One or more of your files were found to have differing observation "
        "ids, meaning they may be from different runs of your SpECTRE "
        "executable or were corrupted.");
  }

  // Braces to specify scope for H5 file
  {
    // Instantiates the output file and the .vol subfile to be filled with the
    // combined data later
    h5::H5File<h5::AccessType::ReadWrite> new_file(output + "0.h5", true);
    new_file.insert<h5::VolumeData>("/" + subfile_name + ".vol");
    new_file.close_current_object();
  }  // End of scope for H5 file

  // Obtains list of observation ids to loop over
  const std::vector<size_t> observation_ids =
      get_observation_ids(file_prefix, subfile_name);

  // Loops over observation ids to write volume data by observation id
  for (const auto& obs_id : observation_ids) {
    // Pre-calculates size of vector to store element data and allocates
    // corresponding memory
    const size_t vector_dim =
        get_number_of_elements(file_names, subfile_name, obs_id);
    std::vector<ElementVolumeData> element_data;
    element_data.reserve(vector_dim);

    double obs_val = 0.0;

    // Loops over input files to append element data into a single vector to be
    // stored in a single H5
    for (auto const& file_name : file_names) {
      const h5::H5File<h5::AccessType::ReadOnly> original_file(file_name,
                                                               false);
      const auto& original_volume_file =
          original_file.get<h5::VolumeData>("/" + subfile_name);
      obs_val = original_volume_file.get_observation_value(obs_id);

      // Get vector of element data for this `obs_id` and `file_name`
      std::vector<ElementVolumeData> data_by_element =
          std::move(std::get<2>(original_volume_file.get_data_by_element(
              obs_val * (1.0 - 4.0 * std::numeric_limits<double>::epsilon()),
              obs_val * (1.0 + 4.0 * std::numeric_limits<double>::epsilon()),
              std::nullopt)[0]));

      // Append vector to total vector of element data for this `obs_id`
      element_data.insert(element_data.end(),
                          std::make_move_iterator(data_by_element.begin()),
                          std::make_move_iterator(data_by_element.end()));
      data_by_element.clear();
      original_file.close_current_object();
    }

    h5::H5File<h5::AccessType::ReadWrite> new_file(output + "0.h5", true);
    auto& new_volume_file = new_file.get<h5::VolumeData>("/" + subfile_name);
    new_volume_file.write_volume_data(obs_id, obs_val, element_data);
    new_file.close_current_object();
  }
}

/*
 * This executable is used for combining a series of HDF5 volume files into one
 * continuous dataset to be stored in a single HDF5 volume file.
 */
int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_prefix", boost::program_options::value<std::string>()->required(),
      "prefix of the files to be combined (omit number and file extension)")(
      "subfile_name", boost::program_options::value<std::string>()->required(),
      "subfile name shared for each volume file in each H5 file (omit file "
      "extension)")("output",
                    boost::program_options::value<std::string>()->required(),
                    "combined output filename (omit file extension)");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_prefix") == 0u or
      vars.count("subfile_name") == 0u or vars.count("output") == 0u) {
    Parallel::printf("%s\n", desc);
    return 1;
  }

  combine_h5(vars["file_prefix"].as<std::string>(),
             vars["subfile_name"].as<std::string>(),
             vars["output"].as<std::string>());
}
