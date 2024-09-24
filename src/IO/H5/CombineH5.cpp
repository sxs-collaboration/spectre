// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/CombineH5.hpp"

#include <boost/program_options.hpp>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5PropertiesMatch.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
// Returns all the observation_ids stored in the volume files. Assumes all
// volume files have the same observation ids
std::vector<std::pair<size_t, double>> get_observation_ids(
    const std::vector<std::string>& file_names,
    const std::string& subfile_name) {
  const h5::H5File<h5::AccessType::ReadOnly> initial_file(file_names[0], false);
  const auto& initial_volume_file =
      initial_file.get<h5::VolumeData>(subfile_name);
  const std::vector<size_t> observation_ids =
      initial_volume_file.list_observation_ids();
  std::vector<std::pair<size_t, double>> observation_ids_and_values(
      observation_ids.size());
  for (size_t i = 0; i < observation_ids.size(); ++i) {
    observation_ids_and_values[i] = std::pair{
        observation_ids[i],
        initial_volume_file.get_observation_value(observation_ids[i])};
  }
  // Sort by the observation value
  alg::sort(observation_ids_and_values,
            [](const std::pair<size_t, double>& id_and_value_a,
               const std::pair<size_t, double>& id_and_value_b) {
              return id_and_value_a.second < id_and_value_b.second;
            });
  return observation_ids_and_values;
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
        original_file.get<h5::VolumeData>(subfile_name);
    total_elements += original_volume_file.get_extents(observation_id).size();
  }
  return total_elements;
}

std::optional<std::unordered_set<size_t>> get_block_numbers_to_use(
    const std::string& file_name, const std::string& subfile_name,
    const size_t observation_id,
    const std::optional<std::vector<std::string>>& blocks_to_combine) {
  if (not blocks_to_combine.has_value() or blocks_to_combine.value().empty()) {
    return std::nullopt;
  }

  const h5::H5File<h5::AccessType::ReadOnly> original_file(file_name, false);
  const auto& volume_file = original_file.get<h5::VolumeData>(subfile_name);

  const auto dim = volume_file.get_dimension();
  auto serialized_domain = volume_file.get_domain(observation_id);
  if (not serialized_domain.has_value()) {
    ERROR("Could not read the domain the from file "
          << file_name << " and subfile " << subfile_name
          << ". This means we cannot filter based on block names. You can "
             "still combine the files but will need to use all blocks.");
  }
  std::unordered_set<std::string> block_names_to_combine{};
  std::vector<std::string> block_names_in_domain{};
  switch (dim) {
    case 1: {
      const auto domain =
          deserialize<Domain<1>>(serialized_domain.value().data());
      block_names_in_domain = domain.block_names();
      block_names_to_combine = domain::expand_block_groups_to_block_names(
          blocks_to_combine.value(), domain.block_names(),
          domain.block_groups());
      break;
    }
    case 2: {
      const auto domain =
          deserialize<Domain<2>>(serialized_domain.value().data());
      block_names_in_domain = domain.block_names();
      block_names_to_combine = domain::expand_block_groups_to_block_names(
          blocks_to_combine.value(), domain.block_names(),
          domain.block_groups());
      break;
    }
    case 3: {
      const auto domain =
          deserialize<Domain<3>>(serialized_domain.value().data());
      block_names_in_domain = domain.block_names();
      block_names_to_combine = domain::expand_block_groups_to_block_names(
          blocks_to_combine.value(), domain.block_names(),
          domain.block_groups());
      break;
    }
    default:
      ERROR("Only can handle 1, 2, or 3d domains not " << dim);
  };

  std::unordered_set<size_t> blocks_to_use{};
  for (const std::string& block_to_combine : block_names_to_combine) {
    auto location_it = alg::find(block_names_in_domain, block_to_combine);
    if (location_it == block_names_in_domain.end()) {
      ERROR("Block name " << block_to_combine << " not found.");
    }
    blocks_to_use.insert(static_cast<size_t>(
        std::distance(block_names_in_domain.begin(), location_it)));
  }

  return blocks_to_use;
}
}  // namespace

namespace h5 {

void combine_h5(
    const std::vector<std::string>& file_names, const std::string& subfile_name,
    const std::string& output, const std::optional<double> start_value,
    const std::optional<double> stop_value,
    const std::optional<std::vector<std::string>>& blocks_to_combine,
    const bool check_src) {
  // Parses for and stores all input files to be looped over
  Parallel::printf("Processing files:\n%s\n",
                   std::string{MakeString{} << file_names}.c_str());

  // Checks that volume data was generated with identical versions of SpECTRE
  if (check_src) {
    if (not h5::check_src_files_match(file_names)) {
      ERROR(
          "One or more of your files were found to have differing src.tar.gz "
          "files, meaning that they may be from differing versions of "
          "SpECTRE.");
    }
  }

  // Checks that volume data files contain the same observation ids
  if (not h5::check_observation_ids_match(file_names, subfile_name)) {
    ERROR(
        "One or more of your files were found to have differing observation "
        "ids, meaning they may be from different runs of your SpECTRE "
        "executable or were corrupted.");
  }

  // Braces to specify scope for H5 file
  {
    // Instantiates the output file and the .vol subfile to be filled with the
    // combined data later
    Parallel::printf("Creating output file: %s\n", output.c_str());
    h5::H5File<h5::AccessType::ReadWrite> new_file(output, true);
    new_file.insert<h5::VolumeData>(subfile_name);
    new_file.close_current_object();
  }  // End of scope for H5 file

  // Obtains list of observation ids to loop over
  const std::vector<std::pair<size_t, double>> observation_ids_and_values =
      get_observation_ids(file_names, subfile_name);

  if (observation_ids_and_values.empty()) {
    ERROR("No observation IDs found in subfile" << subfile_name);
  }

  const std::optional<std::unordered_set<size_t>> blocks_to_use =
      get_block_numbers_to_use(file_names[0], subfile_name,
                               observation_ids_and_values[0].first,
                               blocks_to_combine);

  // Loops over observation ids to write volume data by observation id
  for (size_t obs_index = 0; obs_index < observation_ids_and_values.size();
       ++obs_index) {
    const double obs_value = observation_ids_and_values[obs_index].second;
    if (obs_value > stop_value.value_or(std::numeric_limits<double>::max()) or
        obs_value <
            start_value.value_or(std::numeric_limits<double>::lowest())) {
      Parallel::printf("Skipping observation value %1.6e\n", obs_value);
      continue;
    }

    const size_t obs_id = observation_ids_and_values[obs_index].first;
    // Pre-calculates size of vector to store element data and allocates
    // corresponding memory
    const size_t vector_dim =
        get_number_of_elements(file_names, subfile_name, obs_id);
    std::vector<ElementVolumeData> element_data;
    element_data.reserve(vector_dim);

    double obs_val = 0.0;
    std::optional<std::vector<char>> serialized_domain{};
    std::optional<std::vector<char>> serialized_functions_of_time{};

    // Loops over input files to append element data into a single vector to be
    // stored in a single H5
    bool printed = false;
    for (auto const& file_name : file_names) {
      const h5::H5File<h5::AccessType::ReadOnly> original_file(file_name,
                                                               false);
      const auto& original_volume_file =
          original_file.get<h5::VolumeData>(subfile_name);
      obs_val = original_volume_file.get_observation_value(obs_id);
      if (not printed) {
        Parallel::printf(
            "Processing obsevation ID %lo (%lo/%lo) with value %1.14e\n",
            obs_id, obs_index, observation_ids_and_values.size(), obs_val);
        printed = true;
      }
      Parallel::printf("  Processing file: %s\n", file_name.c_str());

      const auto dim = original_volume_file.get_dimension();
      serialized_domain = original_volume_file.get_domain(obs_id);
      serialized_functions_of_time =
          original_volume_file.get_functions_of_time(obs_id);

      // Get vector of element data for this `obs_id` and `file_name`
      std::vector<ElementVolumeData> data_by_element =
          std::move(std::get<2>(original_volume_file.get_data_by_element(
              obs_val * (1.0 - 4.0 * std::numeric_limits<double>::epsilon()),
              obs_val * (1.0 + 4.0 * std::numeric_limits<double>::epsilon()),
              std::nullopt)[0]));

      auto end_it = data_by_element.end();

      if (blocks_to_use.has_value()) {
        end_it = alg::remove_if(
            data_by_element,
            [&blocks_to_use, &dim](const ElementVolumeData& element) -> bool {
              switch (dim) {
                case 1:
                  return not blocks_to_use->contains(
                      ElementId<1>{element.element_name}.block_id());
                case 2:
                  return not blocks_to_use->contains(
                      ElementId<2>{element.element_name}.block_id());
                case 3:
                  return not blocks_to_use->contains(
                      ElementId<3>{element.element_name}.block_id());
                default:
                  ERROR("Only can handle 1, 2, or 3d domains but got " << dim);
              };
            });
      }

      // Append vector to total vector of element data for this `obs_id`
      element_data.insert(element_data.end(),
                          std::make_move_iterator(data_by_element.begin()),
                          std::make_move_iterator(end_it));
      data_by_element.clear();
      original_file.close_current_object();
    }

    h5::H5File<h5::AccessType::ReadWrite> new_file(output, true);
    auto& new_volume_file = new_file.get<h5::VolumeData>(subfile_name);
    new_volume_file.write_volume_data(obs_id, obs_val, element_data,
                                      serialized_domain,
                                      serialized_functions_of_time);
    new_file.close_current_object();
  }
}
}  // namespace h5
