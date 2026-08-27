// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/ModalTimeSeriesReader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"

namespace spectre::Exporter {

namespace {

std::vector<std::string> resolve_filenames(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob) {
  std::vector<std::string> filenames =
      std::visit(Overloader{[](const std::vector<std::string>& volume_files) {
                              return volume_files;
                            },
                            [](const std::string& volume_files_glob) {
                              return file_system::glob(volume_files_glob);
                            }},
                 volume_files_or_glob);
  if (filenames.empty()) {
    ERROR_NO_TRACE("No volume files found. Specify at least one volume file.");
  }
  return filenames;
}

template <size_t Dim>
void enforce_dimension(const h5::VolumeData& volfile,
                       const std::string& filename) {
  if (volfile.get_dimension() != Dim) {
    ERROR_NO_TRACE("Mismatched dimensions: expected "
                   << Dim << "D volume data, but got "
                   << volfile.get_dimension() << "D in file '" << filename
                   << "'.");
  }
}

std::vector<std::pair<size_t, double>> observation_ids_and_times(
    const h5::VolumeData& volfile, const std::optional<double> start_time,
    const std::optional<double> end_time) {
  const auto observation_ids = volfile.list_observation_ids();
  std::vector<std::pair<size_t, double>> result{};
  result.reserve(observation_ids.size());
  for (const size_t observation_id : observation_ids) {
    const double observation_time =
        volfile.get_observation_value(observation_id);
    if ((start_time.has_value() and observation_time < start_time.value()) or
        (end_time.has_value() and observation_time > end_time.value())) {
      continue;
    }
    result.emplace_back(observation_id, observation_time);
  }
  return result;
}

template <size_t Dim>
void enforce_legendre_basis(const Mesh<Dim>& mesh, const std::string& context) {
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(mesh.basis(), d) != Spectral::Basis::Legendre) {
      ERROR_NO_TRACE("Only the Legendre basis is supported, but found "
                     << gsl::at(mesh.basis(), d) << " in dimension " << d
                     << " for " << context << ".");
    }
  }
}

}  // namespace

namespace ModalTimeSeriesReader_detail {

// Data of one observation in one file that is needed to locate and interpret
// each element's subset of the tensor component datasets
struct ObservationCache {
  size_t obs_id{};
  std::unordered_map<std::string, size_t> grid_index_by_name{};
  // Offset of each grid into the contiguous datasets, plus a total size at
  // the end
  std::vector<size_t> grid_offsets{};
  std::vector<std::vector<size_t>> extents{};
  std::vector<std::vector<Spectral::Basis>> bases{};
  std::vector<std::vector<Spectral::Quadrature>> quadratures{};
};

}  // namespace ModalTimeSeriesReader_detail

namespace {

// The functions operating on the `ObservationCache` are only used in this
// file. Only the struct itself needs external linkage because it is a member
// of `FileCache` below (-Wsubobject-linkage).
using ModalTimeSeriesReader_detail::ObservationCache;

ObservationCache make_observation_cache(const h5::VolumeData& volfile,
                                        const size_t obs_id,
                                        const std::string& filename) {
  ObservationCache cache{};
  cache.obs_id = obs_id;
  const auto grid_names = volfile.get_grid_names(obs_id);
  cache.extents = volfile.get_extents(obs_id);
  cache.bases = volfile.get_bases(obs_id);
  cache.quadratures = volfile.get_quadratures(obs_id);
  cache.grid_index_by_name.reserve(grid_names.size());
  cache.grid_offsets.reserve(grid_names.size() + 1);
  cache.grid_offsets.push_back(0);
  for (size_t grid_index = 0; grid_index < grid_names.size(); ++grid_index) {
    if (not cache.grid_index_by_name.emplace(grid_names[grid_index], grid_index)
                .second) {
      ERROR_NO_TRACE("Grid '" << grid_names[grid_index]
                              << "' appears multiple times at observation ID "
                              << obs_id << " in file '" << filename << "'.");
    }
    size_t num_points = 1;
    for (const size_t extent : cache.extents[grid_index]) {
      num_points *= extent;
    }
    cache.grid_offsets.push_back(cache.grid_offsets.back() + num_points);
  }
  return cache;
}

template <size_t Dim>
Mesh<Dim> mesh_from_cache(const ObservationCache& cache,
                          const size_t grid_index) {
  std::array<size_t, Dim> extents{};
  std::array<Spectral::Basis, Dim> bases{};
  std::array<Spectral::Quadrature, Dim> quadratures{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(extents, d) = cache.extents[grid_index][d];
    gsl::at(bases, d) = cache.bases[grid_index][d];
    gsl::at(quadratures, d) = cache.quadratures[grid_index][d];
  }
  return Mesh<Dim>{extents, bases, quadratures};
}

}  // namespace

// Metadata of one volume file, cached between `modal_time_series()` calls so
// requesting elements grouped by file avoids re-reading the metadata
template <size_t Dim>
struct ModalTimeSeriesReader<Dim>::FileCache {
  FileCache(const size_t in_file_index, const std::string& filename,
            const std::string& subfile_name,
            const std::vector<std::pair<size_t, double>>& obs_ids_and_times,
            const size_t num_elements_in_file)
      : file_index(in_file_index),
        h5file(filename),
        volfile(&h5file.get<h5::VolumeData>(subfile_name)) {
    observations.reserve(obs_ids_and_times.size());
    for (const auto& id_and_time : obs_ids_and_times) {
      const size_t obs_id = id_and_time.first;
      observations.push_back(
          make_observation_cache(*volfile, obs_id, filename));
      if (observations.back().grid_index_by_name.size() !=
          num_elements_in_file) {
        ERROR_NO_TRACE("File '"
                       << filename << "' contains "
                       << observations.back().grid_index_by_name.size()
                       << " elements at observation ID " << obs_id << " but "
                       << num_elements_in_file
                       << " elements at the last observation. Adaptive mesh "
                          "refinement and element migration between files "
                          "(e.g. by load balancing) are not supported.");
      }
    }
  }

  size_t file_index;
  h5::H5File<h5::AccessType::ReadOnly> h5file;
  const h5::VolumeData* volfile;
  std::vector<ModalTimeSeriesReader_detail::ObservationCache> observations{};
};

template <size_t Dim>
ModalTimeSeriesReader<Dim>::ModalTimeSeriesReader(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    std::string subfile_name, std::vector<std::string> tensor_components,
    const std::optional<double> start_time,
    const std::optional<double> end_time)
    : filenames_(resolve_filenames(volume_files_or_glob)),
      subfile_name_(std::move(subfile_name)),
      tensor_components_(std::move(tensor_components)) {
  // Load the observation IDs and times from the first file, restrict them to
  // the requested time interval, and check they are uniformly spaced
  {
    const h5::H5File<h5::AccessType::ReadOnly> first_h5file(filenames_.front());
    const auto& volfile = first_h5file.get<h5::VolumeData>(subfile_name_);
    enforce_dimension<Dim>(volfile, filenames_.front());
    obs_ids_and_times_ =
        observation_ids_and_times(volfile, start_time, end_time);
  }
  if (obs_ids_and_times_.size() < 2) {
    ERROR_NO_TRACE("At least 2 observations are required, but found "
                   << obs_ids_and_times_.size() << " in subfile '"
                   << subfile_name_ << "'"
                   << (start_time.has_value() or end_time.has_value()
                           ? " after restricting to the requested time "
                             "interval."
                           : "."));
  }
  const double first_time = obs_ids_and_times_.front().second;
  time_step_ = obs_ids_and_times_[1].second - first_time;
  const double relative_epsilon =
      100.0 * std::numeric_limits<double>::epsilon();
  if (time_step_ <= 0.0) {
    ERROR_NO_TRACE("Observation times in subfile '"
                   << subfile_name_
                   << "' must be strictly increasing, but the first time step "
                      "is "
                   << time_step_
                   << ". Select a strictly increasing interval with "
                      "start_time and end_time or preprocess the volume "
                      "files.");
  }
  for (size_t i = 2; i < obs_ids_and_times_.size(); ++i) {
    const double actual_time = obs_ids_and_times_[i].second;
    const double expected_time =
        first_time + static_cast<double>(i) * time_step_;
    const double scale =
        std::max({1.0, std::abs(expected_time), std::abs(actual_time)});
    if (not equal_within_roundoff(actual_time, expected_time, relative_epsilon,
                                  scale)) {
      ERROR_NO_TRACE(
          "Observation times in subfile '"
          << subfile_name_
          << "' must be uniformly spaced, but observation index " << i
          << " has time " << actual_time << " where " << expected_time
          << " was expected from the first time " << first_time
          << " and the time step " << time_step_
          << ". Select a uniformly spaced interval with start_time and "
             "end_time or preprocess the volume files.");
    }
  }

  // Check that all other files contain the same observations
  for (size_t file_index = 1; file_index < filenames_.size(); ++file_index) {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filenames_[file_index]);
    const auto& volfile = h5file.get<h5::VolumeData>(subfile_name_);
    enforce_dimension<Dim>(volfile, filenames_[file_index]);
    const auto other_obs_ids_and_times =
        observation_ids_and_times(volfile, start_time, end_time);
    if (other_obs_ids_and_times.size() != obs_ids_and_times_.size()) {
      ERROR_NO_TRACE("File '"
                     << filenames_[file_index] << "' contains "
                     << other_obs_ids_and_times.size()
                     << " observations in the requested time interval, but "
                     << "file '" << filenames_.front() << "' contains "
                     << obs_ids_and_times_.size()
                     << ". All volume files must contain the same "
                        "observations in the requested time interval.");
    }
    for (size_t obs_index = 0; obs_index < obs_ids_and_times_.size();
         ++obs_index) {
      const auto& [obs_id, obs_time] = obs_ids_and_times_[obs_index];
      const auto& [other_obs_id, other_time] =
          other_obs_ids_and_times[obs_index];
      if (other_obs_id != obs_id) {
        ERROR_NO_TRACE("Mismatched observation ID at observation index "
                       << obs_index << ": expected " << obs_id << " from file '"
                       << filenames_.front() << "' but found " << other_obs_id
                       << " in file '" << filenames_[file_index]
                       << "'. All volume files must contain the same "
                          "observations in the requested time interval.");
      }
      if (other_time != obs_time) {
        ERROR_NO_TRACE("Mismatched observation value for observation ID "
                       << obs_id << ": expected " << obs_time << " from file '"
                       << filenames_.front() << "' but found " << other_time
                       << " in file '" << filenames_[file_index] << "'.");
      }
    }
  }

  // Gather the elements and their meshes from all files at the last
  // observation. Consistency with the other observations is checked when
  // reading the data in `modal_time_series`.
  const size_t reference_obs_id = obs_ids_and_times_.back().first;
  for (size_t file_index = 0; file_index < filenames_.size(); ++file_index) {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filenames_[file_index]);
    const auto& volfile = h5file.get<h5::VolumeData>(subfile_name_);
    const auto grid_names = volfile.get_grid_names(reference_obs_id);
    const auto all_extents = volfile.get_extents(reference_obs_id);
    const auto all_bases = volfile.get_bases(reference_obs_id);
    const auto all_quadratures = volfile.get_quadratures(reference_obs_id);
    for (const auto& grid_name : grid_names) {
      const ElementId<Dim> element_id(grid_name);
      const auto mesh = h5::mesh_for_grid<Dim>(
          grid_name, grid_names, all_extents, all_bases, all_quadratures);
      enforce_legendre_basis(mesh, "element " + grid_name + " in file '" +
                                       filenames_[file_index] + "'");
      if (const auto existing_it = element_info_.find(element_id);
          existing_it != element_info_.end()) {
        ERROR_NO_TRACE("Element "
                       << grid_name << " in file '" << filenames_[file_index]
                       << "' already exists in file '"
                       << filenames_[existing_it->second.second]
                       << "'. Each element must reside in exactly one volume "
                          "file.");
      }
      element_info_.emplace(element_id, std::make_pair(mesh, file_index));
      elements_.emplace_back(element_id, mesh);
    }
  }
}

template <size_t Dim>
ModalTimeSeriesReader<Dim>::ModalTimeSeriesReader(ModalTimeSeriesReader&&) =
    default;
template <size_t Dim>
ModalTimeSeriesReader<Dim>& ModalTimeSeriesReader<Dim>::operator=(
    ModalTimeSeriesReader&&) = default;
template <size_t Dim>
ModalTimeSeriesReader<Dim>::~ModalTimeSeriesReader() = default;

template <size_t Dim>
typename ModalTimeSeriesReader<Dim>::Series
ModalTimeSeriesReader<Dim>::modal_time_series(
    const ElementId<Dim>& element_id) {
  const auto info_it = element_info_.find(element_id);
  if (info_it == element_info_.end()) {
    ERROR_NO_TRACE("Element "
                   << element_id
                   << " does not exist in the volume files. The available "
                      "elements are listed by `elements()`.");
  }
  const auto& [mesh, file_index] = info_it->second;
  const std::string& filename = filenames_[file_index];
  if (file_cache_ == nullptr or file_cache_->file_index != file_index) {
    size_t num_elements_in_file = 0;
    for (const auto& info : element_info_) {
      if (info.second.second == file_index) {
        ++num_elements_in_file;
      }
    }
    file_cache_ =
        std::make_unique<FileCache>(file_index, filename, subfile_name_,
                                    obs_ids_and_times_, num_elements_in_file);
  }
  const auto& volfile = *file_cache_->volfile;
  const size_t num_observations = obs_ids_and_times_.size();
  const size_t num_components = tensor_components_.size();
  const std::string element_name = get_output(element_id);
  const size_t num_points = mesh.number_of_grid_points();
  Series series(num_components,
                std::vector<std::vector<double>>(
                    num_points, std::vector<double>(num_observations)));
  DataVector nodal_data(num_points);
  ModalVector modal_data(num_points);
  for (size_t obs_index = 0; obs_index < num_observations; ++obs_index) {
    const auto& obs_cache = file_cache_->observations[obs_index];
    const auto grid_index_it = obs_cache.grid_index_by_name.find(element_name);
    if (grid_index_it == obs_cache.grid_index_by_name.end()) {
      ERROR_NO_TRACE("Element "
                     << element_name << " is missing in file '" << filename
                     << "' at observation ID " << obs_cache.obs_id
                     << ". Each element must reside in the same volume "
                        "file for all observations, so element migration "
                        "between files (e.g. by load balancing) is not "
                        "supported.");
    }
    const size_t grid_index = grid_index_it->second;
    const auto obs_mesh = mesh_from_cache<Dim>(obs_cache, grid_index);
    if (obs_mesh != mesh) {
      ERROR_NO_TRACE("Element "
                     << element_name << " in file '" << filename
                     << "' has mesh " << obs_mesh << " at observation ID "
                     << obs_cache.obs_id << " but mesh " << mesh
                     << " at the last observation. Mesh changes between "
                        "observations (e.g. by adaptive mesh refinement) "
                        "are not supported.");
    }
    const size_t offset = obs_cache.grid_offsets[grid_index];
    for (size_t component_index = 0; component_index < num_components;
         ++component_index) {
      // This currently reads the data of all elements in the file and
      // discards all but this element's subset. Reading only the subset is a
      // possible optimization, see docs of `modal_time_series`.
      const auto tensor_component = volfile.get_tensor_component(
          obs_cache.obs_id, tensor_components_[component_index]);
      std::visit(
          Overloader{
              [&nodal_data, &offset, &num_points](const DataVector& data) {
                std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(offset),
                            static_cast<std::ptrdiff_t>(num_points),
                            nodal_data.begin());
              },
              [&nodal_data, &offset,
               &num_points](const std::vector<float>& data) {
                std::transform(
                    data.begin() + static_cast<std::ptrdiff_t>(offset),
                    data.begin() +
                        static_cast<std::ptrdiff_t>(offset + num_points),
                    nodal_data.begin(), [](const float value) {
                      return static_cast<double>(value);
                    });
              }},
          tensor_component.data);
      to_modal_coefficients(make_not_null(&modal_data), nodal_data, mesh);
      auto& component_series = series[component_index];
      for (size_t mode = 0; mode < num_points; ++mode) {
        component_series[mode][obs_index] = modal_data[mode];
      }
    }
  }
  return series;
}

// Explicit instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ModalTimeSeriesReader<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace spectre::Exporter
