// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/VolumeData.hpp"

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <hdf5.h>
#include <memory>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Version.hpp"

/// \cond HIDDEN_SYMBOLS
namespace h5 {
VolumeData::VolumeData(const bool exists, detail::OpenGroup&& group,
                       const hid_t /*location*/, const std::string& name,
                       const uint32_t version) noexcept
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      version_(version),
      volume_data_group_(group_.id(), name_, h5::AccessType::ReadWrite) {
  if (exists) {
    // We treat this as an internal version for now. We'll need to deal with
    // proper versioning later.
    const Version open_version(true, detail::OpenGroup{},
                               volume_data_group_.id(), "version");
    version_ = open_version.get_version();
    const Header header(true, detail::OpenGroup{}, volume_data_group_.id(),
                        "header");
    header_ = header.get_header();
  } else {  // file does not exist
    {
      Version open_version(false, detail::OpenGroup{},
                           volume_data_group_.id(), "version", version_);
    }
    {
      Header header(false, detail::OpenGroup{}, volume_data_group_.id(),
                    "header");
      header_ = header.get_header();
    }
  }
}

void VolumeData::insert_tensor_data(
    const size_t observation_id, const double observation_value,
    const ExtentsAndTensorVolumeData& extents_and_tensors) noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  if (not contains_attribute(observation_group.id(), "", "observation_value")) {
    h5::write_to_attribute(observation_group.id(), "observation_value",
                           observation_value);
  }
  const auto& first_tensor_name =
      extents_and_tensors.tensor_components.front().name;
  ASSERT(first_tensor_name.find_last_of('/') != std::string::npos,
         "The expected format of the tensor component names is "
         "'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in '"
             << first_tensor_name << "'.");
  const auto spatial_name =
      first_tensor_name.substr(0, first_tensor_name.find_last_of('/'));
  detail::OpenGroup spatial_group(observation_group.id(), spatial_name,
                                  AccessType::ReadWrite);

  // Write extents.
  const auto& extents = extents_and_tensors.extents;
  if (not contains_attribute(spatial_group.id(), "", "extents")) {
    h5::write_to_attribute(spatial_group.id(), "extents", extents);
  }

  // Write the connectivity.
  if (not h5::contains_dataset_or_group(spatial_group.id(), "",
                                        "connectivity")) {
    const std::vector<int> connectivity = [&extents]() noexcept {
      std::vector<int> local_connectivity;
      for (const auto& cell : vis::detail::compute_cells(extents)) {
        for (const auto& bounding_indices : cell.bounding_indices) {
          local_connectivity.emplace_back(bounding_indices);
        }
      }
      return local_connectivity;
    }();
    h5::write_connectivity(spatial_group.id(), connectivity);
  }

  // Write the tensor components.
  for (const auto& tensor_component : extents_and_tensors.tensor_components) {
    ASSERT(tensor_component.name.find_last_of('/') != std::string::npos,
           "The expected format of the tensor component names is "
           "'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in '"
               << tensor_component.name << "'.");
    const auto component_name = tensor_component.name.substr(
        tensor_component.name.find_last_of('/') + 1);
    if (not h5::contains_dataset_or_group(spatial_group.id(), "",
                                          component_name)) {
      h5::write_data(spatial_group.id(), tensor_component.data, component_name);
    } else {
      ERROR("Trying to write tensor component '"
            << component_name
            << "' which already exists in HDF5 file in group '" << name_ << '/'
            << path << '/' << spatial_name << "'.");
    }
  }
}

std::vector<size_t> VolumeData::list_observation_ids() const noexcept {
  const auto names = get_group_names(volume_data_group_.id(), "");
  const auto helper = [](const std::string& s) noexcept {
    return std::stoul(s.substr(std::string("ObservationId").size()));
  };
  return {boost::make_transform_iterator(names.begin(), helper),
          boost::make_transform_iterator(names.end(), helper)};
}

double VolumeData::get_observation_value(const size_t observation_id) const
    noexcept {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  return h5::read_value_attribute<double>(observation_group.id(),
                                          "observation_value");
}

std::vector<std::string> VolumeData::list_grids(
    const size_t observation_id) const noexcept {
  detail::OpenGroup observation_group(
      volume_data_group_.id(),
      "ObservationId" + std::to_string(observation_id), AccessType::ReadOnly);
  return get_group_names(observation_group.id(), "");
}

std::vector<std::string> VolumeData::list_tensor_components(
    size_t observation_id, const std::string& grid_name) const noexcept {
  detail::OpenGroup spatial_group(
      volume_data_group_.id(),
      "ObservationId" + std::to_string(observation_id) + "/" + grid_name,
      AccessType::ReadOnly);
  auto tensor_components = get_group_names(spatial_group.id(), "");
  std::remove(tensor_components.begin(), tensor_components.end(),
              "connectivity");
  tensor_components.pop_back();
  return tensor_components;
}

DataVector VolumeData::get_tensor_component(
    size_t observation_id, const std::string& grid_name,
    const std::string& tensor_component) const noexcept {
  detail::OpenGroup spatial_group(
      volume_data_group_.id(),
      "ObservationId" + std::to_string(observation_id) + "/" + grid_name,
      AccessType::ReadOnly);
  const hid_t dataset_id =
      h5::open_dataset(spatial_group.id(), tensor_component);
  const hid_t dataspace_id = h5::open_dataspace(dataset_id);
  const auto rank =
      static_cast<size_t>(H5Sget_simple_extent_ndims(dataspace_id));
  h5::close_dataspace(dataspace_id);
  h5::close_dataset(dataset_id);
  switch (rank) {
    case 1:
      return h5::read_data<1, DataVector>(spatial_group.id(), tensor_component);
    case 2:
      return h5::read_data<2, DataVector>(spatial_group.id(), tensor_component);
    case 3:
      return h5::read_data<3, DataVector>(spatial_group.id(), tensor_component);
    default:
      ERROR("Rank must be 1, 2, or 3. Received data with Rank = " << rank);
  }
}

std::vector<size_t> VolumeData::get_extents(size_t observation_id,
                                            const std::string& grid_name) const
    noexcept {
  detail::OpenGroup spatial_group(
      volume_data_group_.id(),
      "ObservationId" + std::to_string(observation_id) + "/" + grid_name,
      AccessType::ReadOnly);
  return h5::read_rank1_attribute<size_t>(spatial_group.id(), "extents");
}
}  // namespace h5
/// \endcond HIDDEN_SYMBOLS
