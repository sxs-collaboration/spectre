// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/VolumeData.hpp"

#include <algorithm>
#include <array>
#include <boost/algorithm/string.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <hdf5.h>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/ExtendConnectivityHelpers.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/SpectralIo.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
namespace {
// Append the element extents and connectevity to the total extents and
// connectivity
void append_element_extents_and_connectivity(
    const gsl::not_null<std::vector<size_t>*> total_extents,
    const gsl::not_null<std::vector<int>*> total_connectivity,
    const gsl::not_null<std::vector<int>*> pole_connectivity,
    const gsl::not_null<int*> total_points_so_far, const size_t dim,
    const ElementVolumeData& element) {
  // Process the element extents
  const auto& extents = element.extents;
  ASSERT(alg::none_of(extents, [](const size_t extent) { return extent == 1; }),
         "We cannot generate connectivity for any single grid point elements.");
  if (extents.size() != dim) {
    ERROR("Trying to write data of dimensionality"
          << extents.size() << "but the VolumeData file has dimensionality"
          << dim << ".");
  }
  total_extents->insert(total_extents->end(), extents.begin(), extents.end());
  // Find the number of points in the local connectivity
  const int element_num_points =
      alg::accumulate(extents, 1, std::multiplies<>{});
  // Generate the connectivity data for the element
  // Possible optimization: local_connectivity.reserve(BLAH) if we can figure
  // out size without computing all the connectivities.
  const std::vector<int> connectivity = [&extents, &total_points_so_far]() {
    std::vector<int> local_connectivity;
    for (const auto& cell : vis::detail::compute_cells(extents)) {
      for (const auto& bounding_indices : cell.bounding_indices) {
        local_connectivity.emplace_back(*total_points_so_far +
                                        static_cast<int>(bounding_indices));
      }
    }
    return local_connectivity;
  }();
  *total_points_so_far += element_num_points;
  total_connectivity->insert(total_connectivity->end(), connectivity.begin(),
                             connectivity.end());

  // If element is 2D and the bases are both SphericalHarmonic,
  // then add extra connections to close the surface.
  if (dim == 2) {
    if (element.basis[0] == SpatialDiscretization::Basis::SphericalHarmonic and
        element.basis[1] == SpatialDiscretization::Basis::SphericalHarmonic) {
      // Extents are (l+1, 2l+1)
      const size_t l = element.extents[0] - 1;

      // Connect max(phi) and min(phi) by adding more quads
      // to total_connectivity
      for (size_t j = 0; j < l; ++j) {
        total_connectivity->push_back(j);
        total_connectivity->push_back(j + 1);
        total_connectivity->push_back(2 * l * (l + 1) + j + 1);
        total_connectivity->push_back((2 * l) * (l + 1) + j);
      }

      // Add a new connectivity output for filling the poles
      // First, get the points at min(theta), which define the
      // boundary of the top pole to fill, and the points at
      // max(theta), which define the boundary of the bottom
      // pole to fill. Note: points are stored with theta
      // varying faster than phi.
      std::vector<int> top_pole_points{};
      std::vector<int> bottom_pole_points{};
      for (size_t k = 0; k < (2 * l + 1); ++k) {
        top_pole_points.push_back(k * (l + 1));
        bottom_pole_points.push_back(k * (l + 1) + l);
      }

      // Fill the poles with triangles. Start by connecting
      // points 0,1,2, 2,3,4, etc. into small triangles,
      // then connect points 0,2,4, 4,6,8, etc.,
      // etc., until fewer than 3 points remain.
      const size_t number_of_points_near_poles = top_pole_points.size();
      size_t to_next_triangle_point = 1;
      while (number_of_points_near_poles / to_next_triangle_point >= 3) {
        for (size_t point_starting_triangle = 0;
             point_starting_triangle <
             number_of_points_near_poles - 2 * to_next_triangle_point;
             point_starting_triangle += 2 * to_next_triangle_point) {
          pole_connectivity->push_back(
              gsl::at(top_pole_points, point_starting_triangle));
          pole_connectivity->push_back(
              gsl::at(top_pole_points,
                      point_starting_triangle + to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(top_pole_points,
                      point_starting_triangle + 2 * to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points, point_starting_triangle));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points,
                      point_starting_triangle + to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points,
                      point_starting_triangle + 2 * to_next_triangle_point));
        }
        // If odd number of points, add triangle closing
        // point at max(phi) and point at min(phi)
        if (number_of_points_near_poles % 2 != 0 and
            2 * to_next_triangle_point < number_of_points_near_poles) {
          pole_connectivity->push_back(gsl::at(
              top_pole_points,
              number_of_points_near_poles - 2 * to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(top_pole_points,
                      number_of_points_near_poles - to_next_triangle_point));
          pole_connectivity->push_back(gsl::at(top_pole_points, 0));
          pole_connectivity->push_back(gsl::at(
              bottom_pole_points,
              number_of_points_near_poles - 2 * to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points,
                      number_of_points_near_poles - to_next_triangle_point));
          pole_connectivity->push_back(gsl::at(bottom_pole_points, 0));
        }
        to_next_triangle_point += 1;
      }
    }
  }
}

}  // namespace

VolumeData::VolumeData(const bool subfile_exists, detail::OpenGroup&& group,
                       const hid_t /*location*/, const std::string& name,
                       const uint32_t version)
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      version_(version),
      volume_data_group_(group_.id(), name_, h5::AccessType::ReadWrite) {
  if (subfile_exists) {
    // We treat this as an internal version for now. We'll need to deal with
    // proper versioning later.
    const Version open_version(true, detail::OpenGroup{},
                               volume_data_group_.id(), "version");
    version_ = open_version.get_version();
    const Header header(true, detail::OpenGroup{}, volume_data_group_.id(),
                        "header");
    header_ = header.get_header();
  } else {  // file does not exist
    // Subfiles are closed as they go out of scope, so we have the extra
    // braces here to add the necessary scope
    {
      Version open_version(false, detail::OpenGroup{}, volume_data_group_.id(),
                           "version", version_);
    }
    {
      Header header(false, detail::OpenGroup{}, volume_data_group_.id(),
                    "header");
      header_ = header.get_header();
    }
  }
}

// Write Volume Data stored in a vector of `ElementVolumeData` to
// an `observation_group` in a `VolumeData` file.
void VolumeData::write_volume_data(
    const size_t observation_id, const double observation_value,
    const std::vector<ElementVolumeData>& elements,
    const std::optional<std::vector<char>>& serialized_domain,
    const std::optional<std::vector<char>>& serialized_functions_of_time) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  if (contains_attribute(observation_group.id(), "", "observation_value")) {
    ERROR_NO_TRACE("Trying to write ObservationId "
                   << std::to_string(observation_id)
                   << " with observation_value " << observation_group.id()
                   << " which already exists in file at " << path
                   << ". Did you forget to clean up after an earlier run?");
  }
  h5::write_to_attribute(observation_group.id(), "observation_value",
                         observation_value);
  // Get first element to extract the component names and dimension
  const auto get_component_name = [](const auto& component) {
    ASSERT(component.name.find_last_of('/') == std::string::npos,
           "The expected format of the tensor component names is "
           "'COMPONENT_NAME' but found a '/' in '"
               << component.name << "'.");
    return component.name;
  };
  const std::vector<std::string> component_names(
      boost::make_transform_iterator(elements.front().tensor_components.begin(),
                                     get_component_name),
      boost::make_transform_iterator(elements.front().tensor_components.end(),
                                     get_component_name));
  // The dimension of the grid is the number of extents per element. I.e., if
  // the extents are [8,5,7] for any element, the dimension of the grid is 3.
  // Only written once per VolumeData file (All volume data in a single file
  // should have the same dimensionality)
  if (not contains_attribute(volume_data_group_.id(), "", "dimension")) {
    h5::write_to_attribute(volume_data_group_.id(), "dimension",
                           elements.front().extents.size());
  }
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  // Extract Tensor Data one component at a time
  std::vector<size_t> total_extents;
  std::string grid_names;
  std::vector<int> total_connectivity;
  std::vector<int> pole_connectivity{};
  std::vector<int> quadratures;
  std::vector<int> bases;
  // Keep a running count of the number of points so far to use as a global
  // index for the connectivity
  int total_points_so_far = 0;
  // Loop over tensor components
  for (size_t i = 0; i < component_names.size(); i++) {
    std::string component_name = component_names[i];
    // Write the data for the tensor component
    if (h5::contains_dataset_or_group(observation_group.id(), "",
                                      component_name)) {
      ERROR("Trying to write tensor component '"
            << component_name
            << "' which already exists in HDF5 file in group '" << name_ << '/'
            << "ObservationId" << std::to_string(observation_id) << "'");
    }

    const auto fill_and_write_contiguous_tensor_data =
        [&bases, &component_name, &dim, &elements, &grid_names, i,
         &observation_group, &quadratures, &total_connectivity,
         &pole_connectivity, &total_extents,
         &total_points_so_far](const auto contiguous_tensor_data_ptr) {
          for (const auto& element : elements) {
            if (UNLIKELY(i == 0)) {
              // True if first tensor component being accessed
              grid_names += element.element_name + h5::VolumeData::separator();
              // append element basis
              alg::transform(element.basis, std::back_inserter(bases),
                             [](const SpatialDiscretization::Basis t) {
                               return static_cast<int>(t);
                             });
              // append element quadraature
              alg::transform(element.quadrature,
                             std::back_inserter(quadratures),
                             [](const SpatialDiscretization::Quadrature t) {
                               return static_cast<int>(t);
                             });

              append_element_extents_and_connectivity(
                  &total_extents, &total_connectivity, &pole_connectivity,
                  &total_points_so_far, dim, element);
            }
            using type_from_variant = tmpl::conditional_t<
                std::is_same_v<
                    std::decay_t<decltype(*contiguous_tensor_data_ptr)>,
                    std::vector<double>>,
                DataVector, std::vector<float>>;
            contiguous_tensor_data_ptr->insert(
                contiguous_tensor_data_ptr->end(),
                std::get<type_from_variant>(element.tensor_components[i].data)
                    .begin(),
                std::get<type_from_variant>(element.tensor_components[i].data)
                    .end());
          }  // for each element
          h5::write_data(observation_group.id(), *contiguous_tensor_data_ptr,
                         {contiguous_tensor_data_ptr->size()}, component_name);
        };

    if (elements[0].tensor_components[i].data.index() == 0) {
      std::vector<double> contiguous_tensor_data{};
      fill_and_write_contiguous_tensor_data(
          make_not_null(&contiguous_tensor_data));
    } else if (elements[0].tensor_components[i].data.index() == 1) {
      std::vector<float> contiguous_tensor_data{};
      fill_and_write_contiguous_tensor_data(
          make_not_null(&contiguous_tensor_data));
    } else {
      ERROR("Unknown index value ("
            << elements[0].tensor_components[i].data.index()
            << ") in std::variant of tensor component.");
    }
  }  // for each component
  grid_names.pop_back();

  // Write the grid extents contiguously, the first `dim` belong to the
  // First grid, the second `dim` belong to the second grid, and so on,
  // Ordering is `x, y, z, ... `
  h5::write_data(observation_group.id(), total_extents, {total_extents.size()},
                 "total_extents");
  // Write the names of the grids as vector of chars with individual names
  // separated by `separator()`
  std::vector<char> grid_names_as_chars(grid_names.begin(), grid_names.end());
  h5::write_data(observation_group.id(), grid_names_as_chars,
                 {grid_names_as_chars.size()}, "grid_names");
  // Write the coded quadrature, along with the dictionary
  const auto io_quadratures = h5_detail::allowed_quadratures();
  std::vector<std::string> quadrature_dict(io_quadratures.size());
  alg::transform(io_quadratures, quadrature_dict.begin(),
                 get_output<SpatialDiscretization::Quadrature>);
  h5_detail::write_dictionary("Quadrature dictionary", quadrature_dict,
                              observation_group);
  h5::write_data(observation_group.id(), quadratures, {quadratures.size()},
                 "quadratures");
  // Write the coded basis, along with the dictionary
  const auto io_bases = h5_detail::allowed_bases();
  std::vector<std::string> basis_dict(io_bases.size());
  alg::transform(io_bases, basis_dict.begin(),
                 get_output<SpatialDiscretization::Basis>);
  h5_detail::write_dictionary("Basis dictionary", basis_dict,
                              observation_group);
  h5::write_data(observation_group.id(), bases, {bases.size()}, "bases");
  // Write the Connectivity
  h5::write_data(observation_group.id(), total_connectivity,
                 {total_connectivity.size()}, "connectivity");
  // Note: pole_connectivity stores extra connections that define triangles to
  // fill in the poles on a Strahlkorper and is empty if not outputting
  // Strahlkorper surface data. Because these connections define triangles
  // and not quadrilaterals, they are stored separately instead of just being
  // included in total_connectivity.
  if (not pole_connectivity.empty()) {
    h5::write_data(observation_group.id(), pole_connectivity,
                   {pole_connectivity.size()}, "pole_connectivity");
  }
  // Write the serialized domain
  if (serialized_domain.has_value()) {
    h5::write_data(observation_group.id(), *serialized_domain,
                   {serialized_domain->size()}, "domain");
  }
  // Write the serialized functions of time
  if (serialized_functions_of_time.has_value()) {
    h5::write_data(observation_group.id(), *serialized_functions_of_time,
                   {serialized_functions_of_time->size()}, "functions_of_time");
  }
}

// Write new connectivity connections given a std::vector of observation ids
template <size_t SpatialDim>
void VolumeData::extend_connectivity_data(
    const std::vector<size_t>& observation_ids) {
  for (const size_t& obs_id : observation_ids) {
    auto grid_names = get_grid_names(obs_id);
    auto extents = get_extents(obs_id);
    auto bases = get_bases(obs_id);
    auto quadratures = get_quadratures(obs_id);

    const std::vector<int>& new_connectivity =
        h5::detail::extend_connectivity<SpatialDim>(grid_names, bases,
                                                    quadratures, extents);

    // Deletes the existing connectivity and replaces it with the new one
    const std::string path = "ObservationId" + std::to_string(obs_id);
    detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                        AccessType::ReadWrite);
    const hid_t group_id = observation_group.id();
    delete_connectivity(group_id);
    write_connectivity(group_id, new_connectivity);
  }
}

void VolumeData::write_tensor_component(
    const size_t observation_id, const std::string& component_name,
    const DataVector& contiguous_tensor_data) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  h5::write_data(observation_group.id(), contiguous_tensor_data,
                 component_name);
}

void VolumeData::write_tensor_component(
    const size_t observation_id, const std::string& component_name,
    const std::vector<float>& contiguous_tensor_data) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  h5::write_data(observation_group.id(), contiguous_tensor_data,
                 {contiguous_tensor_data.size()}, component_name);
}

std::vector<size_t> VolumeData::list_observation_ids() const {
  const auto names = get_group_names(volume_data_group_.id(), "");
  const auto helper = [](const std::string& s) {
    return std::stoul(s.substr(std::string("ObservationId").size()));
  };
  std::vector<size_t> obs_ids{
      boost::make_transform_iterator(names.begin(), helper),
      boost::make_transform_iterator(names.end(), helper)};
  alg::sort(obs_ids, [this](const size_t lhs, const size_t rhs) {
    return this->get_observation_value(lhs) < this->get_observation_value(rhs);
  });
  return obs_ids;
}

double VolumeData::get_observation_value(const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  return h5::read_value_attribute<double>(observation_group.id(),
                                          "observation_value");
}

std::vector<std::string> VolumeData::list_tensor_components(
    const size_t observation_id) const {
  auto tensor_components =
      get_group_names(volume_data_group_.id(),
                      "ObservationId" + std::to_string(observation_id));
  // Remove names that are not tensor components
  const std::unordered_set<std::string> non_tensor_components{
      "connectivity", "pole_connectivity", "total_extents",
      "grid_names",   "quadratures",       "bases",
      "domain",       "functions_of_time"};
  tensor_components.erase(
      alg::remove_if(tensor_components,
                     [&non_tensor_components](const std::string& name) {
                       return non_tensor_components.find(name) !=
                              non_tensor_components.end();
                     }),
      tensor_components.end());
  return tensor_components;
}

std::vector<std::string> VolumeData::get_grid_names(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const std::vector<char> names =
      h5::read_data<1, std::vector<char>>(observation_group.id(), "grid_names");
  const std::string all_names(names.begin(), names.end());
  std::vector<std::string> grid_names{};
  boost::split(grid_names, all_names,
               [](const char c) { return c == h5::VolumeData::separator(); });
  return grid_names;
}

TensorComponent VolumeData::get_tensor_component(
    const size_t observation_id, const std::string& tensor_component) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);

  const hid_t dataset_id =
      h5::open_dataset(observation_group.id(), tensor_component);
  const hid_t dataspace_id = h5::open_dataspace(dataset_id);
  const auto rank =
      static_cast<size_t>(H5Sget_simple_extent_ndims(dataspace_id));
  h5::close_dataspace(dataspace_id);
  const bool use_float =
      h5::types_equal(H5Dget_type(dataset_id), h5::h5_type<float>());
  h5::close_dataset(dataset_id);

  const auto get_data = [&observation_group, &rank,
                         &tensor_component](auto type_to_get_v) {
    using type_to_get = tmpl::type_from<decltype(type_to_get_v)>;
    switch (rank) {
      case 1:
        return h5::read_data<1, type_to_get>(observation_group.id(),
                                             tensor_component);
      case 2:
        return h5::read_data<2, type_to_get>(observation_group.id(),
                                             tensor_component);
      case 3:
        return h5::read_data<3, type_to_get>(observation_group.id(),
                                             tensor_component);
      default:
        ERROR("Rank must be 1, 2, or 3. Received data with Rank = " << rank);
    }
  };

  if (use_float) {
    return {tensor_component, get_data(tmpl::type_<std::vector<float>>{})};
  } else {
    return {tensor_component, get_data(tmpl::type_<DataVector>{})};
  }
}

std::vector<std::vector<size_t>> VolumeData::get_extents(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto extents_per_element = static_cast<long>(dim);
  const auto total_extents = h5::read_data<1, std::vector<size_t>>(
      observation_group.id(), "total_extents");
  std::vector<std::vector<size_t>> individual_extents;
  individual_extents.reserve(total_extents.size() / dim);
  for (auto iter = total_extents.begin(); iter != total_extents.end();
       iter += extents_per_element) {
    individual_extents.emplace_back(iter, iter + extents_per_element);
  }
  return individual_extents;
}

std::pair<size_t, size_t> offset_and_length_for_grid(
    const std::string& grid_name,
    const std::vector<std::string>& all_grid_names,
    const std::vector<std::vector<size_t>>& all_extents) {
  auto found_grid_name = alg::find(all_grid_names, grid_name);
  if (found_grid_name == all_grid_names.end()) {
    ERROR("Found no grid named '" + grid_name + "'.");
  } else {
    const auto element_index =
        std::distance(all_grid_names.begin(), found_grid_name);
    const size_t element_data_offset = std::accumulate(
        all_extents.begin(), all_extents.begin() + element_index, 0_st,
        [](const size_t offset, const std::vector<size_t>& extents) {
          return offset + alg::accumulate(extents, 1_st, std::multiplies<>{});
        });
    const size_t element_data_length = alg::accumulate(
        gsl::at(all_extents, element_index), 1_st, std::multiplies<>{});
    return {element_data_offset, element_data_length};
  }
}

auto VolumeData::get_data_by_element(
    const std::optional<double> start_observation_value,
    const std::optional<double> end_observation_value,
    const std::optional<std::vector<std::string>>& components_to_retrieve) const
    -> std::vector<std::tuple<size_t, double, std::vector<ElementVolumeData>>> {
  // First get list of all observations we need to retrieve
  const auto names = get_group_names(volume_data_group_.id(), "");
  const auto get_observation_id_from_group_name = [](const std::string& s) {
    return std::stoul(s.substr(std::string("ObservationId").size()));
  };
  std::vector<size_t> obs_ids{
      boost::make_transform_iterator(names.begin(),
                                     get_observation_id_from_group_name),
      boost::make_transform_iterator(names.end(),
                                     get_observation_id_from_group_name)};
  std::vector<std::tuple<size_t, double, std::vector<ElementVolumeData>>>
      result{};
  result.reserve(obs_ids.size());
  // Sort observation IDs and observation values into the result. This only
  // copies observed times in
  // [`start_observation_value`, `end_observation_value`]
  for (const auto& observation_id : obs_ids) {
    const double observation_value = get_observation_value(observation_id);
    if (start_observation_value.value_or(
            std::numeric_limits<double>::lowest()) <= observation_value and
        observation_value <= end_observation_value.value_or(
                                 std::numeric_limits<double>::max())) {
      result.emplace_back(observation_id, observation_value,
                          std::vector<ElementVolumeData>{});
    }
  }
  result.shrink_to_fit();
  // Sort by observation_value
  alg::sort(result, [](const auto& lhs, const auto& rhs) {
    return std::get<1>(lhs) < std::get<1>(rhs);
  });

  // Retrieve element data and insert into result
  for (auto& single_time_data : result) {
    const auto known_components =
        list_tensor_components(std::get<0>(single_time_data));

    std::vector<ElementVolumeData> element_volume_data{};
    const auto grid_names = get_grid_names(std::get<0>(single_time_data));
    const auto extents = get_extents(std::get<0>(single_time_data));
    const auto bases = get_bases(std::get<0>(single_time_data));
    const auto quadratures = get_quadratures(std::get<0>(single_time_data));
    element_volume_data.reserve(grid_names.size());

    const auto& component_names =
        components_to_retrieve.value_or(known_components);
    std::vector<TensorComponent> tensors{};
    tensors.reserve(grid_names.size());
    for (const std::string& component : component_names) {
      if (not alg::found(known_components, component)) {
        using ::operator<<;  // STL streams
        ERROR("Could not find tensor component '"
              << component
              << "' in file. Known components are: " << known_components);
      }
      tensors.emplace_back(
          get_tensor_component(std::get<0>(single_time_data), component));
    }
    // Now split the data by element
    for (size_t grid_index = 0, offset = 0; grid_index < grid_names.size();
         ++grid_index) {
      const size_t mesh_size =
          alg::accumulate(extents[grid_index], 1_st, std::multiplies<>{});
      std::vector<TensorComponent> tensor_components{tensors.size()};
      for (size_t component_index = 0; component_index < tensors.size();
           ++component_index) {
        std::visit(
            [component_index, &component_names, mesh_size, offset,
             &tensor_components](const auto& tensor_component_data) {
              std::decay_t<decltype(tensor_component_data)> component(
                  mesh_size);
              std::copy(
                  std::next(tensor_component_data.begin(),
                            static_cast<std::ptrdiff_t>(offset)),
                  std::next(tensor_component_data.begin(),
                            static_cast<std::ptrdiff_t>(offset + mesh_size)),
                  component.begin());
              tensor_components[component_index] = TensorComponent{
                  component_names[component_index], std::move(component)};
            },
            tensors[component_index].data);
      }

      // Sort the tensor components by name so that they are in the same order
      // in all elements.
      alg::sort(tensor_components, [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
      });

      element_volume_data.emplace_back(
          grid_names[grid_index], std::move(tensor_components),
          extents[grid_index], bases[grid_index], quadratures[grid_index]);
      offset += mesh_size;
    }  // for grid_index

    // Sort the elements so they are in the same order at all time steps
    alg::sort(element_volume_data,
              [](const ElementVolumeData& lhs, const ElementVolumeData& rhs) {
                return lhs.element_name < rhs.element_name;
              });
    std::get<2>(single_time_data) = std::move(element_volume_data);
  }
  return result;
}

size_t VolumeData::get_dimension() const {
  return h5::read_value_attribute<double>(volume_data_group_.id(), "dimension");
}

std::vector<std::vector<SpatialDiscretization::Basis>> VolumeData::get_bases(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto bases_per_element = static_cast<long>(dim);

  const std::vector<int> bases_coded =
      h5::read_data<1, std::vector<int>>(observation_group.id(), "bases");
  const auto all_bases = h5_detail::decode_with_dictionary_name(
      "Basis dictionary", bases_coded, observation_group);

  std::vector<std::vector<SpatialDiscretization::Basis>> element_bases;

  const auto to_basis = [](const std::string& input_basis) {
    for (const auto basis : h5_detail::allowed_bases()) {
      if (input_basis == get_output(basis)) {
        return basis;
      }
    }
    using ::operator<<;
    ERROR("Failed to convert \""
          << input_basis
          << "\" to SpatialDiscretization::Basis.\nMust be one of "
          << h5_detail::allowed_bases() << ".");
  };

  for (auto iter = all_bases.begin(); iter != all_bases.end();
       std::advance(iter, bases_per_element)) {
    element_bases.emplace_back(
        boost::make_transform_iterator(iter, to_basis),
        boost::make_transform_iterator(std::next(iter, bases_per_element),
                                       to_basis));
  }
  return element_bases;
}
std::vector<std::vector<SpatialDiscretization::Quadrature>>
VolumeData::get_quadratures(const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto quadratures_per_element = static_cast<long>(dim);
  const std::vector<int> quadratures_coded =
      h5::read_data<1, std::vector<int>>(observation_group.id(), "quadratures");
  const auto all_quadratures = h5_detail::decode_with_dictionary_name(
      "Quadrature dictionary", quadratures_coded, observation_group);
  std::vector<std::vector<SpatialDiscretization::Quadrature>>
      element_quadratures;

  const auto to_quadrature = [](const std::string& input_quadrature) {
    for (const auto quadrature : h5_detail::allowed_quadratures()) {
      if (input_quadrature == get_output(quadrature)) {
        return quadrature;
      }
    }
    using ::operator<<;
    ERROR("Failed to convert \""
          << input_quadrature
          << "\" to SpatialDiscretization::Quadrature.\nMust be one of "
          << h5_detail::allowed_quadratures() << ".");
  };

  for (auto iter = all_quadratures.begin(); iter != all_quadratures.end();
       std::advance(iter, quadratures_per_element)) {
    element_quadratures.emplace_back(
        boost::make_transform_iterator(iter, to_quadrature),
        boost::make_transform_iterator(std::next(iter, quadratures_per_element),
                                       to_quadrature));
  }
  return element_quadratures;
}

std::optional<std::vector<char>> VolumeData::get_domain(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  if (not contains_dataset_or_group(observation_group.id(), "", "domain")) {
    return std::nullopt;
  }
  return h5::read_data<1, std::vector<char>>(observation_group.id(), "domain");
}

std::optional<std::vector<char>> VolumeData::get_functions_of_time(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  if (not contains_dataset_or_group(observation_group.id(), "",
                                    "functions_of_time")) {
    return std::nullopt;
  }
  return h5::read_data<1, std::vector<char>>(observation_group.id(),
                                             "functions_of_time");
}

template <size_t Dim>
Mesh<Dim> mesh_for_grid(
    const std::string& grid_name,
    const std::vector<std::string>& all_grid_names,
    const std::vector<std::vector<size_t>>& all_extents,
    const std::vector<std::vector<SpatialDiscretization::Basis>>& all_bases,
    const std::vector<std::vector<SpatialDiscretization::Quadrature>>&
        all_quadratures) {
  const auto found_grid_name = alg::find(all_grid_names, grid_name);
  if (found_grid_name == all_grid_names.end()) {
    ERROR("Found no grid named '" + grid_name + "'.");
  } else {
    const auto element_index =
        std::distance(all_grid_names.begin(), found_grid_name);
    const auto& extents = gsl::at(all_extents, element_index);
    const auto& bases = gsl::at(all_bases, element_index);
    const auto& quadratures = gsl::at(all_quadratures, element_index);
    ASSERT(extents.size() == Dim, "Extents in " << Dim << "D should have size "
                                                << Dim << ", but found size "
                                                << extents.size() << ".");
    ASSERT(bases.size() == Dim, "Bases in " << Dim << "D should have size "
                                            << Dim << ", but found size "
                                            << bases.size() << ".");
    ASSERT(quadratures.size() == Dim, "Quadratures in "
                                          << Dim << "D should have size " << Dim
                                          << ", but found size "
                                          << quadratures.size() << ".");
    return Mesh<Dim>{
        make_array<size_t, Dim>(extents),
        make_array<SpatialDiscretization::Basis, Dim>(bases),
        make_array<SpatialDiscretization::Quadrature, Dim>(quadratures)};
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void h5::VolumeData::extend_connectivity_data<DIM(data)>(           \
      const std::vector<size_t>& observation_ids);                             \
  template Mesh<DIM(data)> mesh_for_grid(                                      \
      const std::string& grid_name,                                            \
      const std::vector<std::string>& all_grid_names,                          \
      const std::vector<std::vector<size_t>>& all_extents,                     \
      const std::vector<std::vector<SpatialDiscretization::Basis>>& all_bases, \
      const std::vector<std::vector<SpatialDiscretization::Quadrature>>&       \
          all_quadratures);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace h5
