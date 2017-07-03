// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Helpers.hpp"

#include <cstring>
#include <regex>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Type.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace h5 {
void write_time(const hid_t group_id, const double time) {
  // Write the time as an attribute to the group
  hid_t s_id = H5Screate(H5S_SCALAR);
  CHECK_H5(s_id, "Failed to create scalar for time");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  hid_t att_id = H5Acreate2(group_id, "Time", H5T_NATIVE_DOUBLE, s_id,
                            H5P_DEFAULT, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(att_id, "Failed to create attribute for time");
  CHECK_H5(H5Awrite(att_id, H5T_NATIVE_DOUBLE, static_cast<const void*>(&time)),
           "Failed to write time");
  CHECK_H5(H5Sclose(s_id), "Failed to close space_id");
  CHECK_H5(H5Aclose(att_id), "Failed to close attribute for time");
}

template <size_t Dim>
void write_data(const hid_t group_id, const DataVector& data,
                const Index<Dim>& extents, const std::string& name) {
  // Write a DataVector into the group
  const std::array<hsize_t, Dim> dims = make_array<hsize_t, Dim>(extents);
  const hid_t space_id = H5Screate_simple(Dim, dims.data(), nullptr);
  CHECK_H5(space_id, "Failed to create dataspace");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t contained_type = h5::h5_type<std::decay_t<decltype(data[0])>>();
  const hid_t dataset_id =
      H5Dcreate2(group_id, name.c_str(), contained_type, space_id, H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT);
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(H5Dwrite(dataset_id, contained_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    static_cast<const void*>(data.data())),
           "Failed to write data to dataset");
#pragma GCC diagnostic pop
  CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
  CHECK_H5(H5Dclose(dataset_id), "Failed to close dataset");
}

template <size_t Dim>
void write_extents(const hid_t group_id, const Index<Dim>& extents,
                   const std::string& name) {
  // Write the current extents as an attribute to the group
  const hsize_t size = Dim;
  const hid_t space_id = H5Screate_simple(1, &size, nullptr);
  CHECK_H5(space_id, "Failed to create dataspace");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t att_id = H5Acreate2(
      group_id, name.c_str(), h5::h5_type<std::decay_t<decltype(extents[0])>>(),
      space_id, H5P_DEFAULT, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(att_id, "Failed to create attribute");
  CHECK_H5(H5Awrite(att_id, h5::h5_type<std::decay_t<decltype(extents[0])>>(),
                    static_cast<const void*>(extents.data())),
           "Failed to write extents");
  CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
  CHECK_H5(H5Aclose(att_id), "Failed to close attribute");
}

void write_connectivity(const hid_t group_id,
                        const std::vector<int>& connectivity) {
  const hsize_t size = connectivity.size();
  const hid_t space_id = H5Screate_simple(1, &size, nullptr);
  CHECK_H5(space_id, "Failed to create dataspace");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t dataset_id =
      H5Dcreate2(group_id, "connectivity", H5T_NATIVE_INT, space_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    static_cast<const void*>(connectivity.data())),
           "Failed to write connectivity");
#pragma GCC diagnostic pop
  CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
  CHECK_H5(H5Dclose(dataset_id), "Failed to close dataset");
}

std::vector<std::string> get_group_names(const hid_t file_id,
                                         const std::string& group_name) {
  // Opens the group, loads the group info and then loops over all the groups
  // retrieving their names and storing them in names
  detail::OpenGroup my_group(file_id, group_name, AccessType::ReadOnly);
  const hid_t group_id = my_group.id();
  H5G_info_t group_info{};
  std::string name;
  std::vector<std::string> names;
  CHECK_H5(H5Gget_info(group_id, &group_info), "Failed to get group info");
  names.reserve(group_info.nlinks);
  for (size_t i = 0; i < group_info.nlinks; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    const hsize_t size =
        static_cast<hsize_t>(1) + static_cast<hsize_t>(H5Lget_name_by_idx(
                                      group_id, ".", H5_INDEX_NAME, H5_ITER_INC,
                                      i, nullptr, 0, H5P_DEFAULT));
    name.resize(size);
    H5Lget_name_by_idx(group_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, &name[0],
                       size, H5P_DEFAULT);
#pragma GCC diagnostic pop
    // We need to remove the bloody trailing \0...
    name.pop_back();
    names.push_back(name);
  }
  return names;
}

std::vector<std::string> get_attribute_names(const hid_t file_id,
                                             const std::string& group_name) {
  // Opens the group, loads the group info and then loops over all the
  // attributes retrieving their names and storing them in names
  detail::OpenGroup my_group(file_id, group_name, AccessType::ReadOnly);
  const hid_t group_id = my_group.id();
  H5O_info_t group_info{};
  std::string name;
  std::vector<std::string> names;
  CHECK_H5(H5Oget_info(group_id, &group_info), "Failed to get group info");
  names.reserve(group_info.num_attrs);
  for (size_t i = 0; i < group_info.num_attrs; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    const hsize_t size =
        static_cast<hsize_t>(1) + static_cast<hsize_t>(H5Aget_name_by_idx(
                                      group_id, ".", H5_INDEX_NAME, H5_ITER_INC,
                                      i, nullptr, 0, H5P_DEFAULT));
    name.resize(size);
    H5Aget_name_by_idx(group_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, &name[0],
                       size, H5P_DEFAULT);
#pragma GCC diagnostic pop
    // We need to remove the bloody trailing \0...
    name.pop_back();
    names.push_back(name);
  }
  return names;
}

bool contains_attribute(const hid_t file_id, const std::string& group_name,
                        const std::string& attribute_name) {
  const std::vector<std::string> names(
      get_attribute_names(file_id, group_name));
  return std::find(std::begin(names), std::end(names), attribute_name) !=
         std::end(names);
}

double get_time(const hid_t file_id, const std::string& group_name,
                const std::string& attr_name) {
  if (not contains_attribute(file_id, group_name, attr_name)) {
    ERROR("group " << group_name << " does not contain attribute '" << attr_name
                   << "'");
  }
  detail::OpenGroup my_group(file_id, group_name, AccessType::ReadOnly);
  const hid_t group_id = my_group.id();
// Read the attr_name attribute and close the attribute.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t attr_id = H5Aopen(group_id, attr_name.c_str(), H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(attr_id, "Failed to open attribute");
  double time;
  CHECK_H5(H5Aread(attr_id, H5T_NATIVE_DOUBLE, &time),
           "Failed to read attribute");
  H5Aclose(attr_id);
  return time;
}

DataVector read_data(const hid_t group_id, const std::string& dataset_name) {
// Read a DataVector from the group
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t dataset_id =
      H5Dopen2(group_id, dataset_name.c_str(), H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(dataset_id, "could not open dataset '" << dataset_name << "'");
  // Get the number of points. These do not need a "close" call.
  const hid_t space_id = H5Dget_space(dataset_id);
  CHECK_H5(space_id, "Failed to open dataspace");
  const hid_t number_of_points = H5Sget_simple_extent_npoints(space_id);
  CHECK_H5(number_of_points, "Failed to get number of points");
  H5Sclose(space_id);
  // Load the data
  DataVector data(static_cast<size_t>(number_of_points));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CHECK_H5(H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   static_cast<void*>(data.data())),
           "Failed to read data");
#pragma GCC diagnostic pop
  CHECK_H5(H5Dclose(dataset_id), "Failed to close dataset");
  return data;
}

template <size_t Dim>
Index<Dim> read_extents(const hid_t group_id, const std::string& extents_name) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t attr_id = H5Aopen(group_id, extents_name.c_str(), H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(attr_id, "Failed to open attribute");
  Index<Dim> extents;
  CHECK_H5(H5Aread(attr_id, h5::h5_type<std::decay_t<decltype(extents[0])>>(),
                   extents.data()),
           "Failed to read extents");
  H5Aclose(attr_id);
  return extents;
}

// Explicit instantiations
template void write_data<1>(const hid_t group_id, const DataVector& data,
                            const Index<1>& extents, const std::string& name);
template void write_data<2>(const hid_t group_id, const DataVector& data,
                            const Index<2>& extents, const std::string& name);
template void write_data<3>(const hid_t group_id, const DataVector& data,
                            const Index<3>& extents, const std::string& name);

template void write_extents<1>(const hid_t group_id, const Index<1>& extents,
                               const std::string& name);
template void write_extents<2>(const hid_t group_id, const Index<2>& extents,
                               const std::string& name);
template void write_extents<3>(const hid_t group_id, const Index<3>& extents,
                               const std::string& name);

template Index<1> read_extents<1>(const hid_t group_id,
                                  const std::string& extents_name);
template Index<2> read_extents<2>(const hid_t group_id,
                                  const std::string& extents_name);
template Index<3> read_extents<3>(const hid_t group_id,
                                  const std::string& extents_name);
}  // namespace h5

namespace h5 {
namespace detail {
void write_strings_to_attribute(const hid_t dataset_id, const std::string& name,
                                const std::vector<std::string>& string_array) {
  // See the HDF5 example:
  // https://support.hdfgroup.org/ftp/HDF5/examples/examples-by-api/
  // hdf5-examples/1_8/C/H5T/h5ex_t_stringatt.c

  const hid_t type_id = H5Tcopy(H5T_FORTRAN_S1);
  CHECK_H5(type_id, "Failed to create type_id");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CHECK_H5(H5Tset_size(type_id, H5T_VARIABLE), "Failed to set size");
#pragma GCC diagnostic pop

  // Create dataspace and attribute in dataspace where we will store the strings
  const hsize_t dim = string_array.size();
  const hid_t space_id = H5Screate_simple(1, &dim, nullptr);
  CHECK_H5(space_id, "Failed to create null space");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t attr_id = H5Acreate2(dataset_id, name.c_str(), type_id, space_id,
                                   H5P_DEFAULT, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(attr_id, "Could not create attribute");

  // We are using C-style strings, which is type to be written into attribute
  const auto memtype_id = H5Tcopy(H5T_C_S1);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CHECK_H5(H5Tset_size(memtype_id, H5T_VARIABLE), "Failed to set size");
#pragma GCC diagnostic pop

  // In order to write strings to an attribute we must have a pointer to
  // pointers, so we use a vector.
  std::vector<const char*> string_pointers(string_array.size());
  std::transform(string_array.begin(), string_array.end(),
                 string_pointers.begin(),
                 [](const auto& t) { return t.c_str(); });
  CHECK_H5(H5Awrite(attr_id, memtype_id, string_pointers.data()),
           "Failed attribute write");

  CHECK_H5(H5Aclose(attr_id), "Failed to close attribute");
  CHECK_H5(H5Sclose(space_id), "Failed to close space_id");
  CHECK_H5(H5Tclose(memtype_id), "Failed to close memtype_id");
  CHECK_H5(H5Tclose(type_id), "Failed to close type_id");
}

std::vector<std::string> read_strings_from_attribute(const hid_t group_id,
                                                     const std::string& name) {
  const auto attribute_exists =
      static_cast<bool>(H5Aexists(group_id, name.c_str()));
  if (not attribute_exists) {
    ERROR("Could not find attribute '" << name << "'");  // LCOV_EXCL_LINE
  }

// Open attribute that holds the strings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t attribute_id = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(attribute_id, "Failed to open attribute: '" << name << "'");
  const hid_t dataspace_id = H5Aget_space(attribute_id);
  CHECK_H5(dataspace_id,
           "Failed to open dataspace for attribute '" << name << "'");
  // Get the size of the strings
  hsize_t legend_dims[1];
  CHECK_H5(H5Sget_simple_extent_dims(dataspace_id, legend_dims, nullptr),
           "Failed to get size of strings");
  // Read the strings as arrays of characters
  std::vector<char*> temp(legend_dims[0]);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t memtype = H5Tcopy(H5T_C_S1);
  CHECK_H5(H5Tset_size(memtype, H5T_VARIABLE), "Failed to set size");
#pragma GCC diagnostic pop
  CHECK_H5(H5Aread(attribute_id, memtype, static_cast<void*>(temp.data())),
           "Failed to read attribute");

  std::vector<std::string> result(temp.size());
  std::transform(temp.begin(), temp.end(), result.begin(),
                 [](const auto& t) { return std::string(t); });

// Clean up memory from variable length arrays and close everything
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CHECK_H5(H5Dvlen_reclaim(memtype, dataspace_id, H5P_DEFAULT, temp.data()),
           "Failed H5Dvlen_reclaim at ");
#pragma GCC diagnostic pop
  CHECK_H5(H5Aclose(attribute_id), "Failed to close attribute");
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close space_id");
  CHECK_H5(H5Tclose(memtype), "Failed to close memtype");
  return result;
}

template <typename Type>
void write_value_to_attribute(const hid_t location_id, const std::string& name,
                              const Type& value) {
  const hid_t space_id = H5Screate(H5S_SCALAR);
  CHECK_H5(space_id, "Failed to create scalar");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t att_id = H5Acreate2(location_id, name.c_str(), h5_type<Type>(),
                                  space_id, H5P_DEFAULT, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(att_id, "Failed to create attribute '" << name << "'");
  CHECK_H5(H5Awrite(att_id, h5_type<Type>(), static_cast<const void*>(&value)),
           "Failed to write value: " << value);
  CHECK_H5(H5Aclose(att_id), "Failed to close attribute '" << name << "'");
  CHECK_H5(H5Sclose(space_id), "Unable to close dataspace");
}

template <typename Type>
Type read_value_from_attribute(const hid_t location_id,
                               const std::string& name) {
  const htri_t attribute_exists = H5Aexists(location_id, name.c_str());
  if (not attribute_exists) {
    ERROR("Could not find attribute '" << name << "'");  // LCOV_EXCL_LINE
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t attribute_id = H5Aopen(location_id, name.c_str(), H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(attribute_id, "Failed to open attribute '" << name << "'");
  Type value;
  CHECK_H5(H5Aread(attribute_id, h5_type<Type>(), &value),
           "Failed to read attribute '" << name << "'");
  CHECK_H5(H5Aclose(attribute_id),
           "Failed to close attribute '" << name << "'");
  return value;
}

template <size_t Dims>
hid_t create_extensible_dataset(const hid_t group_id, const std::string& name,
                                const std::array<hsize_t, Dims>& initial_size,
                                const std::array<hsize_t, Dims>& chunk_size,
                                const std::array<hsize_t, Dims>& max_size) {
  const hid_t dataspace_id =
      H5Screate_simple(Dims, initial_size.data(), max_size.data());
  CHECK_H5(dataspace_id, "Failed to create extensible dataspace");

  const auto property_list = H5Pcreate(H5P_DATASET_CREATE);
  CHECK_H5(property_list, "Failed to create property list");
  CHECK_H5(H5Pset_chunk(property_list, Dims, chunk_size.data()),
           "Failed to set chunk size");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t dataset_id =
      H5Dcreate2(group_id, name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                 H5P_DEFAULT, property_list, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(H5Pclose(property_list), "Failed to close property list");
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");
  return dataset_id;
}

template void write_value_to_attribute<double>(const hid_t location_id,
                                               const std::string& name,
                                               const double& value);
template void write_value_to_attribute<uint32_t>(const hid_t location_id,
                                                 const std::string& name,
                                                 const uint32_t& value);
template void write_value_to_attribute<int>(const hid_t location_id,
                                            const std::string& name,
                                            const int& value);

template double read_value_from_attribute<double>(const hid_t location_id,
                                                  const std::string& name);
template uint32_t read_value_from_attribute<uint32_t>(const hid_t location_id,
                                                      const std::string& name);
template int read_value_from_attribute<int>(const hid_t location_id,
                                            const std::string& name);

template hid_t create_extensible_dataset<1>(
    const hid_t group_id, const std::string& name,
    const std::array<hsize_t, 1>& initial_size,
    const std::array<hsize_t, 1>& chunk_size,
    const std::array<hsize_t, 1>& max_size);
template hid_t create_extensible_dataset<2>(
    const hid_t group_id, const std::string& name,
    const std::array<hsize_t, 2>& initial_size,
    const std::array<hsize_t, 2>& chunk_size,
    const std::array<hsize_t, 2>& max_size);
template hid_t create_extensible_dataset<3>(
    const hid_t group_id, const std::string& name,
    const std::array<hsize_t, 3>& initial_size,
    const std::array<hsize_t, 3>& chunk_size,
    const std::array<hsize_t, 3>& max_size);
}  // namespace detail
}  // namespace h5
