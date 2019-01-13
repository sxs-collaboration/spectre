// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Helpers.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <ostream>
#include <string>

#include "DataStructures/BoostMultiArray.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/StaticAssert.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp" // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_include <boost/multi_array.hpp>
// IWYU pragma: no_include <boost/multi_array/base.hpp>
// IWYU pragma: no_include <boost/multi_array/extent_gen.hpp>

namespace {
// Converts input data (either a std::vector or DataVector) to a T. Depending
// on the rank of the input data, the following outputs T are allowed:
// rank 0: double or int
// rank 1: DataVector or std::vector
// rank 2,3: DataVector or boost::multiarray
template <size_t Rank, typename T>
struct VectorTo;

template <typename T>
struct VectorTo<0, T> {
  static T apply(std::vector<T> raw_data,
                 const std::array<hsize_t, 0>& /*size*/) noexcept {
    return raw_data[0];
  }
};

template <typename T>
struct VectorTo<1, T> {
  static T apply(T raw_data, const std::array<hsize_t, 1>& /*size*/) noexcept {
    return raw_data;
  }
};

template <>
struct VectorTo<2, DataVector> {
  static DataVector apply(DataVector raw_data,
                          const std::array<hsize_t, 2>& /*size*/) noexcept {
    return raw_data;
  }
};

template <typename T>
struct VectorTo<2, boost::multi_array<T, 2>> {
  static boost::multi_array<T, 2> apply(
      const std::vector<T>& raw_data,
      const std::array<hsize_t, 2>& size) noexcept {
    DEBUG_STATIC_ASSERT(cpp17::is_fundamental_v<T>,
                        "VectorTo is optimized for fundamentals. Need to "
                        "use move semantics for handling generic data types.");
    boost::multi_array<T, 2> temp(boost::extents[size[0]][size[1]]);
    for (size_t i = 0; i < size[0]; ++i) {
      for (size_t j = 0; j < size[1]; ++j) {
        temp[i][j] = raw_data[j + i * size[1]];
      }
    }
    return temp;
  }
};

template <>
struct VectorTo<3, DataVector> {
  static DataVector apply(DataVector raw_data,
                          const std::array<hsize_t, 3>& /*size*/) noexcept {
    return raw_data;
  }
};

template <typename T>
struct VectorTo<3, boost::multi_array<T, 3>> {
  static boost::multi_array<T, 3> apply(
      const std::vector<T>& raw_data,
      const std::array<hsize_t, 3>& size) noexcept {
    DEBUG_STATIC_ASSERT(cpp17::is_fundamental_v<T>,
                        "VectorTo is optimized for fundamentals. Need to "
                        "use move semantics for handling generic data types.");
    boost::multi_array<T, 3> temp(boost::extents[size[0]][size[1]][size[2]]);
    for (size_t i = 0; i < size[0]; ++i) {
      for (size_t j = 0; j < size[1]; ++j) {
        for (size_t k = 0; k < size[2]; ++k) {
          temp[i][j][k] = raw_data[k + j * size[2] + i * size[2] * size[1]];
        }
      }
    }
    return temp;
  }
};
}  // namespace

namespace h5 {
template <typename T>
void write_data(const hid_t group_id, const std::vector<T>& data,
                const std::vector<size_t>& extents,
                const std::string& name) noexcept {
  const std::vector<hsize_t> dims(extents.begin(), extents.end());
  const hid_t space_id = H5Screate_simple(dims.size(), dims.data(), nullptr);
  CHECK_H5(space_id, "Failed to create dataspace");
  const hid_t contained_type = h5::h5_type<tt::get_fundamental_type_t<T>>();
  const hid_t dataset_id =
      H5Dcreate2(group_id, name.c_str(), contained_type, space_id,
                 h5::h5p_default(), h5::h5p_default(), h5::h5p_default());
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(H5Dwrite(dataset_id, contained_type, h5::h5s_all(), h5::h5s_all(),
                    h5::h5p_default(), static_cast<const void*>(data.data())),
           "Failed to write data to dataset");
  CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
  CHECK_H5(H5Dclose(dataset_id), "Failed to close dataset");
}

void write_data(const hid_t group_id, const DataVector& data,
                const std::string& name) noexcept {
  const auto number_of_points = static_cast<hsize_t>(data.size());
  const hid_t space_id = H5Screate_simple(1, &number_of_points, nullptr);
  CHECK_H5(space_id, "Failed to create dataspace");
  const hid_t contained_type = h5::h5_type<double>();
  const hid_t dataset_id =
      H5Dcreate2(group_id, name.c_str(), contained_type, space_id,
                 h5::h5p_default(), h5::h5p_default(), h5::h5p_default());
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(H5Dwrite(dataset_id, contained_type, h5::h5s_all(), h5::h5s_all(),
                    h5::h5p_default(), static_cast<const void*>(data.data())),
           "Failed to write data to dataset");
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
  const hid_t att_id = H5Acreate2(
      group_id, name.c_str(), h5::h5_type<std::decay_t<decltype(extents[0])>>(),
      space_id, h5p_default(), h5p_default());
  CHECK_H5(att_id, "Failed to create attribute");
  CHECK_H5(H5Awrite(att_id, h5::h5_type<std::decay_t<decltype(extents[0])>>(),
                    static_cast<const void*>(extents.data())),
           "Failed to write extents");
  CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
  CHECK_H5(H5Aclose(att_id), "Failed to close attribute");
}

void write_connectivity(const hid_t group_id,
                        const std::vector<int>& connectivity) noexcept {
  const hsize_t size = connectivity.size();
  const hid_t space_id = H5Screate_simple(1, &size, nullptr);
  CHECK_H5(space_id, "Failed to create dataspace");
  const hid_t dataset_id =
      H5Dcreate2(group_id, "connectivity", h5_type<int>(), space_id,
                 h5p_default(), h5p_default(), h5p_default());
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(
      H5Dwrite(dataset_id, h5_type<int>(), h5s_all(), h5s_all(), h5p_default(),
               static_cast<const void*>(connectivity.data())),
      "Failed to write connectivity");
  CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
  CHECK_H5(H5Dclose(dataset_id), "Failed to close dataset");
}

std::vector<std::string> get_group_names(
    const hid_t file_id, const std::string& group_name) noexcept {
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
                                      i, nullptr, 0, h5p_default()));
    name.resize(size);
    H5Lget_name_by_idx(group_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, &name[0],
                       size, h5p_default());
#pragma GCC diagnostic pop
    // We need to remove the bloody trailing \0...
    name.pop_back();
    names.push_back(name);
  }
  return names;
}

bool contains_dataset_or_group(const hid_t id, const std::string& group_name,
                               const std::string& dataset_name) noexcept {
  return alg::found(get_group_names(id, group_name), dataset_name);
}

template <typename Type>
void write_to_attribute(const hid_t location_id, const std::string& name,
                        const Type& value) noexcept {
  const hid_t space_id = H5Screate(H5S_SCALAR);
  CHECK_H5(space_id, "Failed to create scalar");
  const hid_t att_id = H5Acreate2(location_id, name.c_str(), h5_type<Type>(),
                                  space_id, h5p_default(), h5p_default());
  CHECK_H5(att_id, "Failed to create attribute '" << name << "'");
  CHECK_H5(H5Awrite(att_id, h5_type<Type>(), static_cast<const void*>(&value)),
           "Failed to write value: " << value);
  CHECK_H5(H5Aclose(att_id), "Failed to close attribute '" << name << "'");
  CHECK_H5(H5Sclose(space_id), "Unable to close dataspace");
}

template <typename Type>
Type read_value_attribute(const hid_t location_id,
                          const std::string& name) noexcept {
  const htri_t attribute_exists = H5Aexists(location_id, name.c_str());
  if (not attribute_exists) {
    ERROR("Could not find attribute '" << name << "'");  // LCOV_EXCL_LINE
  }
  const hid_t attribute_id = H5Aopen(location_id, name.c_str(), h5p_default());
  CHECK_H5(attribute_id, "Failed to open attribute '" << name << "'");
  Type value;
  CHECK_H5(H5Aread(attribute_id, h5_type<Type>(), &value),
           "Failed to read attribute '" << name << "'");
  CHECK_H5(H5Aclose(attribute_id),
           "Failed to close attribute '" << name << "'");
  return value;
}

template <typename T>
void write_to_attribute(const hid_t group_id, const std::string& name,
                        const std::vector<T>& data) noexcept {
  const hsize_t size = data.size();
  const hid_t space_id = H5Screate_simple(1, &size, nullptr);
  CHECK_H5(space_id,
           "Failed to create dataspace for attribute  '" << name << "'");
  const hid_t att_id = H5Acreate2(group_id, name.c_str(), h5::h5_type<T>(),
                                  space_id, h5p_default(), h5p_default());
  CHECK_H5(att_id, "Failed to create attribute '" << name << "'");
  CHECK_H5(
      H5Awrite(att_id, h5::h5_type<T>(), static_cast<const void*>(data.data())),
      "Failed to write extents into attribute '" << name << "'");
  CHECK_H5(H5Sclose(space_id),
           "Failed to close dataspace when writing attribute '" << name << "'");
  CHECK_H5(H5Aclose(att_id),
           "Failed to close attribute '" << name << "' when writing it.");
}

template <typename T>
std::vector<T> read_rank1_attribute(const hid_t group_id,
                                    const std::string& name) noexcept {
  const hid_t attr_id = H5Aopen(group_id, name.c_str(), h5p_default());
  CHECK_H5(attr_id, "Failed to open attribute");
  {  // Check that the datatype in the file matches what we are reading.
    const hid_t datatype_id = H5Aget_type(attr_id);
    CHECK_H5(datatype_id, "Failed to get datatype from attribute " << name);
    const hid_t datatype = H5Tget_native_type(datatype_id, H5T_DIR_DESCEND);
    const auto size = H5Tget_size(datatype);
    if (UNLIKELY(sizeof(T) != size)) {
      ERROR("The read HDF5 type of the attribute ("
            << datatype
            << ") has a different size than the type we are reading. The "
               "stored size is "
            << size << " while the expected size is " << sizeof(T));
    }
    CHECK_H5(H5Tclose(datatype_id),
             "Failed to close datatype while reading attribute " << name);
  }
  const auto size = [&attr_id, &name] {
    const hid_t dataspace_id = H5Aget_space(attr_id);
    const auto rank_of_space =
        H5Sget_simple_extent_ndims(dataspace_id);
    if (UNLIKELY(rank_of_space < 0)) {
      ERROR("Failed to get the rank of the dataspace inside the attribute "
            << name);
    }
    if (UNLIKELY(rank_of_space != 1)) {
      ERROR(
          "The rank of the dataspace being read by read_rank1_attribute should "
          "be 1 but is "
          << rank_of_space);
    }
    std::array<hsize_t, 1> dims{};
    if (UNLIKELY(H5Sget_simple_extent_dims(dataspace_id, dims.data(),
                                           nullptr) != 1)) {
      ERROR(
          "The rank of the dataspace has changed after checking its rank. "
          "Checked rank was "
          << rank_of_space);
    }
    H5Sclose(dataspace_id);
    return dims[0];
  }();
  std::vector<T> data(size);
  CHECK_H5(H5Aread(attr_id, h5::h5_type<T>(), data.data()),
           "Failed to read data from attribute " << name);
  H5Aclose(attr_id);
  return data;
}

template <>
void write_to_attribute<std::string>(
    const hid_t group_id, const std::string& name,
    const std::vector<std::string>& data) noexcept {
  // See the HDF5 example:
  // https://support.hdfgroup.org/ftp/HDF5/examples/examples-by-api/
  // hdf5-examples/1_8/C/H5T/h5ex_t_stringatt.c

  const hid_t type_id = fortran_string();
  // Create dataspace and attribute in dataspace where we will store the strings
  const hsize_t dim = data.size();
  const hid_t space_id = H5Screate_simple(1, &dim, nullptr);
  CHECK_H5(space_id, "Failed to create null space");
  const hid_t attr_id = H5Acreate2(group_id, name.c_str(), type_id, space_id,
                                   h5p_default(), h5p_default());
  CHECK_H5(attr_id, "Could not create attribute " << name);

  // We are using C-style strings, which is type to be written into attribute
  const auto memtype_id = h5_type<std::string>();

  // In order to write strings to an attribute we must have a pointer to
  // pointers, so we use a vector.
  std::vector<const char*> string_pointers(data.size());
  std::transform(data.begin(), data.end(), string_pointers.begin(),
                 [](const auto& t) { return t.c_str(); });
  CHECK_H5(H5Awrite(attr_id, memtype_id, string_pointers.data()),
           "Failed attribute write");

  CHECK_H5(H5Aclose(attr_id), "Failed to close attribute " << name);
  CHECK_H5(H5Sclose(space_id), "Failed to close space_id");
  CHECK_H5(H5Tclose(memtype_id), "Failed to close memtype_id");
  CHECK_H5(H5Tclose(type_id), "Failed to close type_id");
}

template <>
std::vector<std::string> read_rank1_attribute<std::string>(
    const hid_t group_id, const std::string& name) noexcept {
  const auto attribute_exists =
      static_cast<bool>(H5Aexists(group_id, name.c_str()));
  if (not attribute_exists) {
    ERROR("Could not find attribute '" << name << "'");  // LCOV_EXCL_LINE
  }

  // Open attribute that holds the strings
  const hid_t attribute_id = H5Aopen(group_id, name.c_str(), h5p_default());
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
  const hid_t memtype = h5_type<std::string>();
  CHECK_H5(H5Aread(attribute_id, memtype, static_cast<void*>(temp.data())),
           "Failed to read attribute");

  std::vector<std::string> result(temp.size());
  std::transform(temp.begin(), temp.end(), result.begin(),
                 [](const auto& t) { return std::string(t); });

  // Clean up memory from variable length arrays and close everything
  CHECK_H5(H5Dvlen_reclaim(memtype, dataspace_id, h5p_default(), temp.data()),
           "Failed H5Dvlen_reclaim at ");
  CHECK_H5(H5Aclose(attribute_id), "Failed to close attribute");
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close space_id");
  CHECK_H5(H5Tclose(memtype), "Failed to close memtype");
  return result;
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
                                      i, nullptr, 0, h5p_default()));
    name.resize(size);
    H5Aget_name_by_idx(group_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, &name[0],
                       size, h5p_default());
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

hid_t open_dataset(const hid_t group_id,
                   const std::string& dataset_name) noexcept {
  const hid_t dataset_id = H5Dopen2(
                           group_id, dataset_name.c_str(), h5p_default());
  CHECK_H5(dataset_id, "Failed to open dataset '" << dataset_name << "'");
  return dataset_id;
}

void close_dataset(const hid_t dataset_id) noexcept {
  CHECK_H5(H5Dclose(dataset_id), "Failed to close dataset");
}

hid_t open_dataspace(const hid_t dataset_id) noexcept {
  const hid_t dataspace_id = H5Dget_space(dataset_id);
  CHECK_H5(dataspace_id, "Failed to open dataspace");
  return dataspace_id;
}

void close_dataspace(const hid_t dataspace_id) noexcept {
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");
}

template <size_t Rank, typename T>
T read_data(const hid_t group_id, const std::string& dataset_name) noexcept {
  const hid_t dataset_id = open_dataset(group_id, dataset_name);
  const hid_t dataspace_id = open_dataspace(dataset_id);
  std::array<hsize_t, Rank> size{}, max_size{};
  if (static_cast<int>(Rank) !=
      H5Sget_simple_extent_dims(dataspace_id, nullptr, nullptr)) {
    ERROR("Incorrect rank in get_data(). Expected rank = "
          << Rank << " but received array with rank = "
          << H5Sget_simple_extent_dims(dataspace_id, nullptr,
                                       nullptr));
  }
  H5Sget_simple_extent_dims(dataspace_id, size.data(), max_size.data());
  close_dataspace(dataspace_id);

  // Load data from the H5 file by passing a pointer to 'data' to H5Dread
  // The type of 'data' is determined by the template parameter 'T'
  const size_t total_number_of_components = std::accumulate(
      size.begin(), size.end(), static_cast<size_t>(1), std::multiplies<>());
  if (UNLIKELY(total_number_of_components == 0)) {
    using ::operator<<;
    ERROR("At least one element in 'size' is 0. Expected data along "
          << Rank << " dimensions. size = " << size);
  }
  tmpl::conditional_t<
      cpp17::is_same_v<T, DataVector>, DataVector,
      std::vector<tt::get_fundamental_type_t<T>>>
      data(total_number_of_components);

  CHECK_H5(H5Dread(dataset_id,
                     h5_type<tt::get_fundamental_type_t<T>>(),
                     h5s_all(), h5s_all(), h5p_default(), data.data()),
             "Failed to read dataset: '" << dataset_name << "'");
  close_dataset(dataset_id);
  return VectorTo<Rank, T>::apply(std::move(data), size);
}

template <size_t Dim>
Index<Dim> read_extents(const hid_t group_id, const std::string& extents_name) {
  const hid_t attr_id = H5Aopen(group_id, extents_name.c_str(), h5p_default());
  CHECK_H5(attr_id, "Failed to open attribute");
  Index<Dim> extents;
  CHECK_H5(H5Aread(attr_id, h5::h5_type<std::decay_t<decltype(extents[0])>>(),
                   extents.data()),
           "Failed to read extents");
  H5Aclose(attr_id);
  return extents;
}

// Explicit instantiations
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

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define RANK(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_WRITE_DATA(_, DATA)                            \
  template void write_data<TYPE(DATA)>(                            \
      const hid_t group_id, const std::vector<TYPE(DATA)>& data,   \
      const std::vector<size_t>& extents,                          \
      const std::string& name) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_WRITE_DATA,
                        (double, int, unsigned int, long, unsigned long,
                         long long, unsigned long long, char))

#define INSTANTIATE_ATTRIBUTE(_, DATA)                                 \
  template void write_to_attribute<TYPE(DATA)>(                        \
      const hid_t group_id, const std::string& name,                   \
      const std::vector<TYPE(DATA)>& data) noexcept;                   \
  template void write_to_attribute<TYPE(DATA)>(                        \
      const hid_t location_id, const std::string& name,                \
      const TYPE(DATA) & value) noexcept;                              \
  template TYPE(DATA) read_value_attribute<TYPE(DATA)>(                \
      const hid_t location_id, const std::string& name) noexcept;      \
  template std::vector<TYPE(DATA)> read_rank1_attribute<TYPE(DATA)>(   \
      const hid_t group_id, const std::string& name) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_ATTRIBUTE,
                        (double, unsigned int, unsigned long, int))

#define INSTANTIATE_READ_SCALAR(_, DATA)                 \
  template TYPE(DATA) read_data<RANK(DATA), TYPE(DATA)>( \
      const hid_t group_id, const std::string& dataset_name) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_READ_SCALAR,
                        (double, int, unsigned int, long, unsigned long,
                         long long, unsigned long long, char),
                        (0))

#define INSTANTIATE_READ_VECTOR(_, DATA)          \
  template std::vector<TYPE(DATA)>                \
  read_data<RANK(DATA), std::vector<TYPE(DATA)>>( \
      const hid_t group_id, const std::string& dataset_name) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_READ_VECTOR,
                        (double, int, unsigned int, long, unsigned long,
                         long long, unsigned long long, char),
                        (1))

#define INSTANTIATE_READ_MULTIARRAY(_, DATA)                         \
  template boost::multi_array<TYPE(DATA), RANK(DATA)>                \
  read_data<RANK(DATA), boost::multi_array<TYPE(DATA), RANK(DATA)>>( \
      const hid_t group_id, const std::string& dataset_name) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_READ_MULTIARRAY,
                        (double, int, unsigned int, long, unsigned long,
                         long long, unsigned long long, char),
                        (2, 3))

#define INSTANTIATE_READ_DATAVECTOR(_, DATA)             \
  template TYPE(DATA) read_data<RANK(DATA), TYPE(DATA)>( \
      const hid_t group_id, const std::string& dataset_name) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_READ_DATAVECTOR, (DataVector), (1, 2, 3))

#undef INSTANTIATE_ATTRIBUTE
#undef INSTANTIATE_WRITE_DATA
#undef INSTANTIATE_READ_SCALAR
#undef INSTANTIATE_READ_VECTOR
#undef INSTANTIATE_READ_MULTIARRAY
#undef INSTANTIATE_READ_DATAVECTOR
#undef TYPE
#undef RANK
}  // namespace h5

namespace h5 {
namespace detail {
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

  const hid_t dataset_id =
      H5Dcreate2(group_id, name.c_str(), h5_type<double>(), dataspace_id,
                 h5p_default(), property_list, h5p_default());
  CHECK_H5(dataset_id, "Failed to create dataset");
  CHECK_H5(H5Pclose(property_list), "Failed to close property list");
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");
  return dataset_id;
}

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
