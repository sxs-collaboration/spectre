// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for h5 manipulations

#pragma once

#include <hdf5.h>
#include <list>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/OpenGroup.hpp"

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Write a "Time" attribute to the group
 */
void write_time(hid_t group_id, double time);

/*!
 * \ingroup HDF5Group
 * \brief Write a DataVector named `name` to the group `group_id`
 */
template <size_t Dim>
void write_data(hid_t group_id, const DataVector& data,
                const Index<Dim>& extents, const std::string& name = "scalar");

/*!
 * \ingroup HDF5Group
 * \brief Write the extents as an attribute named `name` to the group
 * `group_id`.
 */
template <size_t Dim>
void write_extents(hid_t group_id, const Index<Dim>& extents,
                   const std::string& name = "Extents");

/*!
 * \ingroup HDF5Group
 * \brief Write the connectivity into the group in the H5 file
 */
void write_connectivity(hid_t group_id, const std::vector<int>& connectivity);

/*!
 * \ingroup HDF5Group
 * \brief Get the names of all the groups in a group
 */
std::vector<std::string> get_group_names(hid_t file_id,
                                         const std::string& group_name);

/*!
 * \ingroup HDF5Group
 * \brief Get the names of all the attributes in a group
 */
std::vector<std::string> get_attribute_names(hid_t file_id,
                                             const std::string& group_name);

/*!
 * \ingroup HDF5Group
 * \brief Check if an attribute is in a group
 */
bool contains_attribute(hid_t file_id, const std::string& group_name,
                        const std::string& attribute_name);

/*!
 * \ingroup HDF5Group
 * \brief Get the "Time" attribute from a group
 */
double get_time(hid_t file_id, const std::string& group_name,
                const std::string& attr_name = "Time");

/*!
 * \ingroup HDF5Group
 * \brief Read a DataVector from a dataset in a group
 */
DataVector read_data(hid_t group_id, const std::string& dataset_name);

/*!
 * \ingroup HDF5Group
 * \brief Read the HDF5 attribute representing extents from a group
 */
template <size_t Dim>
Index<Dim> read_extents(hid_t group_id,
                        const std::string& extents_name = "Extents");

}  // namespace h5

namespace h5 {
namespace detail {
/*!
 * \ingroup HDF5Group
 * \brief Write a vector of strings into an H5 dataset as an attribute
 *
 * \requires `dataset_id` is an open dataset
 * \effects Writes the `string_array` into an HDF5 attribute with name `name`
 */
void write_strings_to_attribute(hid_t dataset_id, const std::string& name,
                                const std::vector<std::string>& string_array);

/*!
 * \ingroup HDF5Group
 * \brief Read a vector of strings from an attribute inside an H5 dataset
 *
 * \requires `dataset_id` is an open dataset
 * \effects Reads the HDF5 attribute with name `name` as a vector of strings
 */
std::vector<std::string> read_strings_from_attribute(hid_t group_id,
                                                     const std::string& name);

/*!
 * \ingroup HDF5Group
 * \brief Write a value of type `Type` to an HDF5 attribute named `name`
 */
template <typename Type>
void write_value_to_attribute(hid_t location_id, const std::string& name,
                              const Type& value);

/*!
 * \ingroup HDF5Group
 * \brief Read a value of type `Type` from an HDF5 attribute named `name`
 */
template <typename Type>
Type read_value_from_attribute(hid_t location_id, const std::string& name);

/*!
 * \ingroup HDF5Group
 * \brief Create a dataset that can be extended/appended to
 *
 * \requires group_id is an open group, each element of `initial_size` is less
 * than the respective element in `max_size`, and each element in `max_size` is
 * a positive integer or `H5S_UNLIMITED`
 * \effects creates a potentially extensible dataset of dimension Dim inside the
 * group `group_id`
 * \returns the HDF5 id to the created dataset
 *
 * See the tutorial at https://support.hdfgroup.org/HDF5/Tutor/extend.html
 * for details on the implementation choice.
 */
template <size_t Dims>
hid_t create_extensible_dataset(hid_t group_id, const std::string& name,
                                const std::array<hsize_t, Dims>& initial_size,
                                const std::array<hsize_t, Dims>& chunk_size,
                                const std::array<hsize_t, Dims>& max_size);
}  // namespace detail
}  // namespace h5
