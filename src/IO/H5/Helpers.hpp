// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for h5 manipulations

#pragma once

#include <array>
#include <cstddef>
#include <hdf5.h>
#include <string>
#include <vector>

#include "DataStructures/Index.hpp"

/// \cond
class DataVector;
/// \endcond

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Write a std::vector named `name` to the group `group_id`
 */
template <typename T>
void write_data(hid_t group_id, const std::vector<T>& data,
                const std::vector<size_t>& extents,
                const std::string& name = "scalar") noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Write a DataVector named `name` to the group `group_id`
 */
void write_data(hid_t group_id, const DataVector& data,
                const std::string& name) noexcept;

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
 * \brief Write a value of type `Type` to an HDF5 attribute named `name`
 */
template <typename Type>
void write_to_attribute(hid_t location_id, const std::string& name,
                        const Type& value) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Write the vector `data` to the attribute `attribute_name` in the group
 * `group_id`.
 */
template <typename T>
void write_to_attribute(hid_t group_id, const std::string& name,
                        const std::vector<T>& data) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Read a value of type `Type` from an HDF5 attribute named `name`
 */
template <typename Type>
Type read_value_attribute(hid_t location_id, const std::string& name) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Read rank-1 of type `Type` from an HDF5 attribute named `name`
 */
template <typename T>
std::vector<T> read_rank1_attribute(hid_t group_id,
                                    const std::string& name) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Get the names of all the attributes in a group
 */
std::vector<std::string> get_attribute_names(hid_t file_id,
                                             const std::string& group_name);

/*!
 * \ingroup HDF5Group
 * \brief Write the connectivity into the group in the H5 file
 */
void write_connectivity(hid_t group_id,
                        const std::vector<int>& connectivity) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Get the names of all the groups and datasets in a group
 */
std::vector<std::string> get_group_names(
    hid_t file_id, const std::string& group_name) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Check if `name` is a dataset or group in the subgroup `group_name` of
 * `id`.
 *
 * \note To check the current id for `name`, pass `""` as `group_name`.
 */
bool contains_dataset_or_group(hid_t id, const std::string& group_name,
                               const std::string& dataset_name) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Check if an attribute is in a group
 */
bool contains_attribute(hid_t file_id, const std::string& group_name,
                        const std::string& attribute_name);

/*!
 * \ingroup HDF5Group
 * \brief Open an HDF5 dataset
 */
hid_t open_dataset(hid_t group_id,
                   const std::string& dataset_name) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Close an HDF5 dataset
 */
void close_dataset(hid_t dataset_id) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Open an HDF5 dataspace
 */
hid_t open_dataspace(hid_t dataset_id) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Close an HDF5 dataspace
 */
void close_dataspace(hid_t dataspace_id) noexcept;

/*!
 * \ingroup HDF5Group
 * \brief Read an array of rank 0-3 into an object.
 *
 * For each rank, the data can be read into objects of the following types:
 * rank 0: double or int
 * rank 1: std::vector or DataVector
 * rank 2: boost::multiarray or DataVector
 * rank 3: boost::multiarray or DataVector
 */
template <size_t Rank, typename T>
T read_data(hid_t group_id, const std::string& dataset_name) noexcept;

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
