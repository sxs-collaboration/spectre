// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

/// \cond
class DataVector;
class ElementVolumeData;
class ExtentsAndTensorVolumeData;
/// \endcond

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief A volume data subfile written inside an H5 file.
 *
 * The volume data inside the subfile can be of any dimensionality greater than
 * zero. This means that in a 3D simulation, data on 2-dimensional surfaces are
 * written as a VolumeData subfile. Data can be written using the
 * `write_volume_data()` method. An integral observation id is used to keep
 * track of the observation instance at which the data is written, and
 * associated with it is a floating point observation value, such as the
 * simulation time at which the data was written. The observation id will
 * generally be the result of hashing the temporal identifier used for the
 * simulation.
 *
 * The data stored in the subfile are the tensor components passed to the
 * `write_volume_data()` method as a  `std::vector<ExtentsAndTensorVolumeData>`.
 * The name of each tensor component must follow the format
 * `GRID_NAME/TENSOR_NAME_COMPONENT`, e.g. `Element0/T_xx`. Typically the
 * `GRID_NAME` should be the output of the stream operator of the spatial ID of
 * the parallel component element sending the data to be observed. For example,
 * in the case of a dG evolution where the spatial IDs are `ElementId`s, the
 * grid names would be of the form `[B0,(L2I3,L2I3,L2I3)]`.  The data are
 * written contiguously inside of the H5 subfile, in that each tensor
 * component has a single dataset which holds all of the data from
 * all elements, e.g. a tensor component `T_xx` which is found on all grids
 * appears in the path `H5_FILE_NAME/element_data.vol/T_xx` and that is where
 * all of the `T_xx` data from all of the grids resides.  Note that coordinates
 * must be written as tensors in order to visualize the data in ParaView,
 * Visit, etc.   In order to reconstruct which data came from which grid,
 * the `get_grid_names()`, and `get_extents()` methods list the grids
 * and their extents in the order which they and the data were written.
 * For example, if the first grid has name `GRID_NAME` with extents
 * `{2, 2, 2}`, it was responsible for contributing the first 2*2*2 = 8 grid
 * points worth of data in each tensor dataset. Use the
 * `h5::offset_and_length_for_grid` function to compute the offset into the
 * contiguous dataset that corresponds to a particular grid.
 *
 * \warning Currently the topology of the grids is assumed to be tensor products
 * of lines, i.e. lines, quadrilaterals, and hexahedrons. However, this can be
 * extended in the future. If support for more topologies is required, please
 * file an issue.
 */
class VolumeData : public h5::Object {
 public:
  static std::string extension() noexcept { return ".vol"; }

  VolumeData(bool subfile_exists, detail::OpenGroup&& group, hid_t location,
             const std::string& name, uint32_t version = 1) noexcept;

  VolumeData(const VolumeData& /*rhs*/) = delete;
  VolumeData& operator=(const VolumeData& /*rhs*/) = delete;
  VolumeData(VolumeData&& /*rhs*/) noexcept = delete;             // NOLINT
  VolumeData& operator=(VolumeData&& /*rhs*/) noexcept = delete;  // NOLINT

  ~VolumeData() override = default;

  /*!
   * \returns the header of the VolumeData file
   */
  const std::string& get_header() const noexcept { return header_; }

  /*!
   * \returns the user-specified version number of the VolumeData file
   *
   * \note h5::Version returns a uint32_t, so we return one here too for the
   * version
   */
  uint32_t get_version() const noexcept { return version_; }

  /// Insert tensor components at `observation_id` with floating point value
  /// `observation_value`
  ///
  /// \requires The names of the tensor components is of the form
  /// `GRID_NAME/TENSOR_NAME_COMPONENT`, e.g. `Element0/T_xx`
  void write_volume_data(
      size_t observation_id, double observation_value,
      const std::vector<ElementVolumeData>& elements) noexcept;

  /// List all the integral observation ids in the subfile
  std::vector<size_t> list_observation_ids() const noexcept;

  /// Get the observation value at the the integral observation id in the
  /// subfile
  double get_observation_value(size_t observation_id) const noexcept;

  /// Find the observation ID that matches the `observation_value`
  size_t find_observation_id(const double observation_value) const noexcept {
    for (auto& observation_id : list_observation_ids()) {
      if (get_observation_value(observation_id) == observation_value) {
        return observation_id;
      }
    }
    ERROR("No observation with value " << observation_value
                                       << " found in volume file.");
  }

  /// List all the tensor components at observation id `observation_id`
  std::vector<std::string> list_tensor_components(
      size_t observation_id) const noexcept;

  /// List the names of all the grids at observation id `observation_id`
  std::vector<std::string> get_grid_names(size_t observation_id) const noexcept;

  /// Read a tensor component with name `tensor_component` at observation id
  /// `observation_id` from all grids in the file
  DataVector get_tensor_component(
      size_t observation_id,
      const std::string& tensor_component) const noexcept;

  /// Read the extents of all the grids stored in the file at the observation id
  /// `observation_id`
  std::vector<std::vector<size_t>> get_extents(
      size_t observation_id) const noexcept;

  /// Read the dimensionality of the grids.  Note : This is the dimension of
  /// the grids as manifolds, not the dimension of the embedding space.  For
  /// example, the volume data of a sphere is 2-dimensional, even though
  /// each point has an x, y, and z coordinate.
  size_t get_dimension() const noexcept;

  /// Return the character used as a separator between grids in the subfile.
  static char separator() noexcept { return ':'; }

  /// Return the basis being used for each element along each axis
  std::vector<std::vector<std::string>> get_bases(
      size_t observation_id) const noexcept;

  /// Return the quadrature being used for each element along each axis
  std::vector<std::vector<std::string>> get_quadratures(
      size_t observation_id) const noexcept;

 private:
  detail::OpenGroup group_{};
  std::string name_{};
  uint32_t version_{};
  detail::OpenGroup volume_data_group_{};
  std::string header_{};
};

/*!
 * \brief Find the interval within the contiguous dataset stored in
 * `h5::VolumeData` that holds data for a particular `grid_name`.
 *
 * `h5::VolumeData` stores data for all grids that compose the volume
 * contiguously. This function helps with reconstructing which part of that
 * contiguous dataset belongs to a particular grid. See the `h5::VolumeData`
 * documentation for more information on how it stores data.
 *
 * To use this function, call `h5::VolumeData::get_grid_names` and
 * `h5::VolumeData::get_extents` and pass the results for the `all_grid_names`
 * and `all_extents` arguments, respectively. This means you can retrieve this
 * information from an `h5::VolumeData` once and use it to call
 * `offset_and_length_for_grid` multiple times with different `grid_name`s.
 *
 * Here is an example for using this function:
 *
 * \snippet Test_VolumeData.cpp find_offset
 *
 * \see `h5::VolumeData`
 */
std::pair<size_t, size_t> offset_and_length_for_grid(
    const std::string& grid_name,
    const std::vector<std::string>& all_grid_names,
    const std::vector<std::vector<size_t>>& all_extents) noexcept;

}  // namespace h5
