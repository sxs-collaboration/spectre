// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <string>
#include <vector>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

/// \cond
class DataVector;
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
 * `insert_tensor_data()` method. An integral observation id is used to keep
 * track of the observation instance at which the data is written, and
 * associated with it is a floating point observation value, such as the
 * simulation time at which the data was written. The observation id will
 * generally be the result of hashing the temporal identifier used for the
 * simulation.
 *
 * The data stored in the subfile are the tensor components passed to the
 * `insert_tensor_data()` method as a `ExtentsAndTensorVolumeData`. The name of
 * each tensor component must follow the format
 * `GRID_NAME/TENSOR_NAME_COMPONENT`, e.g. `Element0/T_xx`. Typically the
 * `GRID_NAME` should be the output of the stream operator of the spatial ID of
 * the parallel component element sending the data to be observed. For example,
 * in the case of a dG evolution where the spatial IDs are `ElementId`s, the
 * grid names would be of the form `[B0,(L2I3,L2I3,L2I3)]`.
 *
 * \warning Currently the topology of the grids is assumed to be tensor products
 * of lines, i.e. lines, quadrilaterals, and hexahedrons. However, this can be
 * extended in the future. If support for more topologies is required, please
 * file an issue.
 */
class VolumeData : public h5::Object {
 public:
  static std::string extension() noexcept { return ".vol"; }

  VolumeData(bool exists, detail::OpenGroup&& group, hid_t location,
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
  void insert_tensor_data(
      size_t observation_id, double observation_value,
      const ExtentsAndTensorVolumeData& extents_and_tensors) noexcept;

  /// List all the integral observation ids in the subfile
  std::vector<size_t> list_observation_ids() const noexcept;

  /// Get the observation value at the the integral observation id in the
  /// subfile
  double get_observation_value(size_t observation_id) const noexcept;

  /// List all the grid ids at the the integral observation id in the
  /// subfile
  std::vector<std::string> list_grids(size_t observation_id) const noexcept;

  /// List all the tensor components on the grid `grid_name` at observation id
  /// `observation_id`
  std::vector<std::string> list_tensor_components(
      size_t observation_id, const std::string& grid_name) const noexcept;

  /// Read a tensor component with name `tensor_component` from the grid
  /// `grid_name` at the observation id `observation_id`
  DataVector get_tensor_component(size_t observation_id,
                                  const std::string& grid_name,
                                  const std::string& tensor_component) const
      noexcept;

  /// Read the extents of the grid `grid_name` at the observation id
  /// `observation_id`
  std::vector<size_t> get_extents(size_t observation_id,
                                  const std::string& grid_name) const noexcept;

 private:
  detail::OpenGroup group_{};
  std::string name_{};
  uint32_t version_{};
  detail::OpenGroup volume_data_group_{};
  std::string header_{};
};
}  // namespace h5
