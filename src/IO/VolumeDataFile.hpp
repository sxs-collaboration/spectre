// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines VolumeFile used to write volume data to disk

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/H5/Helpers.hpp"

namespace vis {

/// \ingroup HDF5Group
/// Writes out volume data as an HDF5 file and an XDMF file to be visualized in
/// ParaView or VisIt
class VolumeFile {
 public:
  /// Open a file for volume data output
  VolumeFile(std::string file_name, int format_id);
  VolumeFile(const VolumeFile& /*rhs*/) = delete;
  VolumeFile& operator=(const VolumeFile& /*rhs*/) = delete;
  VolumeFile(VolumeFile&& /*rhs*/) = delete;
  VolumeFile& operator=(VolumeFile&& /*rhs*/) = delete;
  ~VolumeFile();

  /*!
   * Writes the current time into the XDMF file and must only be called once
   * per time step. `write_xdmf_time()` must also be called before looping
   * over the elements and calling
   * `write_element_connectivity_and_coordinates()`
   *
   * \param time the coordinate time
   */
  void write_xdmf_time(double time);

  /*!
   * Write the time, extents, connectivity and grid coordinates into the volume
   * HDF5 file for the specific element.
   *
   * \param time the coordinate time
   * \param grid_coordinates the coordinates of the grid
   * \param extents the extents of the element
   * \param element_id the name/ID of the element
   * \param coordinate_names the names of the grid coordinates
   */
  template <size_t Dim, typename Fr>
  void write_element_connectivity_and_coordinates(
      double time, const tnsr::I<DataVector, Dim, Fr>& grid_coordinates,
      const Index<Dim>& extents, const std::string& element_id,
      const std::vector<std::string>& coordinate_names = {"x-coord", "y-coord",
                                                          "z-coord"});

  /// Write the extents into the volume HDF5 file for the specific element
  /// \param vars_map the map storing all the variables to be written
  /// \param extents the extents of the element
  /// \param element_id the name/ID of the element
  template <size_t Dim>
  void write_element_data(
      std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                                std::vector<DataVector>>>
          vars_map,
      const Index<Dim>& extents, const std::string& element_id);

 private:
  void create_file();

  /// Write the time into the volume HDF5 file for the specific element
  /// \param element_id the name/ID of the element
  void write_element_time(double time, const std::string& element_id);

  /// Write the extents into the volume HDF5 file for the specific element
  /// \param extents the extents of the element
  /// \param element_id the name/ID of the element
  template <size_t Dim>
  void write_element_extents(const Index<Dim>& extents,
                             const std::string& element_id);

  /// Write the connectivity into the volume HDF5 file for the specific element
  /// \param extents the extents of the element
  /// \param element_id the name/ID of the element
  template <size_t Dim>
  void write_element_connectivity(const Index<Dim>& extents,
                                  const std::string& element_id);

  /// Write the coordinates into the volume HDF5 file for the specific element
  /// \param grid_coordinates the coordinates of the grid
  /// \param extents the extents of the element
  /// \param element_id the name/ID of the element
  /// \param coordinate_names the names of the grid coordinates
  template <size_t Dim, typename Fr>
  void write_element_coordinates(
      const tnsr::I<DataVector, Dim, Fr>& grid_coordinates,
      const Index<Dim>& extents, const std::string& element_id,
      const std::vector<std::string>& coordinate_names = {"x-coord", "y-coord",
                                                          "z-coord"});

  std::string h5_file_name_, xml_file_name_;
  int format_id_{1};
  int precision_{8};
  hid_t file_id_{-1};
  FILE* xml_{nullptr};
};

// ======================================================================
// Template Definitions
// ======================================================================

template <size_t Dim, typename Fr>
void VolumeFile::write_element_connectivity_and_coordinates(
    const double time, const tnsr::I<DataVector, Dim, Fr>& grid_coordinates,
    const Index<Dim>& extents, const std::string& element_id,
    const std::vector<std::string>& coordinate_names) {
  write_element_time(time, element_id);
  write_element_extents(extents, element_id);
  write_element_connectivity(extents, element_id);
  write_element_coordinates(grid_coordinates, extents, element_id,
                            coordinate_names);
}

template <size_t Dim, typename Fr>
void VolumeFile::write_element_coordinates(
    const tnsr::I<DataVector, Dim, Fr>& grid_coordinates,
    const Index<Dim>& extents, const std::string& element_id,
    const std::vector<std::string>& coordinate_names) {
  h5::detail::OpenGroup group(file_id_, "/" + element_id,
                              h5::AccessType::ReadWrite);
  for (size_t d = 0; d < Dim; ++d) {
    h5::write_data(group.id(), grid_coordinates.get(d), extents,
                   coordinate_names[d]);
  }
  std::stringstream ss;
  ss << "  <Geometry Type=\"X";
  if (Dim > 1) {
    ss << "_Y";
  }
  if (Dim > 2) {
    ss << "_Z";
  }
  ss << "\">\n";
  std::stringstream extents_in_dims;
  extents_in_dims << extents[0];
  for (size_t k = 1; k < Dim; ++k) {
    extents_in_dims << " " << extents[k];
  }
  for (size_t d = 0; d < Dim; ++d) {
    ss << "    <DataItem Dimensions=\"" << extents_in_dims.str()
       << "\" NumberType=\"Double\" Precision=\"" << precision_
       << "\" Format=\"HDF5\">\n"
       << "      " << h5_file_name_ << ":/" << element_id << "/"
       << coordinate_names[d] << "\n    </DataItem>\n";
  }
  ss << "  </Geometry>\n";
  std::fprintf(xml_, "%s", ss.str().c_str());  // NOLINT
}
}  // namespace vis
