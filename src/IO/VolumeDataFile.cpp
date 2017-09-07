// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "VolumeDataFile.hpp"

#include <numeric>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Type.hpp"
#include "Utilities/FileSystem.hpp"

namespace vis {

VolumeFile::VolumeFile(std::string file_name, int format_id)
    : h5_file_name_(std::move(file_name)), format_id_(format_id) {
  create_file();
}

VolumeFile::~VolumeFile() {
  CHECK_H5(H5Fclose(file_id_),
           "Failed to close file '" << h5_file_name_ << "'");
  std::fclose(xml_);
}

void VolumeFile::write_xdmf_time(const double time) {
  std::fprintf(xml_, "<Time Value=\"%1.18e\"/>\n", time);  // NOLINT
}

void VolumeFile::write_element_time(const double time,
                                    const std::string& element_id) {
  const h5::detail::OpenGroup group(file_id_, "/" + element_id,
                                    h5::AccessType::ReadWrite);
  h5::write_time(group.id(), time);
}

template <size_t Dim>
void VolumeFile::write_element_connectivity(const Index<Dim>& extents,
                                            const std::string& element_id) {
  const std::vector<vis::detail::CellInTopology> cells =
      vis::detail::compute_cells(extents);
  const std::vector<int> connectivity = [&cells]() {
    std::vector<int> local_connectivity;
    for (const auto& cell : cells) {
      for (const auto& bounding_indices : cell.bounding_indices) {
        local_connectivity.emplace_back(bounding_indices);
      }
    }
    return local_connectivity;
  }();

  const h5::detail::OpenGroup group(file_id_, "/" + element_id,
                                    h5::AccessType::ReadWrite);
  h5::write_connectivity(group.id(), connectivity);
  std::stringstream ss;
  ss << "<Grid Name=\"" << element_id << "\" GrideType=\"Uniform\">\n"
     << "  <Topology TopologyType=\"";
  if (Dim == 1) {
    ss << "Polyvertex";
  } else if (Dim == 2) {
    ss << "Quadrilateral";
  } else if (Dim == 3) {
    ss << "Hexahedron";
  }
  const size_t number_of_elements =
      std::accumulate(extents.begin(), extents.end(), 1.0,
                      [](const size_t& state, const size_t& element) {
                        return state * (element - 1);
                      });
  ss << "\" NumberOfElements=\"" << number_of_elements << "\">\n"
     << "    <DataItem Dimensions=\"" << extents[0] - 1;
  for (size_t j = 1; j < Dim; ++j) {
    ss << " " << extents[j] - 1;
  }
  if (Dim == 1) {
    ss << " 2\"";
  } else if (Dim == 2) {
    ss << " 4\"";
  } else if (Dim == 3) {
    ss << " 8\"";
  }
  ss << " NumberType=\"Int\" Format=\"HDF5\">\n      " << h5_file_name_ << ":/"
     << element_id << "/connectivity\n    </DataItem>\n  </Topology>\n";
  std::fprintf(xml_, "%s", ss.str().c_str());  // NOLINT
}

template <size_t Dim>
void VolumeFile::write_element_extents(const Index<Dim>& extents,
                                       const std::string& element_id) {
  const h5::detail::OpenGroup group(file_id_, "/" + element_id,
                                    h5::AccessType::ReadWrite);
  h5::write_extents(group.id(), extents);
}

template <size_t Dim>
void VolumeFile::write_element_data(
    std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                              std::vector<DataVector>>>
        vars_map,
    const Index<Dim>& extents, const std::string& element_id) {
  const h5::detail::OpenGroup group(file_id_, "/" + element_id,
                                    h5::AccessType::ReadWrite);
  const std::string extents_in_dims = [&extents]() {
    std::stringstream local_extents_in_dims;
    local_extents_in_dims << extents[0];
    for (size_t k = 1; k < Dim; ++k) {
      local_extents_in_dims << " " << extents[k];
    }
    return local_extents_in_dims.str();
  }();
  std::stringstream ss;
  for (const auto& key_value : vars_map) {
    const auto& name = key_value.first;
    const auto& suffixes = key_value.second.first;
    const auto& vars = key_value.second.second;
    ASSERT(suffixes.size() == vars.size() or
               (suffixes.empty() and vars.size() == 1),
           "The number of suffixes must be equal to the number of "
           "variables in the serialized tensor. Variable is: '"
               << name << "' with suffixes: " << suffixes
               << " has suffix size: " << suffixes.size() << " and "
               << vars.size() << " variables.");
    if (Dim == suffixes.size() and suffixes[0] != "Scalar" and Dim == 3) {
      // Write vector data into h5 file
      ss << "  <Attribute Name=\"" << name << "\"\n"
         << " AttributeType=\"Vector\" Center=\"Node\">\n"
         << "    <DataItem Dimensions=\"" << extents_in_dims << " " << Dim
         << "\" ItemType = \"Function\" Function = \"JOIN($0,$1,$2)\">\n";
      for (size_t i = 0; i < suffixes.size(); ++i) {
        h5::write_data(group.id(), vars[i], extents, name + "_" + suffixes[i]);
        ss << "    <DataItem Dimensions=\"" << extents_in_dims
           << "\" NumberType=\"Double\" Precision=\"" << precision_
           << "\" Format=\"HDF5\">\n"
           << "      " << h5_file_name_ << ":/" << element_id << "/" << name
           << "_" << suffixes[i] << "\n    </DataItem>\n";
      }
      ss << "    </DataItem>\n"
         << "  </Attribute>\n";
    } else if (suffixes.size() != vars.size() or suffixes[0] == "Scalar") {
      h5::write_data(group.id(), vars[0], extents, name);
      ss << "  <Attribute Name=\"" << name
         << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
         << "    <DataItem Dimensions=\"" << extents_in_dims
         << "\" NumberType=\"Double\" Precision=\"" << precision_
         << "\" Format=\"HDF5\">\n"
         << "      " << h5_file_name_ << ":/" << element_id << "/" << name
         << "\n    </DataItem>\n  </Attribute>\n";
    } else {
      // Write tensor data into h5 file
      for (size_t i = 0; i < suffixes.size(); ++i) {
        h5::write_data(group.id(), vars[i], extents, name + "_" + suffixes[i]);
        ss << "  <Attribute Name=\"" << name << "_" << suffixes[i]
           << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
           << "    <DataItem Dimensions=\"" << extents_in_dims
           << "\" NumberType=\"Double\" Precision=\"" << precision_
           << "\" Format=\"HDF5\">\n"
           << "      " << h5_file_name_ << ":/" << element_id << "/" << name
           << "_" << suffixes[i] << "\n    </DataItem>\n  </Attribute>\n";
      }
    }
  }
  ss << "</Grid>\n";
  std::fprintf(xml_, "%s", ss.str().c_str());  // NOLINT
  std::fflush(xml_);
}

void VolumeFile::create_file() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  file_id_ =
      H5Fcreate(h5_file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(file_id_, "Failed to create file '" << h5_file_name_ << "'");
  const h5::detail::OpenGroup group(file_id_, "/", h5::AccessType::ReadWrite);
  hid_t s_id = H5Screate(H5S_SCALAR);
  CHECK_H5(s_id, "Failed to create dataspace");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  const hid_t att_id = H5Acreate2(group.id(), "FileFormatVersion",
                                  h5::h5_type<decltype(format_id_)>(), s_id,
                                  H5P_DEFAULT, H5P_DEFAULT);
#pragma GCC diagnostic pop
  CHECK_H5(att_id, "Failed to create attribute 'FileFormatVersion'");
  CHECK_H5(H5Awrite(att_id, h5::h5_type<decltype(format_id_)>(),
                    static_cast<void*>(&format_id_)),
           "Failed to write file format version");
  CHECK_H5(H5Sclose(s_id), "Failed to close dataspace");
  CHECK_H5(H5Aclose(att_id), "Failed to close attribute");

  xml_file_name_ +=
      h5_file_name_.substr(0, h5_file_name_.find_last_of('.')) + ".xmf";

  xml_ = std::fopen(xml_file_name_.c_str(), "w");
  if (nullptr == xml_) {
    // LCOV_EXCL_START
    ERROR("Failed to open file '" << xml_file_name_ << "' with error '"
                                  << std::strerror(errno) << "'.");
    // LCOV_EXCL_STOP
  }
}

// Explicit instantiations
template void VolumeFile::write_element_connectivity<1>(
    const Index<1>& extents, const std::string& element_id);
template void VolumeFile::write_element_connectivity<2>(
    const Index<2>& extents, const std::string& element_id);
template void VolumeFile::write_element_connectivity<3>(
    const Index<3>& extents, const std::string& element_id);

/// \cond HIDDEN_SYMBOLS
template void VolumeFile::write_element_extents<1>(
    const Index<1>& extents, const std::string& element_id);
template void VolumeFile::write_element_extents<2>(
    const Index<2>& extents, const std::string& element_id);
template void VolumeFile::write_element_extents<3>(
    const Index<3>& extents, const std::string& element_id);

template void VolumeFile::write_element_data<1>(
    std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                              std::vector<DataVector>>>
        vars_map,
    const Index<1>& extents, const std::string& element_id);
template void VolumeFile::write_element_data<2>(
    std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                              std::vector<DataVector>>>
        vars_map,
    const Index<2>& extents, const std::string& element_id);
template void VolumeFile::write_element_data<3>(
    std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                              std::vector<DataVector>>>
        vars_map,
    const Index<3>& extents, const std::string& element_id);
/// \endcond
}  // namespace vis
