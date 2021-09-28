// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexModalVector.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"

namespace Cce {
namespace TestHelpers {

// records worldtube data in the SpEC h5 style.
struct WorldtubeModeRecorder {
 public:
  WorldtubeModeRecorder(const std::string& filename, const size_t l_max)
      : output_file_{filename} {
    // write the .ver that indicates that the derivatives are correctly
    // normalized.
    output_file_.insert<h5::Version>(
        "/VersionHist", "Bugfix in CCE radial derivatives (ticket 1096).");
    output_file_.close_current_object();
    file_legend_.emplace_back("time");
    for (int l = 0; l <= static_cast<int>(l_max); ++l) {
      for (int m = -l; m <= l; ++m) {
        file_legend_.push_back("Real Y_" + std::to_string(l) + "," +
                               std::to_string(m));
        file_legend_.push_back("Imag Y_" + std::to_string(l) + "," +
                               std::to_string(m));
      }
    }
  }

  // append to `dataset_path` the vector created by `time` followed by the
  // `modes` rearranged to be compatible with SpEC h5 format.
  void append_worldtube_mode_data(const std::string& dataset_path,
                                  const double time,
                                  const ComplexModalVector& modes,
                                  const size_t l_max) {
    auto& output_mode_dataset =
        output_file_.try_insert<h5::Dat>(dataset_path, file_legend_, 0);
    const size_t output_size = square(l_max + 1);
    std::vector<double> data_to_write(2 * output_size + 1);
    data_to_write[0] = time;
    for (int l = 0; l <= static_cast<int>(l_max); ++l) {
      for (int m = -l; m <= l; ++m) {
        data_to_write[2 * Spectral::Swsh::goldberg_mode_index(
                              l_max, static_cast<size_t>(l), -m) +
                      1] =
            real(modes[Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m)]);
        data_to_write[2 * Spectral::Swsh::goldberg_mode_index(
                              l_max, static_cast<size_t>(l), -m) +
                      2] =
            imag(modes[Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m)]);
      }
    }
    output_mode_dataset.append(data_to_write);
    output_file_.close_current_object();
  }

 private:
  h5::H5File<h5::AccessType::ReadWrite> output_file_;
  std::vector<std::string> file_legend_;
};
}  // namespace TestHelpers
}  // namespace Cce
