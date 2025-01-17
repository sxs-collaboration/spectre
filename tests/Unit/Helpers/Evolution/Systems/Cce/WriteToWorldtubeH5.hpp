// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Cce::TestHelpers {

// records worldtube data in the SpEC h5 style.
struct WorldtubeModeRecorder {
 public:
  WorldtubeModeRecorder(const size_t l_max, const std::string& filename)
      : l_max_(l_max), output_file_{filename, true} {
    // write the .ver that indicates that the derivatives are correctly
    // normalized.
    output_file_.try_insert<h5::Version>(
        "/VersionHist", "Bugfix in CCE radial derivatives (ticket 1096).");
    output_file_.close_current_object();
    all_modal_file_legend_.emplace_back("time");
    all_nodal_file_legend_.emplace_back("time");
    real_file_legend_.emplace_back("time");
    complex_nodal_legend_.emplace_back("time");

    // Modal legends
    for (int l = 0; l <= static_cast<int>(l_max_); ++l) {
      for (int m = -l; m <= l; ++m) {
        all_modal_file_legend_.push_back("Real Y_" + std::to_string(l) + "," +
                                         std::to_string(m));
        all_modal_file_legend_.push_back("Imag Y_" + std::to_string(l) + "," +
                                         std::to_string(m));

        if (m >= 0) {
          real_file_legend_.push_back("Real Y_" + std::to_string(l) + "," +
                                      std::to_string(m));
          if (m != 0) {
            real_file_legend_.push_back("Imag Y_" + std::to_string(l) + "," +
                                        std::to_string(m));
          }
        }
      }
    }

    const size_t num_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max_);

    // Nodal legends
    for (size_t i = 0; i < num_points; i++) {
      all_nodal_file_legend_.push_back("Point" + std::to_string(i));

      complex_nodal_legend_.push_back("Real Point" + std::to_string(i));
      complex_nodal_legend_.push_back("Imag Point" + std::to_string(i));
    }
  }

  // append to `dataset_path` the vector created by `time` followed by the
  // `modes` rearranged to be compatible with SpEC h5 format.
  void append_worldtube_mode_data(const std::string& dataset_path,
                                  const double time,
                                  const ComplexModalVector& modes,
                                  const bool is_spec_data = true,
                                  const bool is_real = false) {
    const size_t modal_size = square(l_max_ + 1);
    ASSERT(modes.size() == modal_size, "Expected modes of size "
                                           << modal_size << " but got "
                                           << modes.size() << " instead.");
    auto& output_mode_dataset = output_file_.try_insert<h5::Dat>(
        dataset_path, is_real ? real_file_legend_ : all_modal_file_legend_, 0);
    const size_t output_size = (is_real ? 1 : 2) * modal_size;
    std::vector<double> data_to_write(output_size + 1);
    data_to_write[0] = time;
    for (int l = 0; l <= static_cast<int>(l_max_); ++l) {
      for (int m = (is_real ? 0 : -l); m <= l; ++m) {
        const int em = is_spec_data ? -m : m;
        const size_t to_write_index =
            is_real ? static_cast<size_t>(m == 0 ? square(l) + 1
                                                 : square(l) + 2 * abs(m))
                    : (2 * Spectral::Swsh::goldberg_mode_index(
                               l_max_, static_cast<size_t>(l), em) +
                       1);
        const size_t mode_index = Spectral::Swsh::goldberg_mode_index(
            l_max_, static_cast<size_t>(l), m);
        data_to_write[to_write_index] = real(modes[mode_index]);
        if (not is_real or m != 0) {
          data_to_write[to_write_index + 1] = imag(modes[mode_index]);
        }
      }
    }

    output_mode_dataset.append(data_to_write);
    output_file_.close_current_object();
  }

  template <typename T>
  void append_worldtube_mode_data(const std::string& dataset_path,
                                  const double time, const T& nodes,
                                  const bool /*unused*/ = true) {
    static_assert(std::is_same_v<T, ComplexDataVector> or
                  std::is_same_v<T, DataVector>);
    constexpr bool is_complex = std::is_same_v<T, ComplexDataVector>;

    auto& output_mode_dataset = output_file_.try_insert<h5::Dat>(
        dataset_path,
        is_complex ? complex_nodal_legend_ : all_nodal_file_legend_, 0);
    const size_t output_size =
        (is_complex ? 2 : 1) *
        Spectral::Swsh::number_of_swsh_collocation_points(l_max_);

    std::vector<double> data_to_write(output_size + 1);
    data_to_write[0] = time;

    if ((is_complex ? 2 : 1) * nodes.size() != output_size) {
      ERROR("Trying to write test worldtube data. Data passed in has size "
            << (is_complex ? 2 : 1) * nodes.size() << " but expected size "
            << output_size);
    }
    for (size_t i = 0; i < nodes.size(); i++) {
      if constexpr (is_complex) {
        data_to_write[2 * i + 1] = real(nodes[i]);
        data_to_write[2 * i + 2] = imag(nodes[i]);
      } else {
        data_to_write[i + 1] = nodes[i];
      }
    }

    output_mode_dataset.append(data_to_write);
    output_file_.close_current_object();
  }

 private:
  size_t l_max_;
  h5::H5File<h5::AccessType::ReadWrite> output_file_;
  std::vector<std::string> all_modal_file_legend_;
  std::vector<std::string> all_nodal_file_legend_;
  std::vector<std::string> real_file_legend_;
  std::vector<std::string> complex_nodal_legend_;
};
}  // namespace Cce::TestHelpers
