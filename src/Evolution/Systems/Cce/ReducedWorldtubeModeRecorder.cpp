// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/ReducedWorldtubeModeRecorder.hpp"

#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "Utilities/ForceInline.hpp"

namespace Cce {

void ReducedWorldtubeModeRecorder::append_worldtube_mode_data(
    const std::string& dataset_path, const double time,
    const ComplexModalVector& modes, const size_t l_max, const bool is_real) {
  std::vector<std::string> legend;
  const size_t output_size = square(l_max + 1);
  legend.reserve(is_real ? output_size + 1 : 2 * output_size + 1);
  legend.emplace_back("time");
  for (int l = 0; l <= static_cast<int>(l_max); ++l) {
    for (int m = is_real ? 0 : -l; m <= l; ++m) {
      legend.push_back("Re(" + std::to_string(l) + "," + std::to_string(m) +
                       ")");
      if (LIKELY(not is_real or m != 0)) {
        legend.push_back("Im(" + std::to_string(l) + "," + std::to_string(m) +
                         ")");
      }
    }
  }
  auto& output_mode_dataset =
      output_file_.try_insert<h5::Dat>(dataset_path, legend, 0);
  std::vector<double> data_to_write;
  if (is_real) {
    data_to_write.resize(output_size + 1);
    data_to_write[0] = time;
    for (int l = 0; l <= static_cast<int>(l_max); ++l) {
      data_to_write[static_cast<size_t>(square(l)) + 1] =
          real(modes[Spectral::Swsh::goldberg_mode_index(
              l_max, static_cast<size_t>(l), 0)]);
      for (int m = 1; m <= l; ++m) {
        // this is the right order of the casts, other orders give the wrong
        // answer
        // NOLINTNEXTLINE(misc-misplaced-widening-cast)
        data_to_write[static_cast<size_t>(square(l) + 2 * m)] =
            real(modes[Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m)]);
        // this is the right order of the casts, other orders give the wrong
        // answer
        // NOLINTNEXTLINE(misc-misplaced-widening-cast)
        data_to_write[static_cast<size_t>(square(l) + 2 * m + 1)] =
            imag(modes[Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m)]);
      }
    }
  } else {
    data_to_write.resize(2 * output_size + 1);
    data_to_write[0] = time;
    for (int l = 0; l <= static_cast<int>(l_max); ++l) {
      for (int m = -l; m <= l; ++m) {
        data_to_write[2 * Spectral::Swsh::goldberg_mode_index(
                              l_max, static_cast<size_t>(l), m) +
                      1] =
            real(modes[Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m)]);
        data_to_write[2 * Spectral::Swsh::goldberg_mode_index(
                              l_max, static_cast<size_t>(l), m) +
                      2] =
            imag(modes[Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m)]);
      }
    }
  }
  output_mode_dataset.append(data_to_write);
  output_file_.close_current_object();
}
}  // namespace Cce
