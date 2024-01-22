// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "IO/Exporter/Exporter.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MakeArray.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  m.def(
      "interpolate_to_points",
      [](const std::variant<std::vector<std::string>, std::string>&
             volume_files_or_glob,
         const std::string& subfile_name, int observation_step,
         const std::vector<std::string>& tensor_components,
         std::vector<std::vector<double>> target_points,
         const std::optional<size_t>& num_threads) {
        const size_t dim = target_points.size();
        if (dim == 1) {
          return spectre::Exporter::interpolate_to_points(
              volume_files_or_glob, subfile_name, observation_step,
              tensor_components,
              make_array<std::vector<double>, 1>(std::move(target_points)),
              num_threads);
        } else if (dim == 2) {
          return spectre::Exporter::interpolate_to_points(
              volume_files_or_glob, subfile_name, observation_step,
              tensor_components,
              make_array<std::vector<double>, 2>(std::move(target_points)),
              num_threads);
        } else if (dim == 3) {
          return spectre::Exporter::interpolate_to_points(
              volume_files_or_glob, subfile_name, observation_step,
              tensor_components,
              make_array<std::vector<double>, 3>(std::move(target_points)),
              num_threads);
        } else {
          ERROR("Invalid dimension of target points: "
                << dim
                << ". Must be 1, 2, or 3. The first dimension of the "
                   "target points must the spatial dimension of the volume "
                   "data, and the second dimension is the number of points.");
        }
      },
      py::arg("volume_files_or_glob"), py::arg("subfile_name"),
      py::arg("observation_step"), py::arg("tensor_components"),
      py::arg("target_points"), py::arg("num_threads") = std::nullopt);
}
