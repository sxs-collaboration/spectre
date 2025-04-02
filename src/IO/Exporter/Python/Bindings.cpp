// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MakeArray.hpp"

namespace py = pybind11;

namespace {

template <size_t Dim, typename Frame>
void bind_interpolate_to_points_impl(py::module& m) {
  m.def(
      "interpolate_to_points",
      [](const std::variant<std::vector<std::string>, std::string>&
             volume_files_or_glob,
         const std::string& subfile_name, const size_t observation_id,
         const std::vector<std::string>& tensor_components,
         const tnsr::I<DataVector, Dim, Frame>& target_points,
         const bool extrapolate_into_excisions,
         const std::optional<size_t>& num_threads) {
        return spectre::Exporter::interpolate_to_points(
            volume_files_or_glob, subfile_name,
            spectre::Exporter::ObservationId{observation_id}, tensor_components,
            target_points, extrapolate_into_excisions, num_threads);
      },
      py::arg("volume_files_or_glob"), py::arg("subfile_name"),
      py::arg("observation_id"), py::arg("tensor_components"),
      py::arg("target_points"), py::arg("extrapolate_into_excisions") = false,
      py::arg("num_threads") = std::nullopt);
}

}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures.Tensor");
  bind_interpolate_to_points_impl<1, Frame::Grid>(m);
  bind_interpolate_to_points_impl<2, Frame::Grid>(m);
  bind_interpolate_to_points_impl<3, Frame::Grid>(m);
  bind_interpolate_to_points_impl<1, Frame::Inertial>(m);
  bind_interpolate_to_points_impl<2, Frame::Inertial>(m);
  bind_interpolate_to_points_impl<3, Frame::Inertial>(m);
}
