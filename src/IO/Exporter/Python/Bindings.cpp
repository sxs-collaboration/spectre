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
  m.def("interpolate_to_points",
        py::overload_cast<
            const std::variant<std::vector<std::string>, std::string>&,
            const std::string&, const spectre::Exporter::ObservationVariant&,
            const std::vector<std::string>&,
            const tnsr::I<DataVector, Dim, Frame>&, bool, bool,
            std::optional<size_t>>(
            &spectre::Exporter::interpolate_to_points<Dim, Frame>),
      py::arg("volume_files_or_glob"), py::arg("subfile_name"),
        py::arg("observation"), py::arg("tensor_components"),
      py::arg("target_points"), py::arg("extrapolate_into_excisions") = false,
      py::arg("error_on_missing_points") = false,
      py::arg("num_threads") = std::nullopt);
}

}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures.Tensor");
  py::class_<spectre::Exporter::ObservationId>(m, "ObservationId")
      .def(py::init<size_t>())
      .def_readonly("value", &spectre::Exporter::ObservationId::value);
  py::class_<spectre::Exporter::ObservationStep>(m, "ObservationStep")
      .def(py::init<int>())
      .def_readonly("value", &spectre::Exporter::ObservationStep::value);
  bind_interpolate_to_points_impl<1, Frame::Grid>(m);
  bind_interpolate_to_points_impl<2, Frame::Grid>(m);
  bind_interpolate_to_points_impl<3, Frame::Grid>(m);
  bind_interpolate_to_points_impl<1, Frame::Inertial>(m);
  bind_interpolate_to_points_impl<2, Frame::Inertial>(m);
  bind_interpolate_to_points_impl<3, Frame::Inertial>(m);
}
