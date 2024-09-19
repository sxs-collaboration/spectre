// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/SurfaceFinder/SurfaceFinder.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace SurfaceFinder {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.SphericalHarmonics");
  m.def("find_radial_surface", &find_radial_surface, py::arg("data"),
        py::arg("target"), py::arg("mesh"), py::arg("angular_coords"),
        py::arg("relative_tolerance") = 1e-10,
        py::arg("absolute_tolerance") = 1e-10);
}

}  // namespace SurfaceFinder
