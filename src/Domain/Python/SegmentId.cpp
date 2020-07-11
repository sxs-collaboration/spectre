// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Structure/SegmentId.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

void bind_segment_id(py::module& m) {  // NOLINT
  // These bindings don't cover the full public interface yet. More bindings
  // can be added as needed.
  py::class_<SegmentId>(m, "SegmentId")
      .def(py::init<size_t, size_t>(), py::arg("refinement_level"),
           py::arg("index"))
      .def_property("refinement_level", &SegmentId::refinement_level, nullptr)
      .def_property("index", &SegmentId::index, nullptr)
      .def("__repr__",
           [](const SegmentId& segment_id) { return get_output(segment_id); })
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);
}

}  // namespace domain::py_bindings
