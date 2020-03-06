// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/ElementId.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain {
namespace py_bindings {

namespace detail {
template <size_t Dim>
void bind_element_id_impl(py::module& m) {  // NOLINT
  // These bindings don't cover the full public interface yet. More bindings
  // can be added as needed.
  py::class_<ElementId<Dim>>(m, ("ElementId" + get_output(Dim) + "D").c_str())
      .def(py::init<size_t>(), py::arg("block_id"))
      .def(py::init<size_t, std::array<SegmentId, Dim>>(), py::arg("block_id"),
           py::arg("segment_ids"))
      .def_property("block_id", &ElementId<Dim>::block_id, nullptr)
      .def_property("segment_ids", &ElementId<Dim>::segment_ids, nullptr)
      .def_static("external_boundary_id", &ElementId<Dim>::external_boundary_id)
      .def("__repr__",
           [](const ElementId<Dim>& element_id) {
             return get_output(element_id);
           })
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);
}
}  // namespace detail

void bind_element_id(py::module& m) {  // NOLINT
  detail::bind_element_id_impl<1>(m);
  detail::bind_element_id_impl<2>(m);
  detail::bind_element_id_impl<3>(m);
}

}  // namespace py_bindings
}  // namespace domain
