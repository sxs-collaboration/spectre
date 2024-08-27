// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/Rectilinear.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
template <size_t Dim>
void bind_rectilinear_impl(py::module& m) {
  py::class_<Rectilinear<Dim>, DomainCreator<Dim>>(
      m, Rectilinear<Dim>::name().c_str())
      .def(py::init<
               std::array<double, Dim>, std::array<double, Dim>,
               std::array<size_t, Dim>, std::array<size_t, Dim>,
               std::array<bool, Dim>,
               std::array<CoordinateMaps::DistributionAndSingularityPosition,
                          Dim>>(),
           py::arg("lower_bounds"), py::arg("upper_bounds"),
           py::arg("initial_refinement_levels"), py::arg("initial_num_points"),
           py::arg("is_periodic") = make_array<Dim>(false),
           py::arg("distribution") =
               std::array<CoordinateMaps::DistributionAndSingularityPosition,
                          Dim>{});
}

void bind_rectilinear(py::module& m) {
  bind_rectilinear_impl<1>(m);
  bind_rectilinear_impl<2>(m);
  bind_rectilinear_impl<3>(m);
}
}  // namespace domain::creators::py_bindings
