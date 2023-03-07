// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/RadiallyCompressedCoordinates.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Utilities/Gsl.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim, typename CoordsFrame>
void bind_radially_compressed_coordinates_impl(py::module& m) {  // NOLINT
  m.def("radially_compressed_coordinates",
        static_cast<tnsr::I<DataVector, Dim, CoordsFrame> (*)(
            const tnsr::I<DataVector, Dim, CoordsFrame>&, double, double,
            CoordinateMaps::Distribution)>(
            &radially_compressed_coordinates<DataVector, Dim, CoordsFrame>),
        py::arg(std::is_same_v<CoordsFrame, Frame::Inertial> ? "inertial_coords"
                                                             : "grid_coords"),
        py::arg("inner_radius"), py::arg("outer_radius"),
        py::arg("compression"));
}
}  // namespace

void bind_radially_compressed_coordinates(py::module& m) {  // NOLINT
  bind_radially_compressed_coordinates_impl<1, Frame::Inertial>(m);
  bind_radially_compressed_coordinates_impl<2, Frame::Inertial>(m);
  bind_radially_compressed_coordinates_impl<3, Frame::Inertial>(m);
}

}  // namespace domain::py_bindings
