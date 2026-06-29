// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/SphericalShells.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/SphericalShells.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_spherical_shells(py::module& m) {
  py::class_<SphericalShells, DomainCreator<3>>(m, "SphericalShells")
      .def(py::init<double, double, size_t, size_t, size_t, std::vector<double>,
                    const SphericalShells::RadialDistribution::type&>(),
           py::arg("inner_radius"), py::arg("outer_radius"),
           py::arg("initial_radial_refinement"),
           py::arg("initial_number_of_radial_grid_points"),
           py::arg("initial_spherical_harmonic_l"),
           py::arg("radial_partitioning") = std::vector<double>{},
           py::arg("radial_distribution") =
               SphericalShells::RadialDistribution::type{
                   domain::CoordinateMaps::Distribution::Linear});
}
}  // namespace domain::creators::py_bindings
