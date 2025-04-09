// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Python/Strahlkorper.hpp"

#include <array>
#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/StrahlkorperCoordsToTextFile.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
namespace {
template <typename Frame>
void bind_strahlkorper_impl(pybind11::module& m) {  // NOLINT
  using Strahlkorper = ylm::Strahlkorper<Frame>;
  py::class_<Strahlkorper>(m, ("Strahlkorper" + get_output(Frame{})).c_str())
      .def(py::init<size_t, double, std::array<double, 3>>(), py::arg("l_max"),
           py::arg("radius"), py::arg("center"))
      .def(py::init<size_t, size_t, const DataVector&, std::array<double, 3>>(),
           py::arg("l_max"), py::arg("m_max"),
           py::arg("radius_at_collocation_points"), py::arg("center"))
      .def(py::init<size_t, size_t, const Strahlkorper&>(), py::arg("l_max"),
           py::arg("m_max"), py::arg("another_strahlkorper"))
      .def(
          py::init<size_t, size_t, const ModalVector&, std::array<double, 3>>(),
          py::arg("l_max"), py::arg("m_max"), py::arg("spectral_coefficients"),
          py::arg("center"))
      .def_property_readonly("l_max", &Strahlkorper::l_max)
      .def_property_readonly("m_max", &Strahlkorper::m_max)
      .def_property_readonly(
          "physical_extents",
          [](const Strahlkorper& strahlkorper) {
            return strahlkorper.ylm_spherepack().physical_extents();
          })
      .def_property_readonly("expansion_center",
                             &Strahlkorper::expansion_center)
      .def_property_readonly("physical_center", &Strahlkorper::physical_center)
      .def_property_readonly("average_radius", &Strahlkorper::average_radius)
      .def("radius", &Strahlkorper::radius, py::arg("theta"), py::arg("phi"))
      .def("point_is_contained", &Strahlkorper::point_is_contained,
           py::arg("x"))
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);
  m.def(
      "ylm_legend_and_data",
      [](const Strahlkorper& strahlkorper, const double time,
         const size_t max_l)
          -> std::pair<std::vector<std::string>, std::vector<double>> {
        std::vector<std::string> legend{};
        std::vector<double> data{};
        ylm::fill_ylm_legend_and_data(make_not_null(&legend),
                                      make_not_null(&data), strahlkorper, time,
                                      max_l);
        return std::make_pair(legend, data);
      },
      py::arg("strahlkorper"), py::arg("time"), py::arg("max_l"));
}
}  // namespace

void bind_strahlkorper(py::module& m) {
  bind_strahlkorper_impl<Frame::Inertial>(m);
  bind_strahlkorper_impl<Frame::Grid>(m);
  bind_strahlkorper_impl<Frame::Distorted>(m);
  m.def("read_surface_ylm", &ylm::read_surface_ylm<Frame::Inertial>,
        py::arg("file_name"), py::arg("surface_subfile_name"),
        py::arg("requested_number_of_times_from_end"));
  m.def("read_surface_ylm_single_time",
        &ylm::read_surface_ylm_single_time<Frame::Inertial>,
        py::arg("file_name"), py::arg("surface_subfile_name"), py::arg("time"),
        py::arg("relative_epsilon"), py::arg("check_frame"));
  py::enum_<ylm::AngularOrdering>(m, "AngularOrdering")
      .value("Strahlkorper", ylm::AngularOrdering::Strahlkorper)
      .value("Cce", ylm::AngularOrdering::Cce);
  m.def(
      "write_sphere_of_points_to_text_file",
      [](const double radius, const size_t l_max,
         const std::array<double, 3>& center,
         const std::string& output_file_name,
         const ylm::AngularOrdering ordering, const bool overwrite_file) {
        ylm::write_strahlkorper_coords_to_text_file(
            radius, l_max, center, output_file_name, ordering, overwrite_file);
      },
      py::arg("radius"), py::arg("l_max"), py::arg("center"),
      py::arg("output_file_name"), py::arg("ordering"),
      py::arg("overwrite_file") = false);
}
}  // namespace ylm::py_bindings
