// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/Exporter/SpacetimeInterpolator.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
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

template <size_t Dim, typename Frame>
void bind_pointwise_interpolator_impl(py::module& m) {
  using PointwiseInterpolator =
      spectre::Exporter::PointwiseInterpolator<Dim, Frame>;
  py::class_<PointwiseInterpolator>(
      m, ("PointwiseInterpolator" + std::to_string(Dim) + "D" +
          get_output(Frame{}))
             .c_str())
      .def(py::init<const std::variant<std::vector<std::string>, std::string>&,
                    const std::string&,
                    const spectre::Exporter::ObservationVariant&,
                    const std::vector<std::string>&>(),
           py::arg("volume_files_or_glob"), py::arg("subfile_name"),
           py::arg("observation"), py::arg("tensor_components"))
      .def("obs_id", &PointwiseInterpolator::obs_id)
      .def("time", &PointwiseInterpolator::time)
      .def("domain", &PointwiseInterpolator::domain)
      .def("functions_of_time",
           [](const PointwiseInterpolator& self) {
             return clone_unique_ptrs(self.functions_of_time());
           })
      .def(
          "interpolate_to_point",
          [](const PointwiseInterpolator& self,
             const tnsr::I<double, Dim, Frame>& target_point) {
            std::vector<double> result{};
            self.interpolate_to_point(make_not_null(&result), target_point);
            return result;
          },
          py::arg("target_point"));
}

template <size_t Dim>
void bind_spacetime_interpolator_impl(py::module& m) {
  using SpacetimeInterpolator =
      spectre::Exporter::SpacetimeInterpolator<Dim, Frame::Inertial>;
  py::class_<SpacetimeInterpolator>(
      m, ("SpacetimeInterpolator" + std::to_string(Dim) + "D").c_str())
      .def(py::init<std::variant<std::vector<std::string>, std::string>,
                    std::string, std::vector<std::string>>(),
           py::arg("volume_files_or_glob"), py::arg("subfile_name"),
           py::arg("tensor_components"))
      .def("max_time_bounds", &SpacetimeInterpolator::max_time_bounds)
      .def("load_time_bounds", &SpacetimeInterpolator::load_time_bounds,
           py::arg("time_bounds"))
      .def("time_bounds", &SpacetimeInterpolator::time_bounds)
      .def(
          "interpolate_to_point",
          [](const SpacetimeInterpolator& self,
             const tnsr::I<double, Dim, Frame::Inertial>& target_point,
             const double time) {
            std::vector<double> result{};
            self.interpolate_to_point(make_not_null(&result), target_point,
                                      time);
            return result;
          },
          py::arg("target_point"), py::arg("time"));
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
  py::class_<spectre::Exporter::ObservationValue>(m, "ObservationValue")
      .def(py::init<double, double>(), py::arg("value"),
           py::arg("epsilon") = 1e-12)
      .def_readonly("value", &spectre::Exporter::ObservationValue::value)
      .def_readonly("epsilon", &spectre::Exporter::ObservationValue::epsilon);
  bind_interpolate_to_points_impl<1, Frame::Grid>(m);
  bind_interpolate_to_points_impl<2, Frame::Grid>(m);
  bind_interpolate_to_points_impl<3, Frame::Grid>(m);
  bind_interpolate_to_points_impl<1, Frame::Inertial>(m);
  bind_interpolate_to_points_impl<2, Frame::Inertial>(m);
  bind_interpolate_to_points_impl<3, Frame::Inertial>(m);
  bind_pointwise_interpolator_impl<1, Frame::Grid>(m);
  bind_pointwise_interpolator_impl<2, Frame::Grid>(m);
  bind_pointwise_interpolator_impl<3, Frame::Grid>(m);
  bind_pointwise_interpolator_impl<1, Frame::Inertial>(m);
  bind_pointwise_interpolator_impl<2, Frame::Inertial>(m);
  bind_pointwise_interpolator_impl<3, Frame::Inertial>(m);
  bind_spacetime_interpolator_impl<1>(m);
  bind_spacetime_interpolator_impl<2>(m);
  bind_spacetime_interpolator_impl<3>(m);
}
