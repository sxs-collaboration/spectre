// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Python/CoordinateMap.hpp"

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
using FuncOfTimeMap =
    std::unordered_map<std::string,
                       const domain::FunctionsOfTime::FunctionOfTime&>;

// Transform functions-of-time map to unique_ptrs because pybind11
// can't handle them easily as function arguments (it's hard to
// transfer ownership of a Python object to C++)
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
transform_functions_of_time(
    const std::optional<FuncOfTimeMap>& functions_of_time) {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time_ptrs{};
  if (functions_of_time.has_value()) {
    for (const auto& [name, fot] : *functions_of_time) {
      functions_of_time_ptrs[name] = fot.get_clone();
    }
  }
  return functions_of_time_ptrs;
}

template <typename SourceFrame, typename TargetFrame, size_t Dim>
void bind_coordinate_map_impl(py::module& m) {  // NOLINT
  using CoordMapType = CoordinateMapBase<SourceFrame, TargetFrame, Dim>;
  py::class_<CoordMapType>(m,
                           ("CoordinateMap" + get_output(SourceFrame{}) + "To" +
                            get_output(TargetFrame{}) + get_output(Dim) + "D")
                               .c_str())
      .def("is_identity", &CoordMapType::is_identity)
      .def("inv_jacobian_is_time_dependent",
           &CoordMapType::inv_jacobian_is_time_dependent)
      .def("jacobian_is_time_dependent",
           &CoordMapType::jacobian_is_time_dependent)
      .def(
          "__call__",
          [](const CoordMapType& map,
             const tnsr::I<DataVector, Dim, SourceFrame>& source_point,
             const std::optional<double> time,
             const std::optional<FuncOfTimeMap>& functions_of_time) {
            return map(
                source_point,
                time.value_or(std::numeric_limits<double>::signaling_NaN()),
                transform_functions_of_time(functions_of_time));
          },
          py::arg("source_point"), py::arg("time") = std::nullopt,
          py::arg("functions_of_time") = std::nullopt)
      .def(
          "inverse",
          [](const CoordMapType& map,
             const tnsr::I<double, Dim, TargetFrame>& source_point,
             const std::optional<double> time,
             const std::optional<FuncOfTimeMap>& functions_of_time) {
            return map.inverse(
                source_point,
                time.value_or(std::numeric_limits<double>::signaling_NaN()),
                transform_functions_of_time(functions_of_time));
          },
          py::arg("source_point"), py::arg("time") = std::nullopt,
          py::arg("functions_of_time") = std::nullopt)
      .def(
          "inv_jacobian",
          [](const CoordMapType& map,
             const tnsr::I<DataVector, Dim, SourceFrame>& source_point,
             const std::optional<double> time,
             const std::optional<FuncOfTimeMap>& functions_of_time) {
            return map.inv_jacobian(
                source_point,
                time.value_or(std::numeric_limits<double>::signaling_NaN()),
                transform_functions_of_time(functions_of_time));
          },
          py::arg("source_point"), py::arg("time") = std::nullopt,
          py::arg("functions_of_time") = std::nullopt)
      .def(
          "jacobian",
          [](const CoordMapType& map,
             const tnsr::I<DataVector, Dim, SourceFrame>& source_point,
             const std::optional<double> time,
             const std::optional<FuncOfTimeMap>& functions_of_time) {
            return map.jacobian(
                source_point,
                time.value_or(std::numeric_limits<double>::signaling_NaN()),
                transform_functions_of_time(functions_of_time));
          },
          py::arg("source_point"), py::arg("time") = std::nullopt,
          py::arg("functions_of_time") = std::nullopt);
}
}  // namespace

void bind_coordinate_map(py::module& m) {  // NOLINT
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  bind_coordinate_map_impl<Frame::ElementLogical, Frame::BlockLogical,         \
                           DIM(data)>(m);                                      \
  bind_coordinate_map_impl<Frame::ElementLogical, Frame::Inertial, DIM(data)>( \
      m);                                                                      \
  bind_coordinate_map_impl<Frame::BlockLogical, Frame::Grid, DIM(data)>(m);    \
  bind_coordinate_map_impl<Frame::BlockLogical, Frame::Inertial, DIM(data)>(   \
      m);                                                                      \
  bind_coordinate_map_impl<Frame::Grid, Frame::Distorted, DIM(data)>(m);       \
  bind_coordinate_map_impl<Frame::Grid, Frame::Inertial, DIM(data)>(m);        \
  bind_coordinate_map_impl<Frame::Distorted, Frame::Inertial, DIM(data)>(m);

  GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}
}  // namespace domain::py_bindings
