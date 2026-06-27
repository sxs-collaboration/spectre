// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/StrahlkorperTransformations.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"

namespace py = pybind11;

namespace domain::py_bindings {
namespace {

using PyFunctionsOfTimeMap =
    std::unordered_map<std::string,
                       const domain::FunctionsOfTime::FunctionOfTime&>;

// Transform functions-of-time map to unique_ptrs because pybind11
// can't handle them easily as function arguments (it's hard to
// transfer ownership of a Python object to C++)
domain::FunctionsOfTimeMap transform_functions_of_time(
    const std::optional<PyFunctionsOfTimeMap>& functions_of_time) {
  domain::FunctionsOfTimeMap functions_of_time_ptrs{};
  if (functions_of_time.has_value()) {
    for (const auto& [name, fot] : *functions_of_time) {
      functions_of_time_ptrs[name] = fot.get_clone();
    }
  }
  return functions_of_time_ptrs;
}

template <typename SrcFrame, typename DestFrame>
void bind_strahlkorper_transformations_impl(py::module& m) {  // NOLINT
  m.def(
      "strahlkorper_in_inertial_frame",
      [](const ylm::Strahlkorper<SrcFrame>& strahlkorper,
         const Domain<3>& domain,
         const std::optional<PyFunctionsOfTimeMap>& functions_of_time,
         const std::optional<double>& time) {
        ylm::Strahlkorper<DestFrame> result{};
        strahlkorper_in_different_frame<SrcFrame, DestFrame>(
            make_not_null(&result), strahlkorper, domain,
            transform_functions_of_time(functions_of_time),
            time.value_or(std::numeric_limits<double>::signaling_NaN()));
        return result;
      },
      py::arg("strahlkorper"), py::arg("domain"),
      py::arg("functions_of_time") = std::nullopt,
      py::arg("time") = std::nullopt);
  m.def(
      "strahlkorper_in_inertial_frame_aligned",
      [](const ylm::Strahlkorper<SrcFrame>& strahlkorper,
         const Domain<3>& domain,
         const std::optional<PyFunctionsOfTimeMap>& functions_of_time,
         const std::optional<double>& time) {
        ylm::Strahlkorper<DestFrame> result{};
        strahlkorper_in_different_frame_aligned<SrcFrame, DestFrame>(
            make_not_null(&result), strahlkorper, domain,
            transform_functions_of_time(functions_of_time),
            time.value_or(std::numeric_limits<double>::signaling_NaN()));
        return result;
      },
      py::arg("strahlkorper"), py::arg("domain"),
      py::arg("functions_of_time") = std::nullopt,
      py::arg("time") = std::nullopt);
}
}  // namespace

void bind_strahlkorper_transformations(py::module& m) {  // NOLINT
  // Only instantiating for Grid->Inertial because the Py functions are
  // currently named like that
  bind_strahlkorper_transformations_impl<Frame::Grid, Frame::Inertial>(m);
}

}  // namespace domain::py_bindings
