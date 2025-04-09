// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/StrahlkorperTransformations.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace py = pybind11;

namespace domain::py_bindings {
namespace {
template <typename SrcFrame, typename DestFrame>
void bind_strahlkorper_transformations_impl(py::module& m) {  // NOLINT
  m.def(
      "strahlkorper_in_inertial_frame",
      [](const ylm::Strahlkorper<SrcFrame>& strahlkorper,
         const Domain<3>& domain,
         const std::optional<std::unordered_map<
             std::string, const domain::FunctionsOfTime::FunctionOfTime&>>&
             functions_of_time,
         const std::optional<double>& time) {
        // Transform functions-of-time map to unique_ptrs because pybind11 can't
        // handle unique_ptrs easily as function arguments (it's hard to
        // transfer ownership of a Python object to C++)
        std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
            functions_of_time_ptrs{};
        if (functions_of_time.has_value()) {
          for (const auto& [name, fot] : *functions_of_time) {
            functions_of_time_ptrs[name] = fot.get_clone();
          }
        }
        ylm::Strahlkorper<DestFrame> result{};
        strahlkorper_in_different_frame<SrcFrame, DestFrame>(
            make_not_null(&result), strahlkorper, domain,
            functions_of_time_ptrs,
            time.value_or(std::numeric_limits<double>::signaling_NaN()));
        return result;
      },
      py::arg("strahlkorper"), py::arg("domain"),
      py::arg("functions_of_time") = std::nullopt,
      py::arg("time") = std::nullopt);
}
}  // namespace

void bind_strahlkorper_transformations(py::module& m) {  // NOLINT
  bind_strahlkorper_transformations_impl<Frame::Grid, Frame::Inertial>(m);
  bind_strahlkorper_transformations_impl<Frame::Inertial, Frame::Grid>(m);
  bind_strahlkorper_transformations_impl<Frame::Inertial, Frame::Distorted>(m);
}

}  // namespace domain::py_bindings
