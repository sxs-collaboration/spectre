// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/BlockLogicalCoordinates.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim>
void bind_block_logical_coordinates_impl(py::module& m) {  // NOLINT
  py::class_<
      IdPair<domain::BlockId, tnsr::I<double, Dim, Frame::BlockLogical>>>(
      m, ("BlockIdAndLogicalCoord" + get_output(Dim) + "D").c_str());
  m.def(
      "block_logical_coordinates",
      [](const Domain<Dim>& domain,
         const tnsr::I<DataVector, Dim>& inertial_coords,
         const std::optional<double>& time = std::nullopt,
         const std::optional<std::unordered_map<
             std::string, const domain::FunctionsOfTime::FunctionOfTime&>>&
             functions_of_time = std::nullopt) {
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
        return block_logical_coordinates(
            domain, inertial_coords,
            time.value_or(std::numeric_limits<double>::signaling_NaN()),
            functions_of_time_ptrs);
      },
      py::arg("domain"), py::arg("inertial_coords"), py::arg("time"),
      py::arg("functions_of_time"));
}
}  // namespace

void bind_block_logical_coordinates(py::module& m) {  // NOLINT
  py::class_<BlockId>(m, "BlockId").def("get_index", &BlockId::get_index);
  bind_block_logical_coordinates_impl<1>(m);
  bind_block_logical_coordinates_impl<2>(m);
  bind_block_logical_coordinates_impl<3>(m);
}

}  // namespace domain::py_bindings
