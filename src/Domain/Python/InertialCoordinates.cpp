// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/InertialCoordinates.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim>
void bind_inertial_coordinates_impl(py::module& m) {  // NOLINT
  m.def(
      "inertial_coordinates",
      [](const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
             element_logical_coords,
         const ElementId<Dim>& element_id, const Domain<Dim>& domain,
         const std::optional<double> time = std::nullopt,
         const std::optional<std::unordered_map<
             std::string, const domain::FunctionsOfTime::FunctionOfTime&>>&
             functions_of_time = std::nullopt) {
        // Map logical to inertial coords
        const auto& block = domain.blocks()[element_id.block_id()];
        if (block.is_time_dependent()) {
          if (not time.has_value()) {
            throw std::invalid_argument(
                "The 'time' argument is required for time-dependent domains.");
          }
          if (not functions_of_time.has_value()) {
            throw std::invalid_argument(
                "The 'functions_of_time' argument is required for "
                "time-dependent domains.");
          }
          ASSERT(not block.has_distorted_frame(),
                 "Not implemented for blocks with distorted-frame maps. Add "
                 "support if you need it.");
          // - Logical to grid
          const ElementMap<Dim, Frame::Grid> element_map{
              element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
          const auto grid_coords = element_map(element_logical_coords);
          // - Grid to inertial
          // Transform functions-of-time map to unique_ptrs because pybind11
          // can't handle them easily as function arguments (it's hard to
          // transfer ownership of a Python object to C++)
          std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
              functions_of_time_ptrs{};
          for (const auto& [name, fot] : *functions_of_time) {
            functions_of_time_ptrs[name] = fot.get_clone();
          }
          return block.moving_mesh_grid_to_inertial_map()(
              grid_coords, *time, functions_of_time_ptrs);
        } else {
          const ElementMap<Dim, Frame::Inertial> element_map{
              element_id, block.stationary_map().get_clone()};
          return element_map(element_logical_coords);
        }
      },
      py::arg("element_logical_coords"), py::arg("element_id"),
      py::arg("domain"), py::arg("time"), py::arg("functions_of_time"));
}
}  // namespace

void bind_inertial_coordinates(py::module& m) {  // NOLINT
  bind_inertial_coordinates_impl<1>(m);
  bind_inertial_coordinates_impl<2>(m);
  bind_inertial_coordinates_impl<3>(m);
}

}  // namespace domain::py_bindings
