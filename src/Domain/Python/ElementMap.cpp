// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/ElementMap.hpp"

#include <array>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {

template <size_t Dim>
void bind_element_map_impl(py::module& m) {  // NOLINT
  // We construct a `Composition` coordinate map here instead of the
  // `ElementMap` class so we can construct a full time-dependent ElementLogical
  // -> Inertial map. We could actually replace the `ElementMap` class
  // throughout the code with a `Composition`, since `ElementMap` is just a
  // composition in disguise. Note that in evolutions we want to keep the
  // (static) ElementLogical -> Grid part of the composition separate from the
  // (time-dependent) Grid -> Inertial part to avoid reevaluating the static
  // part at every time step. This probably isn't important in Python, but could
  // be done as well.
  //
  // We return the type-erased base class `CoordinateMapBase` here, but could
  // also bind and return the `Composition` class if we need to in the future
  // (e.g. if we want access to parts of the composition in Python).
  m.def(
      "ElementMap",
      [](const ElementId<Dim>& element_id, const Domain<Dim>& domain)
          -> std::unique_ptr<
              CoordinateMapBase<Frame::ElementLogical, Frame::Inertial, Dim>> {
        const auto& block = domain.blocks()[element_id.block_id()];
        // ElementLogical -> BlockLogical
        auto element_to_block_logical_map =
            domain::element_to_block_logical_map(element_id);
        // BlockLogical -> Inertial
        if (block.is_time_dependent()) {
          if (block.has_distorted_frame()) {
            using CompositionType = CoordinateMaps::Composition<
                tmpl::list<Frame::ElementLogical, Frame::BlockLogical,
                           Frame::Grid, Frame::Distorted, Frame::Inertial>,
                Dim>;
            return std::make_unique<CompositionType>(
                std::move(element_to_block_logical_map),
                block.moving_mesh_logical_to_grid_map().get_clone(),
                block.moving_mesh_grid_to_distorted_map().get_clone(),
                block.moving_mesh_distorted_to_inertial_map().get_clone());
          } else {
            using CompositionType = CoordinateMaps::Composition<
                tmpl::list<Frame::ElementLogical, Frame::BlockLogical,
                           Frame::Grid, Frame::Inertial>,
                Dim>;
            return std::make_unique<CompositionType>(
                std::move(element_to_block_logical_map),
                block.moving_mesh_logical_to_grid_map().get_clone(),
                block.moving_mesh_grid_to_inertial_map().get_clone());
          }
        } else {
          using CompositionType = CoordinateMaps::Composition<
              tmpl::list<Frame::ElementLogical, Frame::BlockLogical,
                         Frame::Inertial>,
              Dim>;
          return std::make_unique<CompositionType>(
              std::move(element_to_block_logical_map),
              block.stationary_map().get_clone());
        }
      },
      py::arg("element_id"), py::arg("domain"));
}
}  // namespace

void bind_element_map(py::module& m) {  // NOLINT
  bind_element_map_impl<1>(m);
  bind_element_map_impl<2>(m);
  bind_element_map_impl<3>(m);
}
}  // namespace domain::py_bindings
