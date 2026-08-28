// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Python/SphericalShellPowerMonitor.hpp"

#include <array>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/ScalarWave/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/ScalarWave/SphericalShellPowerMonitor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace ScalarWave::power_monitor::py_bindings {
namespace {

using FuncOfTimeMap =
    std::unordered_map<std::string,
                       const domain::FunctionsOfTime::FunctionOfTime&>;

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
clone_functions_of_time(const FuncOfTimeMap& functions_of_time) {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time_ptrs{};
  for (const auto& [name, function_of_time] : functions_of_time) {
    functions_of_time_ptrs[name] = function_of_time.get_clone();
  }
  return functions_of_time_ptrs;
}

py::dict shell_power_monitor_to_dict(const std::array<DataVector, 2>& monitor) {
  py::dict result{};
  result["radial"] = monitor[0];
  result["angular"] = monitor[1];
  return result;
}

py::dict sw_shell_power_monitors_to_dict(const SwShellPowerMonitors& monitors) {
  py::dict result{};
  result["Psi"] = shell_power_monitor_to_dict(monitors.psi);
  result["Pi"] = shell_power_monitor_to_dict(monitors.pi);
  result["Phi"] = shell_power_monitor_to_dict(monitors.phi);
  return result;
}

InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
identity_grid_to_inertial_jacobian(const size_t size) {
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> jacobian{size,
                                                                        0.0};
  for (size_t i = 0; i < 3; ++i) {
    jacobian.get(i, i) = 1.0;
  }
  return jacobian;
}

InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
grid_to_inertial_jacobian(const Mesh<3>& mesh, const ElementId<3>& element_id,
                          const Domain<3>& domain, const double time,
                          const FuncOfTimeMap& functions_of_time) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto& block = domain.blocks()[element_id.block_id()];
  if (not block.is_time_dependent()) {
    return identity_grid_to_inertial_jacobian(number_of_grid_points);
  }

  auto functions_of_time_ptrs = clone_functions_of_time(functions_of_time);
  const auto element_logical_coords = logical_coordinates(mesh);
  const auto element_to_block_logical_map =
      domain::element_to_block_logical_map(element_id);
  const auto block_logical_coords =
      (*element_to_block_logical_map)(element_logical_coords);
  const auto grid_coords =
      block.moving_mesh_logical_to_grid_map()(block_logical_coords);
  return block.moving_mesh_grid_to_inertial_map().jacobian(
      grid_coords, time, functions_of_time_ptrs);
}

py::dict sw_shell_power_monitors_from_jacobian(
    const ScalarWave::Tags::Psi::type& psi,
    const ScalarWave::Tags::Pi::type& pi,
    const ScalarWave::Tags::Phi<3>::type& phi, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid) {
  Variables<ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>
      sw_vars{mesh.number_of_grid_points()};
  get<ScalarWave::Tags::Psi>(sw_vars) = psi;
  get<ScalarWave::Tags::Pi>(sw_vars) = pi;
  get<ScalarWave::Tags::Phi<3, Frame::Inertial>>(sw_vars) = phi;
  SwCartToSphereMatrix cart_to_sphere_matrix{};
  return sw_shell_power_monitors_to_dict(
      sw_shell_power_monitors(make_not_null(&cart_to_sphere_matrix), sw_vars,
                              mesh, jac_inertial_to_grid));
}

py::dict sw_shell_power_monitors_from_domain(
    const ScalarWave::Tags::Psi::type& psi,
    const ScalarWave::Tags::Pi::type& pi,
    const ScalarWave::Tags::Phi<3>::type& phi, const Mesh<3>& mesh,
    const ElementId<3>& element_id, const Domain<3>& domain, const double time,
    const FuncOfTimeMap& functions_of_time) {
  return sw_shell_power_monitors_from_jacobian(
      psi, pi, phi, mesh,
      grid_to_inertial_jacobian(mesh, element_id, domain, time,
                                functions_of_time));
}

}  // namespace

void bind_spherical_shell_power_monitor(py::module& m) {  // NOLINT
  m.def("sw_shell_power_monitors", &sw_shell_power_monitors_from_domain,
        py::arg("psi"), py::arg("pi"), py::arg("phi"), py::arg("mesh"),
        py::arg("element_id"), py::arg("domain"), py::arg("time"),
        py::arg("functions_of_time"));
}

}  // namespace ScalarWave::power_monitor::py_bindings
