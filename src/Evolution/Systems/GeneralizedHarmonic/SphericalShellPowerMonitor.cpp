// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/SphericalShellPowerMonitor.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TensorYlmTransforms.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/TensorYlm/CartToSphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::power_monitor {
namespace {

constexpr size_t radial_monitor_index = 0;
constexpr size_t angular_monitor_index = 1;

}  // namespace

void fill_cart_to_sphere_matrices(
    const gsl::not_null<CartToSphereMatrices*> matrices, const size_t ell_max) {
  if (not matrices->i.has_value()) {
    matrices->i.emplace();
    ylm::TensorYlm::fill_cart_to_sphere<
        typename tnsr::i<DataVector, 3>::structure>(
        make_not_null(&matrices->i.value()), ell_max,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
  }
  if (not matrices->ii.has_value()) {
    matrices->ii.emplace();
    ylm::TensorYlm::fill_cart_to_sphere<
        typename tnsr::ii<DataVector, 3>::structure>(
        make_not_null(&matrices->ii.value()), ell_max,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
  }
  if (not matrices->ij.has_value()) {
    matrices->ij.emplace();
    ylm::TensorYlm::fill_cart_to_sphere<
        typename tnsr::ij<DataVector, 3>::structure>(
        make_not_null(&matrices->ij.value()), ell_max,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
  }
  if (not matrices->ijj.has_value()) {
    matrices->ijj.emplace();
    ylm::TensorYlm::fill_cart_to_sphere<
        typename tnsr::ijj<DataVector, 3>::structure>(
        make_not_null(&matrices->ijj.value()), ell_max,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
  }
}

GhShellPowerMonitors gh_shell_power_monitors(
    const gsl::not_null<CartToSphereMatrices*> cart_to_sphere_matrices,
    const Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list>&
        gh_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid) {
  if (mesh.basis(0) == Spectral::Basis::SphericalHarmonic or
      mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(2) != Spectral::Basis::SphericalHarmonic) {
    ERROR(
        "GH spherical-shell power monitors require the mesh dimensions to "
        "be ordered (radial, theta, phi), but the mesh is "
        << mesh);
  }
  const size_t radial_extents = mesh.extents(0);
  const size_t ell_max = mesh.extents(1) - 1;
  const size_t m_max = (mesh.extents(2) - 1) / 2;
  if (ell_max != m_max) {
    ERROR(
        "GH spherical-shell power monitors require l_max == m_max, but got "
        "l_max = "
        << ell_max << " and m_max = " << m_max << ".");
  }
  const auto& spherepack = ylm::get_spherepack_cache(ell_max);

  GhShellPowerMonitors result{};
  result.spacetime_metric[radial_monitor_index] =
      PowerMonitors::spherical_shell_tensor_radial_power_monitor(
          get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
              gh_vars),
          mesh);
  result.pi[radial_monitor_index] =
      PowerMonitors::spherical_shell_tensor_radial_power_monitor(
          get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(gh_vars), mesh);
  result.phi[radial_monitor_index] =
      PowerMonitors::spherical_shell_tensor_radial_power_monitor(
          get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(gh_vars), mesh);

  fill_cart_to_sphere_matrices(cart_to_sphere_matrices, ell_max);
  Variables<ylm::TensorYlm::filter_detail::gh_spatial_vars_list<Frame::Grid>>
      gh_spatial_tensor_ylm_coefficients(radial_extents *
                                         spherepack.spectral_size());
  Variables<ylm::TensorYlm::filter_detail::gh_spatial_vars_list<Frame::Grid>>
      temp_storage(radial_extents * spherepack.spectral_size());
  ylm::TensorYlm::gh_variables_to_tensor_ylm_coefficients(
      make_not_null(&gh_spatial_tensor_ylm_coefficients),
      make_not_null(&temp_storage), gh_vars, jac_inertial_to_grid,
      cart_to_sphere_matrices->i.value(), cart_to_sphere_matrices->ii.value(),
      cart_to_sphere_matrices->ij.value(), cart_to_sphere_matrices->ijj.value(),
      spherepack, radial_extents);

  result.spacetime_metric[angular_monitor_index] = DataVector(ell_max + 1, 0.0);
  result.pi[angular_monitor_index] = DataVector(ell_max + 1, 0.0);
  result.phi[angular_monitor_index] = DataVector(ell_max + 1, 0.0);
  std::vector<size_t> metric_counts(ell_max + 1, 0);
  std::vector<size_t> pi_counts(ell_max + 1, 0);
  std::vector<size_t> phi_counts(ell_max + 1, 0);

  namespace detail = ylm::TensorYlm::filter_detail;
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      make_not_null(&metric_counts),
      get<detail::Tags::Metric00<DataVector>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      make_not_null(&metric_counts),
      get<detail::Tags::Metrick0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      make_not_null(&metric_counts),
      get<detail::Tags::Metrickj<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);

  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts),
      get<detail::Tags::Pi00<DataVector>>(gh_spatial_tensor_ylm_coefficients),
      mesh);
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts),
      get<detail::Tags::Pik0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts),
      get<detail::Tags::Pikj<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);

  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<detail::Tags::Phik00<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<detail::Tags::Phiki0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  PowerMonitors::accumulate_spherical_shell_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<detail::Tags::Phikij<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);

  PowerMonitors::normalize_spherical_shell_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      metric_counts);
  PowerMonitors::normalize_spherical_shell_angular_power(
      make_not_null(&result.pi[angular_monitor_index]), pi_counts);
  PowerMonitors::normalize_spherical_shell_angular_power(
      make_not_null(&result.phi[angular_monitor_index]), phi_counts);
  return result;
}

}  // namespace gh::power_monitor
