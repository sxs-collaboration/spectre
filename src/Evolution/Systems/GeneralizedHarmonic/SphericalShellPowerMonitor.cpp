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
#include "NumericalAlgorithms/TensorYlm/Helpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::power_monitor {
namespace {

constexpr size_t radial_monitor_index = 0;
constexpr size_t angular_monitor_index = 1;

template <typename TensorType>
DataVector tensor_radial_power_monitor(const TensorType& tensor,
                                       const Mesh<3>& mesh) {
  DataVector squared_power(mesh.extents(0), 0.0);
  DataVector component_power{};
  for (size_t component = 0; component < tensor.size(); ++component) {
    PowerMonitors::spherical_shell_radial_power_monitor(
        make_not_null(&component_power), tensor[component], mesh);
    squared_power += square(component_power);
  }
  squared_power = sqrt(squared_power / static_cast<double>(tensor.size()));
  return squared_power;
}

size_t number_of_angular_coefficients(const size_t ell, const int spin_weight,
                                      const bool zero_m_is_real,
                                      const size_t radial_extents) {
  // If l < |s|, spin weighted spherical harmonics vanish, so just return
  // 0. In accumulate_tensor_angular_power, this will cause such terms to
  // not contribute to accumulated tensor angular power.
  if (ell < static_cast<size_t>(std::abs(spin_weight))) {
    return 0;
  }
  // Spherepack stores real and imaginary parts separately. For real scalars,
  // m=0 has only a real coefficient, so there are 1 + 2*l coefficients at
  // each l. TensorYlm components are generally complex and retain both parts
  // at m=0, giving 2*(l+1) coefficients. zero_m_is_real says whether we
  // are dealing with a real scalar or not; coefficients_at_ell is set
  // accordingly. See the help text of `SpherepackIterator` for details on the
  // zero_m_is_real parameter.
  const size_t coefficients_at_ell =
      zero_m_is_real ? 2 * ell + 1 : 2 * (ell + 1);
  // Every angular coefficient occurs independently at each radial point.
  return radial_extents * coefficients_at_ell;
}

template <typename TensorType>
void accumulate_tensor_angular_power(
    const gsl::not_null<DataVector*> weighted_squared_power,
    const gsl::not_null<std::vector<size_t>*> counts, const TensorType& tensor,
    const Mesh<3>& mesh) {
  const size_t radial_extents = mesh.extents(0);
  const size_t ell_max = mesh.extents(1) - 1;
  constexpr bool zero_m_is_real = TensorType::rank() == 0;
  DataVector component_power{};
  for (size_t component = 0; component < tensor.size(); ++component) {
    const int spin_weight = ylm::TensorYlm::helpers::component_spin_weight<
        typename TensorType::structure>(component);
    PowerMonitors::spherical_shell_angular_power_monitor(
        make_not_null(&component_power), tensor[component], mesh, spin_weight,
        zero_m_is_real);
    for (size_t ell = 0; ell <= ell_max; ++ell) {
      const size_t component_count = number_of_angular_coefficients(
          ell, spin_weight, zero_m_is_real, radial_extents);
      (*weighted_squared_power)[ell] +=
          static_cast<double>(component_count) * square(component_power[ell]);
      (*counts)[ell] += component_count;
    }
  }
}

void normalize_angular_power(const gsl::not_null<DataVector*> power,
                             const std::vector<size_t>& counts) {
  if (power->size() != counts.size()) {
    ERROR(
        "The angular power and count buffers must have the same size, but "
        "got "
        << power->size() << " and " << counts.size() << ".");
  }
  for (size_t ell = 0; ell < power->size(); ++ell) {
    (*power)[ell] =
        counts[ell] == 0
            ? 0.0
            : sqrt((*power)[ell] / static_cast<double>(counts[ell]));
  }
}

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
  result.spacetime_metric[radial_monitor_index] = tensor_radial_power_monitor(
      get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(gh_vars),
      mesh);
  result.pi[radial_monitor_index] = tensor_radial_power_monitor(
      get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(gh_vars), mesh);
  result.phi[radial_monitor_index] = tensor_radial_power_monitor(
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
  accumulate_tensor_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      make_not_null(&metric_counts),
      get<detail::Tags::Metric00<DataVector>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      make_not_null(&metric_counts),
      get<detail::Tags::Metrick0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      make_not_null(&metric_counts),
      get<detail::Tags::Metrickj<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);

  accumulate_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts),
      get<detail::Tags::Pi00<DataVector>>(gh_spatial_tensor_ylm_coefficients),
      mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts),
      get<detail::Tags::Pik0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts),
      get<detail::Tags::Pikj<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);

  accumulate_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<detail::Tags::Phik00<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<detail::Tags::Phiki0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<detail::Tags::Phikij<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      mesh);

  normalize_angular_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      metric_counts);
  normalize_angular_power(make_not_null(&result.pi[angular_monitor_index]),
                          pi_counts);
  normalize_angular_power(make_not_null(&result.phi[angular_monitor_index]),
                          phi_counts);
  return result;
}

}  // namespace gh::power_monitor
