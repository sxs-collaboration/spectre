// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/FilledSpherePowerMonitor.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TensorYlmTransforms.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"

namespace gh::power_monitor {
namespace {

constexpr size_t radial_monitor_index = 0;
constexpr size_t angular_monitor_index = 1;

}  // namespace

GhFilledSpherePowerMonitors gh_filled_sphere_power_monitors(
    const gsl::not_null<CartToSphereMatrices*> cart_to_sphere_matrices,
    const Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list>&
        gh_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid) {
  if (mesh.basis(0) != Spectral::Basis::ZernikeB3 or
      mesh.quadrature(0) != Spectral::Quadrature::GaussRadauUpper or
      mesh.quadrature(1) != Spectral::Quadrature::Gauss or
      mesh.quadrature(2) != Spectral::Quadrature::Equiangular) {
    ERROR(
        "GH filled-sphere power monitors require a ZernikeB3 mesh with "
        "quadratures (GaussRadauUpper, Gauss, Equiangular), but the mesh is "
        << mesh);
  }
  const size_t n_r = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  const size_t m_max = (mesh.extents(2) - 1) / 2;
  if (l_max != m_max) {
    ERROR(
        "GH filled-sphere power monitors require l_max == m_max, but got "
        "l_max = "
        << l_max << " and m_max = " << m_max << ".");
  }
  ASSERT(n_r > 1,
         "At least 2 radial grid points required for B3 power monitors, got "
             << n_r);
  ASSERT(
      l_max >= 1,
      "At least l_max=1 required for B3 power monitors, got l_max=" << l_max);
  const size_t n_r_max = 2 * n_r - 2;
  ASSERT(l_max <= n_r_max,
         "ZernikeB3 constraint l_max <= 2*n_r-2 violated: l_max="
             << l_max << ", n_r=" << n_r);
  const auto& spherepack = ylm::get_spherepack_cache(l_max);

  // Precompute SPHEREPACK offsets grouped by l for both zero_m_is_real cases.
  // zero_m_is_real=true: scalars (rank-0 TensorYlm); false: all higher ranks.
  std::vector<std::vector<size_t>> offsets_by_l_real(l_max + 1);
  for (ylm::SpherepackIterator it{l_max, l_max, 1, true}; it; ++it) {
    offsets_by_l_real[it.l()].push_back(it());
  }
  std::vector<std::vector<size_t>> offsets_by_l_complex(l_max + 1);
  for (ylm::SpherepackIterator it{l_max, l_max, 1, false}; it; ++it) {
    offsets_by_l_complex[it.l()].push_back(it());
  }

  // Single allocation split into two regions:
  //   [scratch region] gathered + modal_buf for gather/Jacobi NTM steps
  //   [accum region]   12 DataVector accumulator views (zero-initialised)
  //
  // Scratch layout: 2 * max_n_modes_l * n_r  (max_n_modes_l = 2*(l_max+1))
  // Accum layout (per GH variable, 3 variables total):
  //   sum_sq_radial | counts_radial | sum_sq_angular | counts_angular
  const size_t max_n_modes_l = 2 * (l_max + 1);
  const size_t n_angular = l_max + 1;
  const size_t buf_per_var = 2 * n_r + 2 * n_angular;
  const size_t scratch_size = 2 * max_n_modes_l * n_r;
  const size_t accum_size = 3 * buf_per_var;
  const auto buf =
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      cpp20::make_unique_for_overwrite<double[]>(scratch_size + accum_size);
  double* const gathered = buf.get();
  double* const modal_buf = gathered + max_n_modes_l * n_r;  // NOLINT
  double* const accum_start = buf.get() + scratch_size;      // NOLINT
  std::fill(accum_start, accum_start + accum_size, 0.0);     // NOLINT

  fill_cart_to_sphere_matrices(cart_to_sphere_matrices, l_max);

  Variables<ylm::TensorYlm::filter_detail::gh_spatial_vars_list<Frame::Grid>>
      gh_spatial_tensor_ylm_coefficients(n_r * spherepack.spectral_size());
  Variables<ylm::TensorYlm::filter_detail::gh_spatial_vars_list<Frame::Grid>>
      temp_storage(n_r * spherepack.spectral_size());
  ylm::TensorYlm::gh_variables_to_tensor_ylm_coefficients(
      make_not_null(&gh_spatial_tensor_ylm_coefficients),
      make_not_null(&temp_storage), gh_vars, jac_inertial_to_grid,
      cart_to_sphere_matrices->i.value(), cart_to_sphere_matrices->ii.value(),
      cart_to_sphere_matrices->ij.value(), cart_to_sphere_matrices->ijj.value(),
      spherepack, n_r);

  double* ap = accum_start;
  DataVector metric_sum_sq_radial{};
  metric_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector metric_counts_radial{};
  metric_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector metric_sum_sq_angular{};
  metric_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector metric_counts_angular{};
  metric_counts_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT

  DataVector pi_sum_sq_radial{};
  pi_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector pi_counts_radial{};
  pi_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector pi_sum_sq_angular{};
  pi_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector pi_counts_angular{};
  pi_counts_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT

  DataVector phi_sum_sq_radial{};
  phi_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector phi_counts_radial{};
  phi_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector phi_sum_sq_angular{};
  phi_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector phi_counts_angular{};
  phi_counts_angular.set_data_ref(ap, n_angular);

  namespace detail = ylm::TensorYlm::filter_detail;

  // Metric: Metric00 (scalar), Metrick0 (vector), Metrickj (sym rank-2).
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&metric_sum_sq_radial),
      make_not_null(&metric_counts_radial),
      make_not_null(&metric_sum_sq_angular),
      make_not_null(&metric_counts_angular),
      get<detail::Tags::Metric00<DataVector>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&metric_sum_sq_radial),
      make_not_null(&metric_counts_radial),
      make_not_null(&metric_sum_sq_angular),
      make_not_null(&metric_counts_angular),
      get<detail::Tags::Metrick0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&metric_sum_sq_radial),
      make_not_null(&metric_counts_radial),
      make_not_null(&metric_sum_sq_angular),
      make_not_null(&metric_counts_angular),
      get<detail::Tags::Metrickj<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);

  // Pi: Pi00 (scalar), Pik0 (vector), Pikj (sym rank-2).
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&pi_sum_sq_radial), make_not_null(&pi_counts_radial),
      make_not_null(&pi_sum_sq_angular), make_not_null(&pi_counts_angular),
      get<detail::Tags::Pi00<DataVector>>(gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&pi_sum_sq_radial), make_not_null(&pi_counts_radial),
      make_not_null(&pi_sum_sq_angular), make_not_null(&pi_counts_angular),
      get<detail::Tags::Pik0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&pi_sum_sq_radial), make_not_null(&pi_counts_radial),
      make_not_null(&pi_sum_sq_angular), make_not_null(&pi_counts_angular),
      get<detail::Tags::Pikj<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);

  // Phi: Phik00 (vector), Phiki0 (rank-2), Phikij (rank-3 sym on last 2).
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&phi_sum_sq_radial), make_not_null(&phi_counts_radial),
      make_not_null(&phi_sum_sq_angular), make_not_null(&phi_counts_angular),
      get<detail::Tags::Phik00<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&phi_sum_sq_radial), make_not_null(&phi_counts_radial),
      make_not_null(&phi_sum_sq_angular), make_not_null(&phi_counts_angular),
      get<detail::Tags::Phiki0<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&phi_sum_sq_radial), make_not_null(&phi_counts_radial),
      make_not_null(&phi_sum_sq_angular), make_not_null(&phi_counts_angular),
      get<detail::Tags::Phikij<DataVector, 3, Frame::Grid>>(
          gh_spatial_tensor_ylm_coefficients),
      n_r, n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered,
      modal_buf);

  GhFilledSpherePowerMonitors result{};
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.spacetime_metric[radial_monitor_index]),
      metric_sum_sq_radial, metric_counts_radial);
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.spacetime_metric[angular_monitor_index]),
      metric_sum_sq_angular, metric_counts_angular);
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.pi[radial_monitor_index]), pi_sum_sq_radial,
      pi_counts_radial);
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.pi[angular_monitor_index]), pi_sum_sq_angular,
      pi_counts_angular);
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.phi[radial_monitor_index]), phi_sum_sq_radial,
      phi_counts_radial);
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.phi[angular_monitor_index]), phi_sum_sq_angular,
      phi_counts_angular);
  return result;
}

}  // namespace gh::power_monitor
