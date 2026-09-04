// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/FilledSpherePowerMonitor.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/TensorYlm/ApplyFilter.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"

namespace ScalarWave::power_monitor {
namespace {

constexpr size_t radial_monitor_index = 0;
constexpr size_t angular_monitor_index = 1;

}  // namespace

SwFilledSpherePowerMonitors sw_filled_sphere_power_monitors(
    const gsl::not_null<SwCartToSphereMatrix*> cart_to_sphere_matrix,
    const Variables<
        ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>& sw_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid) {
  if (mesh.basis(0) != Spectral::Basis::ZernikeB3 or
      mesh.quadrature(0) != Spectral::Quadrature::GaussRadauUpper or
      mesh.quadrature(1) != Spectral::Quadrature::Gauss or
      mesh.quadrature(2) != Spectral::Quadrature::Equiangular) {
    ERROR(
        "SW filled-sphere power monitors require a ZernikeB3 mesh with "
        "quadratures (GaussRadauUpper, Gauss, Equiangular), but the mesh is "
        << mesh);
  }
  const size_t n_r = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  const size_t m_max = (mesh.extents(2) - 1) / 2;
  if (l_max != m_max) {
    ERROR(
        "SW filled-sphere power monitors require l_max == m_max, but got "
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
  // zero_m_is_real=true: scalars Psi and Pi (rank-0); false: co-vector Phi.
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
  // Accum layout (per SW variable, 3 variables total):
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

  // Transform SW variables to TensorYlm spectral coefficients.
  // Frame-transform Phi to the grid frame; Psi and Pi are scalars (unchanged).
  namespace fd = ylm::TensorYlm::filter_detail;

  const size_t n_phys = mesh.number_of_grid_points();
  const size_t n_spec = n_r * spherepack.spectral_size();

  Variables<fd::sw_vars_list<Frame::Grid>> sw_grid_frame(n_phys);
  fd::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&sw_grid_frame), sw_vars,
                                    jac_inertial_to_grid);

  // SH analysis (nodal to modal) in the angular directions.
  Variables<fd::sw_vars_list<Frame::Grid>> sw_modal(n_spec);
  fd::nodal_to_modal_ylm(make_not_null(&sw_modal), sw_grid_frame, spherepack,
                         n_r);

  // Apply Cartesian-to-TensorYlm transform to Phi SH coefficients.
  fill_sw_cart_to_sphere_matrix(cart_to_sphere_matrix, l_max);
  Variables<fd::sw_vars_list<Frame::Grid>> phi_tensor_ylm_vars(n_spec);
  for (auto& comp :
       get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars)) {
    comp = 0.0;
  }
  {
    const gsl::span<double> src(
        get<ScalarWave::Tags::Phi<3, Frame::Grid>>(sw_modal)[0].data(),
        3 * n_spec);
    gsl::span<double> dest(
        get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars)[0]
            .data(),
        3 * n_spec);
    for (size_t offset = 0; offset < n_r; ++offset) {
      cart_to_sphere_matrix->i->increment_multiply_on_right(
          make_not_null(&dest), offset, n_r, src, offset, n_r);
    }
  }

  // Set up accumulator DataVector views into the flat buffer.
  double* ap = accum_start;
  DataVector psi_sum_sq_radial{};
  psi_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector psi_counts_radial{};
  psi_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector psi_sum_sq_angular{};
  psi_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector psi_counts_angular{};
  psi_counts_angular.set_data_ref(ap, n_angular);
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

  // Psi: scalar (rank-0), zero_m_is_real=true, spin_weight=0.
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&psi_sum_sq_radial), make_not_null(&psi_counts_radial),
      make_not_null(&psi_sum_sq_angular), make_not_null(&psi_counts_angular),
      get<ScalarWave::Tags::Psi>(sw_modal), n_r, n_r_max, offsets_by_l_real,
      offsets_by_l_complex, gathered, modal_buf);

  // Pi: scalar (rank-0), zero_m_is_real=true, spin_weight=0.
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&pi_sum_sq_radial), make_not_null(&pi_counts_radial),
      make_not_null(&pi_sum_sq_angular), make_not_null(&pi_counts_angular),
      get<ScalarWave::Tags::Pi>(sw_modal), n_r, n_r_max, offsets_by_l_real,
      offsets_by_l_complex, gathered, modal_buf);

  // Phi: co-vector (rank-1), zero_m_is_real=false, spin_weight from TensorYlm.
  PowerMonitors::accumulate_b3_tensor_sums(
      make_not_null(&phi_sum_sq_radial), make_not_null(&phi_counts_radial),
      make_not_null(&phi_sum_sq_angular), make_not_null(&phi_counts_angular),
      get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars), n_r,
      n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered, modal_buf);

  SwFilledSpherePowerMonitors result{};
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.psi[radial_monitor_index]), psi_sum_sq_radial,
      psi_counts_radial);
  PowerMonitors::normalize_b3_power(
      make_not_null(&result.psi[angular_monitor_index]), psi_sum_sq_angular,
      psi_counts_angular);
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

}  // namespace ScalarWave::power_monitor
