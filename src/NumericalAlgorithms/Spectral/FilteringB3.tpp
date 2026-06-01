// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/Spectral/FilteringB3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

namespace Spectral::filtering {
namespace b3_detail {
[[maybe_unused]] inline void apply_matrix_in_first_dim(
    double* result, const double* const input, const Matrix& matrix,
    const size_t size) {
  dgemm_<true>('N', 'N',
               matrix.rows(),            // rows of matrix and result
               size / matrix.columns(),  // columns of result and input
               matrix.columns(),         // columns of matrix and rows of input
               1.0,                      // overall multiplier
               matrix.data(),            // matrix
               matrix.spacing(),         // rows of matrix including padding
               input,                    // input
               matrix.columns(),         // rows of input
               0.0,                      // overwrite output
               result,                   // result
               matrix.rows());           // rows of result
}

[[maybe_unused]] inline void check_valid_extents_b3(const size_t n_r,
                                                    const size_t l_max) {
  ASSERT(n_r > 1,
         "At least 2 radial grid points are required to filter ZernikeB3, got "
         "n_r = "
             << n_r);
  ASSERT(l_max >= 2,
         "At least l_max=2 (3 latitudinal points) is required to filter "
         "ZernikeB3, got l_max = "
             << l_max);
  ASSERT(l_max <= 2 * n_r - 2,
         "ZernikeB3 radial resolution is insufficient for the requested "
         "angular resolution. Need l_max <= 2*n_r-2, but l_max="
             << l_max << " and n_r=" << n_r);
}
}  // namespace b3_detail

template <typename TagsList>
void zernike_b3_ball_radial_exponential_filter(
    const gsl::not_null<Variables<TagsList>*> u,
    const gsl::not_null<DataVector*> buf, const Mesh<3>& mesh,
    const double alpha, const unsigned half_power) {
  const size_t n_r = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  b3_detail::check_valid_extents_b3(n_r, l_max);
  const size_t n_r_max = 2 * n_r - 2;
  const size_t n_order = n_r - 1;

  const auto& ylm = ylm::get_spherepack_cache(l_max);
  const size_t n_spectral = ylm.spectral_size();
  const size_t n_phys = mesh.number_of_grid_points();
  constexpr size_t n_components =
      Variables<TagsList>::number_of_independent_components;

  const auto& modes = ylm::get_modes_by_degree_cache(l_max);

  // Partition the caller-provided buffer into three contiguous regions:
  //   spec_buf  : n_components * n_spectral * n_r  (SH spectral coefficients)
  //   gathered  : max_batch * n_r                  (same-l radial profiles)
  //   modal_buf : max_batch * n_r                  (ZernikeB3 modal space)
  // where max_batch = (2*l_max+1) * n_components.
  const size_t max_batch = (2 * l_max + 1) * n_components;
  const size_t spec_buf_size = n_components * n_spectral * n_r;
  const size_t buf_size = spec_buf_size + 2 * max_batch * n_r;
  if (UNLIKELY(buf->size() < buf_size)) {
    buf->destructive_resize(buf_size);
  }
  double* const spec_buf = buf->data();
  double* const gathered = spec_buf + spec_buf_size;     // NOLINT
  double* const modal_buf = gathered + max_batch * n_r;  // NOLINT

  // SH analysis: physical space -> SH spectral space for all components.
  // After this, spec_buf[comp*n_spectral*n_r + s*n_r + i_r] holds the SH
  // spectral coefficient for mode s at radial collocation point i_r.
  for (size_t comp = 0; comp < n_components; ++comp) {
    ylm.phys_to_spec_all_offsets(
        make_not_null(spec_buf + comp * n_spectral * n_r),  // NOLINT
        make_not_null(u->data() + comp * n_phys), n_r);
  }

  // For each angular degree l: gather same-l SH modes, apply ZernikeB3
  // nodal-to-modal, apply filter weights, apply modal-to-nodal, scatter back.
  for (size_t l = 0; l <= l_max; ++l) {
    const size_t* offsets = modes.offsets.data() + modes.l_start[l];  // NOLINT
    const size_t n_modes_l = modes.l_start[l + 1] - modes.l_start[l];
    // spectral_size_l = number of ZernikeB3 radial modes for this l
    const size_t spectral_size_l = (n_r_max - l + 2) / 2;
    const size_t n_batch = n_modes_l * n_components;

    const auto& ntm =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB3,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, l, n_r_max);
    const auto& mtn =
        Spectral::modal_to_nodal_matrix<Spectral::Basis::ZernikeB3,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, l, n_r_max);

    // Gather: copy radial profiles for all same-l modes across all components
    // into the contiguous `gathered` buffer (layout: n_r × n_batch,
    // column-major).
    for (size_t comp = 0; comp < n_components; ++comp) {
      const double* src = spec_buf + comp * n_spectral * n_r;  // NOLINT
      for (size_t k = 0; k < n_modes_l; ++k) {
        const size_t col = comp * n_modes_l + k;
        std::copy(src + offsets[k] * n_r,        // NOLINT
                  src + (offsets[k] + 1) * n_r,  // NOLINT
                  gathered + col * n_r);         // NOLINT
      }
    }

    // NTM: modal_buf (spectral_size_l × n_batch) = ntm x gathered (n_r ×
    // n_batch)
    b3_detail::apply_matrix_in_first_dim(modal_buf, gathered, ntm,
                                         n_batch * n_r);

    // Apply filter weights: w_r(n_jacobi, l), where
    //   n_total = l + 2*n_jacobi,  n_i = n_total/2 (integer division)
    //   w_r = exp(-alpha * (n_i / n_order)^(2p))
    // The modal buffer is (spectral_size_l × n_batch) in column-major order,
    // so modal_buf[k_spec + spectral_size_l * col] is element (k_spec, col).
    for (size_t k_spec = 0; k_spec < spectral_size_l; ++k_spec) {
      const auto n_i =
          static_cast<double>(static_cast<size_t>((l + 2 * k_spec) / 2));
      const double w_r =
          std::exp(-alpha * integer_pow(n_i / static_cast<double>(n_order),
                                        static_cast<int>(2 * half_power)));
      for (size_t col = 0; col < n_batch; ++col) {
        modal_buf[k_spec + spectral_size_l * col] *= w_r;  // NOLINT
      }
    }

    // MTN: gathered (n_r × n_batch) = mtn x modal_buf (spectral_size_l ×
    // n_batch)
    b3_detail::apply_matrix_in_first_dim(gathered, modal_buf, mtn,
                                         n_batch * spectral_size_l);

    // Scatter: write filtered radial profiles back to the spectral buffer
    for (size_t comp = 0; comp < n_components; ++comp) {
      double* dst = spec_buf + comp * n_spectral * n_r;  // NOLINT
      for (size_t k = 0; k < n_modes_l; ++k) {
        const size_t col = comp * n_modes_l + k;
        std::copy(gathered + col * n_r,        // NOLINT
                  gathered + (col + 1) * n_r,  // NOLINT
                  dst + offsets[k] * n_r);     // NOLINT
      }
    }
  }

  // SH synthesis: SH spectral space -> physical space for all components
  for (size_t comp = 0; comp < n_components; ++comp) {
    ylm.spec_to_phys_all_offsets(
        make_not_null(u->data() + comp * n_phys),
        make_not_null(spec_buf + comp * n_spectral * n_r), n_r);  // NOLINT
  }
}

template <typename TagsList>
void zernike_b3_ball_radial_exponential_filter(
    const gsl::not_null<Variables<TagsList>*> u, const Mesh<3>& mesh,
    const double alpha, const unsigned half_power) {
  constexpr size_t n_components =
      Variables<TagsList>::number_of_independent_components;
  const size_t n_r = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  b3_detail::check_valid_extents_b3(n_r, l_max);
  const size_t n_spectral = ylm::get_spherepack_cache(l_max).spectral_size();
  const size_t max_batch = (2 * l_max + 1) * n_components;
  DataVector buf(n_components * n_spectral * n_r + 2 * max_batch * n_r);
  zernike_b3_ball_radial_exponential_filter(u, make_not_null(&buf), mesh, alpha,
                                            half_power);
}
}  // namespace Spectral::filtering
