// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/Spectral/FilteringB2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Transpose.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MemoryHelpers.hpp"

namespace Spectral::filtering {
namespace {
void apply_matrix_in_first_dim(double* result, const double* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  dgemm_<true>(
      'N', 'N',
      matrix.rows(),              // rows of matrix and result
      size / matrix.columns(),    // columns of result and input
      matrix.columns(),           // columns of matrix and rows of input
      1.0,                        // overall multiplier
      matrix.data(),              // matrix
      matrix.spacing(),           // rows of matrix including padding
      input,                      // input
      matrix.columns(),           // rows of input
      add_to_result ? 1.0 : 0.0,  // overwrite output with result or add to it
      result,                     // result
      matrix.rows());             // rows of result
}

void check_valid_extents(const size_t n_r, const size_t n_phi,
                         const size_t n_r_max, const size_t M) {
  ASSERT(
      n_r > 1,
      "At least 2 radial grid points are required to filter ZernikeB2, got n_r "
      "= " << n_r);
  ASSERT(n_phi % 2 == 1,
         "Fourier with an even number of grid points can be unstable due to "
         "the top derivative not being representable");
  ASSERT(M <= n_r_max,
         "We choose to enforce the restriction that the Fourier modal space "
         "is not larger than the Zernike angular capabilities\nn_phi / 2 = "
             << M << ", Maximum from Zernike = " << n_r_max);
}

// Fills the combined ZernikeB2 (radial, angular) filter weights for a single
// disk slice. When `half_power` has a value an exponential roll-off (with
// coefficient `alpha`) is applied to both the radial mode `n` and the angular
// mode `m`; when `num_modes_to_kill` is nonzero the highest `num_modes_to_kill`
// angular `m`-modes are zeroed, with the `m = 0` mode always retained.
DataVector zernike_b2_filter_weights(const size_t n_r,
                                     const std::optional<unsigned> half_power,
                                     const size_t num_modes_to_kill,
                                     const double alpha,
                                     const size_t spectral_space_size,
                                     const size_t n_r_max, const size_t M) {
  const bool do_exp = half_power.has_value();
  const auto power = half_power.value_or(0u);
  const size_t n_order = n_r - 1;
  DataVector weights(spectral_space_size);
  size_t index = 0;
  for (size_t n_index = 0; n_index <= n_r_max; n_index += 2, ++index) {
    const auto n_i = static_cast<double>(n_index / 2);
    weights[index] =
        do_exp
            ? exp(-alpha * pow(n_i / static_cast<double>(n_order), 2 * power))
            : 1.0;
  }
  for (size_t m_index = 1; m_index <= M; ++m_index) {
    const bool kill = m_index + num_modes_to_kill > M;
    const double w_phi =
        do_exp ? exp(-alpha *
                     pow(static_cast<double>(m_index) / static_cast<double>(M),
                         2 * power))
               : 1.0;
    // cos and sin components share the same (n, m) weights
    for (size_t rep = 0; rep < 2; ++rep) {
      for (size_t n_index = m_index; n_index <= n_r_max;
           n_index += 2, ++index) {
        if (kill) {
          weights[index] = 0.0;
        } else {
          const auto n_i = static_cast<double>((n_index + 1) / 2);
          weights[index] =
              (do_exp ? exp(-alpha *
                            pow(n_i / static_cast<double>(n_order), 2 * power))
                      : 1.0) *
              w_phi;
        }
      }
    }
  }
  return weights;
}
}  // namespace

template <typename TagsList>
void zernike_b2_disk_filter(const gsl::not_null<Variables<TagsList>*> u,
                            const Mesh<2>& mesh, const double alpha,
                            const std::optional<unsigned> half_power,
                            const size_t num_modes_to_kill) {
  const auto [n_r, n_phi] = mesh.extents().indices();
  const size_t n_r_max = 2 * n_r - 2;
  const size_t M = n_phi / 2;
  check_valid_extents(n_r, n_phi, n_r_max, M);
  ASSERT(num_modes_to_kill <= M,
         "Cannot zero " << num_modes_to_kill << " angular modes when only " << M
                        << " m-modes are resolved.");
  // Nothing to do if neither an exponential roll-off nor a top-mode cutoff is
  // requested.
  if (not half_power.has_value() and num_modes_to_kill == 0) {
    return;
  }

  const Matrix& nodal_to_modal_phi =
      Spectral::nodal_to_modal_matrix<Spectral::Basis::Fourier,
                                      Spectral::Quadrature::Equiangular>(n_phi);
  const Matrix& modal_to_nodal_phi =
      Spectral::modal_to_nodal_matrix<Spectral::Basis::Fourier,
                                      Spectral::Quadrature::Equiangular>(n_phi);
  const size_t num_grid_points = mesh.number_of_grid_points();
  constexpr size_t num_components =
      Variables<TagsList>::number_of_independent_components;
  const size_t vars_size =
      Variables<TagsList>::number_of_independent_components * num_grid_points;

  auto buffer = cpp20::make_unique_for_overwrite<double[]>(vars_size);
  Variables<TagsList> temp(&buffer[0], vars_size);
  const size_t num_components_times_xi_slices = vars_size / n_r;

  // Go to phi modal space
  transpose<Variables<TagsList>, Variables<TagsList>>(
      make_not_null(&temp), (*u), mesh.extents(0),
      num_components_times_xi_slices);
  apply_matrix_in_first_dim((*u).data(), temp.data(), nodal_to_modal_phi,
                            vars_size, false);
  raw_transpose(make_not_null(temp.data()), (*u).data(),
                num_components_times_xi_slices, mesh.extents(0));

  // Apply Zernike nodal-to-modal for each component.
  // The matrix depends only on m, so fetch it once in the outer m loop and
  // iterate over components in the inner loop.
  const size_t spectral_size_m0 = (n_r_max + 2) / 2;
  const auto& nodal_to_modal_m0 =
      Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB2,
                                      Spectral::Quadrature::GaussRadauUpper>(
          n_r, 0, n_r_max);
  for (size_t i = 0; i < num_components; ++i) {
    dgemv_('N', spectral_size_m0, n_r, 1.0, nodal_to_modal_m0.data(),
           nodal_to_modal_m0.spacing(), temp.data() + i * num_grid_points, 1,
           0.0, (*u).data() + i * num_grid_points, 1);
  }
  size_t offset = spectral_size_m0;
  size_t j = 1;
  for (size_t m = 1; m <= M; ++m) {
    const size_t spectral_size = (n_r_max - m + 2) / 2;
    const auto& nodal_to_modal_mm =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB2,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, m, n_r_max);
    for (size_t i = 0; i < num_components; ++i) {
      dgemm_<true>('N', 'N', spectral_size, 2, n_r, 1.0,
                   nodal_to_modal_mm.data(), nodal_to_modal_mm.spacing(),
                   temp.data() + i * num_grid_points + j * n_r, n_r, 0.0,
                   (*u).data() + i * num_grid_points + offset, spectral_size);
    }
    j += 2;
    offset += 2 * spectral_size;
  }

  // Precompute filter weights for each spectral mode
  const size_t spectral_space_size = offset;
  const DataVector weights =
      zernike_b2_filter_weights(n_r, half_power, num_modes_to_kill, alpha,
                                spectral_space_size, n_r_max, M);

  // Do filtering: both radial and angular simultaneously
  for (size_t i = 0; i < Variables<TagsList>::number_of_independent_components;
       ++i) {
    for (size_t s = 0; s < spectral_space_size; ++s) {
      *((*u).data() + i * num_grid_points + s) *= weights[s];
    }
  }

  // Go back to phi-modal space.
  const auto& modal_to_nodal_m0 =
      Spectral::modal_to_nodal_matrix<Spectral::Basis::ZernikeB2,
                                      Spectral::Quadrature::GaussRadauUpper>(
          n_r, 0, n_r_max);
  for (size_t i = 0; i < num_components; ++i) {
    dgemv_('N', n_r, spectral_size_m0, 1.0, modal_to_nodal_m0.data(),
           modal_to_nodal_m0.spacing(), (*u).data() + i * num_grid_points, 1,
           0.0, temp.data() + i * num_grid_points, 1);
  }
  offset = spectral_size_m0;
  j = 1;
  for (size_t m = 1; m <= M; ++m) {
    const size_t spectral_size = (n_r_max - m + 2) / 2;
    const auto& modal_to_nodal_mm =
        Spectral::modal_to_nodal_matrix<Spectral::Basis::ZernikeB2,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, m, n_r_max);
    for (size_t i = 0; i < num_components; ++i) {
      dgemm_<true>('N', 'N', n_r, 2, spectral_size, 1.0,
                   modal_to_nodal_mm.data(), modal_to_nodal_mm.spacing(),
                   (*u).data() + i * num_grid_points + offset, spectral_size,
                   0.0, temp.data() + i * num_grid_points + j * n_r, n_r);
    }
    j += 2;
    offset += 2 * spectral_size;
  }

  // Go back to phi-nodal space
  raw_transpose(make_not_null((*u).data()), temp.data(), mesh.extents(0),
                num_components_times_xi_slices);
  apply_matrix_in_first_dim(temp.data(), (*u).data(), modal_to_nodal_phi,
                            vars_size, false);
  raw_transpose(make_not_null((*u).data()), temp.data(),
                num_components_times_xi_slices, mesh.extents(0));
}

template <typename TagsList>
void zernike_b2_disk_exponential_filter(
    const gsl::not_null<Variables<TagsList>*> u, const Mesh<2>& mesh,
    const double alpha, const unsigned half_power) {
  zernike_b2_disk_filter(u, mesh, alpha, std::optional<unsigned>{half_power},
                         0);
}

template <typename TagsList>
void zernike_b2_cylinder_filter(
    const gsl::not_null<Variables<TagsList>*> u, const Mesh<3>& mesh,
    const double alpha, const std::optional<unsigned> radial_angular_half_power,
    const std::optional<unsigned> z_half_power,
    const size_t num_modes_to_kill) {
  const auto [n_r, n_phi, n_z] = mesh.extents().indices();
  const size_t n_r_max = 2 * n_r - 2;
  const size_t M = n_phi / 2;
  check_valid_extents(n_r, n_phi, n_r_max, M);
  ASSERT(num_modes_to_kill <= M,
         "Cannot zero " << num_modes_to_kill << " angular modes when only " << M
                        << " m-modes are resolved.");

  const bool filter_disk =
      radial_angular_half_power.has_value() or num_modes_to_kill > 0;
  const bool filter_z = z_half_power.has_value();
  if (not filter_disk and not filter_z) {
    return;
  }

  const size_t num_grid_points = mesh.number_of_grid_points();
  constexpr size_t num_components =
      Variables<TagsList>::number_of_independent_components;
  const size_t vars_size = num_components * num_grid_points;

  auto buffer = cpp20::make_unique_for_overwrite<double[]>(vars_size);
  Variables<TagsList> temp(&buffer[0], vars_size);

  if (filter_disk) {
    const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
    const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
    const size_t num_components_times_xi_slices = vars_size / n_r;

    // Go to phi-modal space.
    transpose<Variables<TagsList>, Variables<TagsList>>(
        make_not_null(&temp), (*u), n_r, num_components_times_xi_slices);
    apply_matrix_in_first_dim((*u).data(), temp.data(), nodal_to_modal_phi,
                              vars_size, false);
    raw_transpose(make_not_null(temp.data()), (*u).data(),
                  num_components_times_xi_slices, n_r);

    const size_t spectral_space_size =
        (n_r + 2 * (2 * n_r - 1 - M / 2) * (M / 2) +
         (M % 2 ? 2 * n_r - M - 1 : 0));

    // Apply Zernike nodal-to-modal for each component and each z-slice.
    // The matrix depends only on m, so fetch it once in the outer m loop and
    // iterate over (component, z-slice) in the inner loops.
    const size_t spectral_size_m0 = (n_r_max + 2) / 2;
    const auto& nodal_to_modal_m0 =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB2,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, 0, n_r_max);
    for (size_t i = 0; i < num_components; ++i) {
      for (size_t k = 0; k < n_z; ++k) {
        dgemv_('N', spectral_size_m0, n_r, 1.0, nodal_to_modal_m0.data(),
               nodal_to_modal_m0.spacing(),
               temp.data() + i * num_grid_points + k * n_r * n_phi, 1, 0.0,
               (*u).data() + i * num_grid_points + k * spectral_space_size, 1);
      }
    }
    size_t offset = spectral_size_m0;
    size_t j = 1;
    for (size_t m = 1; m <= M; ++m) {
      const size_t spectral_size_m = (n_r_max - m + 2) / 2;
      const auto& nodal_to_modal_mm = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, m, n_r_max);
      for (size_t i = 0; i < num_components; ++i) {
        for (size_t k = 0; k < n_z; ++k) {
          dgemm_<true>(
              'N', 'N', spectral_size_m, 2, n_r, 1.0, nodal_to_modal_mm.data(),
              nodal_to_modal_mm.spacing(),
              temp.data() + i * num_grid_points + k * n_r * n_phi + j * n_r,
              n_r, 0.0,
              (*u).data() + i * num_grid_points + k * spectral_space_size +
                  offset,
              spectral_size_m);
        }
      }
      j += 2;
      offset += 2 * spectral_size_m;
    }

    // Precompute (r, phi) filter weights for one z-slice
    const DataVector weights = zernike_b2_filter_weights(
        n_r, radial_angular_half_power, num_modes_to_kill, alpha,
        spectral_space_size, n_r_max, M);

    // Do filtering: radial and angular only. The z-direction is handled
    // separately below as a nodal-space matrix apply in z's own spectral basis.
    for (size_t i = 0;
         i < Variables<TagsList>::number_of_independent_components; ++i) {
      for (size_t k = 0; k < n_z; ++k) {
        double* const u_spec =
            (*u).data() + i * num_grid_points + k * spectral_space_size;
        for (size_t s = 0; s < spectral_space_size; ++s) {
          u_spec[s] *= weights[s];  // NOLINT
        }
      }
    }

    // Go back to nodal space: Zernike modal-to-nodal for each component and
    // z-slice, writing results into temp (phi-modal layout).
    const auto& modal_to_nodal_m0 =
        Spectral::modal_to_nodal_matrix<Spectral::Basis::ZernikeB2,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, 0, n_r_max);
    for (size_t i = 0; i < num_components; ++i) {
      for (size_t k = 0; k < n_z; ++k) {
        dgemv_('N', n_r, spectral_size_m0, 1.0, modal_to_nodal_m0.data(),
               modal_to_nodal_m0.spacing(),
               (*u).data() + i * num_grid_points + k * spectral_space_size, 1,
               0.0, temp.data() + i * num_grid_points + k * n_r * n_phi, 1);
      }
    }
    offset = spectral_size_m0;
    j = 1;
    for (size_t m = 1; m <= M; ++m) {
      const size_t spectral_size_m = (n_r_max - m + 2) / 2;
      const auto& modal_to_nodal_mm = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, m, n_r_max);
      for (size_t i = 0; i < num_components; ++i) {
        for (size_t k = 0; k < n_z; ++k) {
          dgemm_<true>(
              'N', 'N', n_r, 2, spectral_size_m, 1.0, modal_to_nodal_mm.data(),
              modal_to_nodal_mm.spacing(),
              (*u).data() + i * num_grid_points + k * spectral_space_size +
                  offset,
              spectral_size_m, 0.0,
              temp.data() + i * num_grid_points + k * n_r * n_phi + j * n_r,
              n_r);
        }
      }
      j += 2;
      offset += 2 * spectral_size_m;
    }

    // Go back to phi-nodal space
    raw_transpose(make_not_null((*u).data()), temp.data(), mesh.extents(0),
                  num_components_times_xi_slices);
    apply_matrix_in_first_dim(temp.data(), (*u).data(), modal_to_nodal_phi,
                              vars_size, false);
    raw_transpose(make_not_null((*u).data()), temp.data(),
                  num_components_times_xi_slices, mesh.extents(0));
  }

  if (filter_z) {
    // Apply z-direction filter
    const Matrix z_filter = Spectral::filtering::exponential_filter(
        mesh.slice_through(2), alpha, *z_half_power);
    const size_t chunk_size_z = n_r * n_phi;
    const size_t num_chunks_z = vars_size / chunk_size_z;
    transpose<Variables<TagsList>, Variables<TagsList>>(
        make_not_null(&temp), (*u), chunk_size_z, num_chunks_z);
    apply_matrix_in_first_dim((*u).data(), temp.data(), z_filter, vars_size,
                              false);
    raw_transpose(make_not_null(temp.data()), (*u).data(), num_chunks_z,
                  chunk_size_z);
    // Unable to avoid this copy in the current set-up due to an odd number of
    // operations on the data
    std::copy(temp.data(), temp.data() + vars_size, (*u).data());
  }
}

template <typename TagsList>
void zernike_b2_cylinder_exponential_filter(
    const gsl::not_null<Variables<TagsList>*> u, const Mesh<3>& mesh,
    const double alpha, const unsigned half_power) {
  zernike_b2_cylinder_filter(u, mesh, alpha,
                             std::optional<unsigned>{half_power},
                             std::optional<unsigned>{half_power}, 0);
}
}  // namespace Spectral::filtering
