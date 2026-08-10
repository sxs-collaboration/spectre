// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/ZernikeB2.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Transpose.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {

namespace {
template <size_t Dim>
std::tuple<size_t, size_t, size_t> check_b2_mesh(const DataVector& u,
                                                 const Mesh<Dim>& mesh) {
  const size_t n_r = mesh.extents(0);
  const size_t n_phi = mesh.extents(1);
  const size_t n_z = Dim == 3 ? mesh.extents(2) : 1_st;
  const size_t n_r_max = 2 * n_r - 2;
  const size_t M = n_phi / 2;
  ASSERT(
      mesh.basis(0) == Spectral::Basis::ZernikeB2,
      "First mesh dimension must use ZernikeB2 basis, got " << mesh.basis(0));
  ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussRadauUpper,
         "First mesh dimension must use GaussRadauUpper quadrature, got "
             << mesh.quadrature(0));
  ASSERT(mesh.quadrature(1) == Spectral::Quadrature::Equiangular,
         "Second mesh dimension must use Equiangular quadrature, got "
             << mesh.quadrature(1));
  ASSERT(n_r > 1,
         "At least 2 radial grid points are required for the B2 power "
         "monitor, got n_r = "
             << n_r);
  ASSERT(
      n_phi % 2 == 1,
      "The number of azimuthal grid points must be odd, got n_phi = " << n_phi);
  ASSERT(M <= n_r_max,
         "The number of Fourier modes M=" << M
                                          << " exceeds the maximum Zernike "
                                             "angular capability n_r_max="
                                          << n_r_max);
  ASSERT(u.size() == mesh.number_of_grid_points(),
         "Size mismatch: u.size()=" << u.size()
                                    << " != mesh.number_of_grid_points()="
                                    << mesh.number_of_grid_points());
  return {n_r_max, M, n_z};
}

size_t zernike_b2_disk_spectral_size(const size_t n_r_max, const size_t M) {
  // m=0 contributes (n_r_max + 2) / 2 modes.
  // Each m >= 1 contributes 2 * ((n_r_max - m + 2) / 2) modes (cos + sin).
  size_t size = (n_r_max + 2) / 2;
  for (size_t m = 1; m <= M; ++m) {
    size += 2 * ((n_r_max - m + 2) / 2);
  }
  return size;
}

// Accumulate squared spectral coefficients and slot counts from one disk slice.
// If radial is true, index by radial level ell = (n+1)/2; otherwise by m.
// sum_sq and counts must each have size (n_r for radial, M+1 for angular).
// Counts are always accumulated so the caller can normalize as sum_sq/counts.
void accumulate_b2_spectral_sums(const gsl::not_null<DataVector*> sum_sq,
                                 const gsl::not_null<DataVector*> counts,
                                 const DataVector& spec, const size_t n_r_max,
                                 const size_t M, const bool radial) {
  const size_t spectral_size_m0 = (n_r_max + 2) / 2;
  // m=0: Zernike degrees n = 0, 2, ..., n_r_max.  ell = (n+1)/2 = k.
  for (size_t k = 0; k < spectral_size_m0; ++k) {
    const size_t idx = radial ? k : 0_st;
    (*sum_sq)[idx] += square(spec[k]);
    (*counts)[idx] += 1.0;
  }
  // m>=1: Zernike degrees n = m, m+2, ..., n_r_max.
  // ell = (m + 2k + 1) / 2  (integer division, where k is the sub-index).
  size_t offset = spectral_size_m0;
  for (size_t m = 1; m <= M; ++m) {
    const size_t spectral_size_m = (n_r_max - m + 2) / 2;
    for (size_t k = 0; k < spectral_size_m; ++k) {
      // cos at spec[offset + k], sin at spec[offset + spectral_size_m + k]
      const size_t idx = radial ? (m + 2 * k + 1) / 2 : m;
      (*sum_sq)[idx] +=
          square(spec[offset + k]) + square(spec[offset + spectral_size_m + k]);
      (*counts)[idx] += 2.0;
    }
    offset += 2 * spectral_size_m;
  }
}
}  // namespace

void zernike_b2_disk_nodal_to_modal(const gsl::not_null<DataVector*> modal,
                                    const gsl::not_null<DataVector*> buf,
                                    const DataVector& u, const size_t n_r,
                                    const size_t n_phi, const size_t n_r_max,
                                    const size_t num_components) {
  const size_t M = n_phi / 2;
  const size_t n_disk = n_r * n_phi;
  const size_t spectral_size = zernike_b2_disk_spectral_size(n_r_max, M);
  // buf is split into two halves of size num_components * n_disk each.
  if (UNLIKELY(buf->size() < 2 * num_components * n_disk)) {
    buf->destructive_resize(2 * num_components * n_disk);
  }
  double* const buf_a = buf->data();
  double* const buf_b = buf->data() + num_components * n_disk;

  // Fourier NTM for all components simultaneously.
  const Matrix& ntm_phi =
      nodal_to_modal_matrix<Basis::Fourier, Quadrature::Equiangular>(n_phi);
  raw_transpose(make_not_null(buf_b), u.data(), n_r, num_components * n_phi);
  dgemm_<true>('N', 'N', n_phi, num_components * n_r, n_phi, 1.0,
               ntm_phi.data(), ntm_phi.spacing(), buf_b, n_phi, 0.0, buf_a,
               n_phi);
  raw_transpose(make_not_null(buf_b), buf_a, num_components * n_phi, n_r);

  // Zernike NTM for each azimuthal wavenumber m, looping over
  // components.
  const size_t spectral_size_m0 = (n_r_max + 2) / 2;
  const Matrix& ntm_m0 =
      nodal_to_modal_matrix<Basis::ZernikeB2, Quadrature::GaussRadauUpper>(
          n_r, 0, n_r_max);
  for (size_t comp = 0; comp < num_components; ++comp) {
    dgemv_('N', spectral_size_m0, n_r, 1.0, ntm_m0.data(), ntm_m0.spacing(),
           buf_b + comp * n_disk,  // NOLINT
           1, 0.0, modal->data() + comp * spectral_size, 1);
  }
  size_t offset = spectral_size_m0;
  for (size_t m = 1, j = 1; m <= M; ++m, j += 2) {
    const size_t spectral_size_m = (n_r_max - m + 2) / 2;
    const Matrix& ntm_mm =
        nodal_to_modal_matrix<Basis::ZernikeB2, Quadrature::GaussRadauUpper>(
            n_r, m, n_r_max);
    for (size_t comp = 0; comp < num_components; ++comp) {
      // Process cos (column j) and sin (column j+1) jointly.
      dgemm_<true>('N', 'N', spectral_size_m, 2, n_r, 1.0, ntm_mm.data(),
                   ntm_mm.spacing(),
                   buf_b + comp * n_disk + j * n_r,  // NOLINT
                   n_r, 0.0, modal->data() + comp * spectral_size + offset,
                   spectral_size_m);
    }
    offset += 2 * spectral_size_m;
  }
}

void b2_power_monitor_radial(const gsl::not_null<DataVector*> result,
                             const DataVector& u, const Mesh<2>& mesh) {
  const auto [n_r_max, M, n_z] = check_b2_mesh(u, mesh);
  const size_t n_r = mesh.extents(0);
  const size_t n_phi = mesh.extents(1);
  const size_t spectral_size = zernike_b2_disk_spectral_size(n_r_max, M);
  result->destructive_resize(n_r);
  DataVector sum_sq(n_r, 0.0);
  DataVector counts(n_r, 0.0);
  DataVector spec(spectral_size);
  DataVector scratch(2 * n_r * n_phi);
  Spectral::zernike_b2_disk_nodal_to_modal(
      make_not_null(&spec), make_not_null(&scratch), u, n_r, n_phi, n_r_max);
  accumulate_b2_spectral_sums(make_not_null(&sum_sq), make_not_null(&counts),
                              spec, n_r_max, M, true);
  for (size_t ell = 0; ell < n_r; ++ell) {
    (*result)[ell] = sqrt(sum_sq[ell] / counts[ell]);
  }
}

void b2_power_monitor_radial(const gsl::not_null<DataVector*> result,
                             const DataVector& u, const Mesh<3>& mesh) {
  const auto [n_r_max, M, n_z] = check_b2_mesh(u, mesh);
  const size_t n_r = mesh.extents(0);
  const size_t n_phi = mesh.extents(1);
  const size_t n_disk = n_r * n_phi;
  const size_t spectral_size = zernike_b2_disk_spectral_size(n_r_max, M);
  result->destructive_resize(n_r);
  DataVector sum_sq(n_r, 0.0);
  DataVector counts(n_r, 0.0);
  DataVector spec(spectral_size);
  DataVector scratch(2 * n_disk);
  for (size_t k_z = 0; k_z < n_z; ++k_z) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* const u_data = const_cast<double*>(u.data());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const DataVector u_slice(u_data + k_z * n_disk, n_disk);
    Spectral::zernike_b2_disk_nodal_to_modal(make_not_null(&spec),
                                             make_not_null(&scratch), u_slice,
                                             n_r, n_phi, n_r_max);
    accumulate_b2_spectral_sums(make_not_null(&sum_sq), make_not_null(&counts),
                                spec, n_r_max, M, true);
  }
  for (size_t ell = 0; ell < n_r; ++ell) {
    (*result)[ell] = sqrt(sum_sq[ell] / counts[ell]);
  }
}

void b2_power_monitor_angular(const gsl::not_null<DataVector*> result,
                              const DataVector& u, const Mesh<2>& mesh) {
  const auto [n_r_max, M, n_z] = check_b2_mesh(u, mesh);
  const size_t n_r = mesh.extents(0);
  const size_t n_phi = mesh.extents(1);
  const size_t spectral_size = zernike_b2_disk_spectral_size(n_r_max, M);
  result->destructive_resize(M + 1);
  DataVector sum_sq(M + 1, 0.0);
  DataVector counts(M + 1, 0.0);
  DataVector spec(spectral_size);
  DataVector scratch(2 * n_r * n_phi);
  Spectral::zernike_b2_disk_nodal_to_modal(
      make_not_null(&spec), make_not_null(&scratch), u, n_r, n_phi, n_r_max);
  accumulate_b2_spectral_sums(make_not_null(&sum_sq), make_not_null(&counts),
                              spec, n_r_max, M, false);
  for (size_t m = 0; m <= M; ++m) {
    (*result)[m] = sqrt(sum_sq[m] / counts[m]);
  }
}

void b2_power_monitor_angular(const gsl::not_null<DataVector*> result,
                              const DataVector& u, const Mesh<3>& mesh) {
  const auto [n_r_max, M, n_z] = check_b2_mesh(u, mesh);
  const size_t n_r = mesh.extents(0);
  const size_t n_phi = mesh.extents(1);
  const size_t n_disk = n_r * n_phi;
  const size_t spectral_size = zernike_b2_disk_spectral_size(n_r_max, M);
  result->destructive_resize(M + 1);
  DataVector sum_sq(M + 1, 0.0);
  DataVector counts(M + 1, 0.0);
  DataVector spec(spectral_size);
  DataVector scratch(2 * n_disk);
  for (size_t k_z = 0; k_z < n_z; ++k_z) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* const u_data = const_cast<double*>(u.data());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const DataVector u_slice(u_data + k_z * n_disk, n_disk);
    Spectral::zernike_b2_disk_nodal_to_modal(make_not_null(&spec),
                                             make_not_null(&scratch), u_slice,
                                             n_r, n_phi, n_r_max);
    accumulate_b2_spectral_sums(make_not_null(&sum_sq), make_not_null(&counts),
                                spec, n_r_max, M, false);
  }
  for (size_t m = 0; m <= M; ++m) {
    (*result)[m] = sqrt(sum_sq[m] / counts[m]);
  }
}
}  // namespace Spectral
