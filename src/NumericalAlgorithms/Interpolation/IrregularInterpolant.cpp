// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IrregularInterpolant.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/CardinalInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"

namespace {
template <size_t Dim>
void compute_zernike_b2_interpolation(
    gsl::not_null<Matrix*> result, const Mesh<Dim>& mesh,
    const std::array<Matrix, Dim>& interpolation_matrices,
    const size_t number_of_target_points) {
  static_assert(Dim == 2 or Dim == 3,
                "ZernikeB2 interpolation only supports 2D and 3D");
  const size_t n_r = mesh.extents(0);
  const size_t n_phi = mesh.extents(1);
  const size_t n_z = Dim == 3 ? mesh.extents(2) : 1;

  const size_t combined_dim = n_phi * n_z;  // n_z=1 for 2D case
  Matrix zernike_weights(2 * number_of_target_points, combined_dim);

  intrp::Cardinal<Dim>::compute_zernike_b2_weights(
      make_not_null(&zernike_weights), mesh, interpolation_matrices,
      number_of_target_points);

  // Now fill the result matrix using the pre-computed weights
  for (size_t k = 0; k < number_of_target_points; ++k) {
    for (size_t s = 0; s < mesh.number_of_grid_points(); ++s) {
      if constexpr (Dim == 2) {
        const size_t i_phi = s / n_r;  // NOLINT(clang-analyzer-core.DivideZero)
        const size_t i_r = s % n_r;

        // Even contribution (radial_offset=0)
        (*result)(k, s) =
            interpolation_matrices[0](k, i_r) * zernike_weights(2 * k, i_phi);

        // Odd contribution (radial_offset=n_r)
        if (i_r + n_r < interpolation_matrices[0].columns()) {
          (*result)(k, s) += interpolation_matrices[0](k, i_r + n_r) *
                             zernike_weights(2 * k + 1, i_phi);
        }
      } else {  // Dim == 3
        // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
        const size_t i_z = s / (n_phi * n_r);
        const size_t i_phi = (s % (n_phi * n_r)) / n_r;
        const size_t i_r = s % n_r;
        const size_t idx = i_z * n_phi + i_phi;

        // Even contribution (radial_offset=0)
        (*result)(k, s) =
            interpolation_matrices[0](k, i_r) * zernike_weights(2 * k, idx);

        // Odd contribution (radial_offset=n_r)
        if (i_r + n_r < interpolation_matrices[0].columns()) {
          (*result)(k, s) += interpolation_matrices[0](k, i_r + n_r) *
                             zernike_weights(2 * k + 1, idx);
        }
      }
    }
  }
}

template <typename DataType>
void compute_zernike_b3_interpolation(
    gsl::not_null<Matrix*> result, const Mesh<3>& mesh,
    const std::array<Matrix, 3>& interpolation_matrices,
    const tnsr::I<DataType, 3, Frame::ElementLogical>& target_points,
    const size_t number_of_target_points) {
  const size_t n_r = mesh.extents(0);
  const size_t n_th = mesh.extents(1);
  const size_t n_phi = mesh.extents(2);
  const size_t n_ang = n_th * n_phi;
  const size_t l_max = n_th - 1;
  const size_t m_max = (n_phi - 1) / 2;

  const ylm::Spherepack b3_ylm(l_max, m_max);
  const size_t n_spectral = b3_ylm.spectral_size();

  Matrix zernike_b3_weights(2 * number_of_target_points, n_spectral);
  if constexpr (std::is_same_v<DataType, DataVector>) {
    intrp::Cardinal<3>::compute_zernike_b3_weights(
        make_not_null(&zernike_b3_weights), mesh, target_points, b3_ylm,
        number_of_target_points);
  } else {
    tnsr::I<DataVector, 3, Frame::ElementLogical> target_points_dv{1_st};
    get<0>(target_points_dv) = get<0>(target_points);
    get<1>(target_points_dv) = get<1>(target_points);
    get<2>(target_points_dv) = get<2>(target_points);
    intrp::Cardinal<3>::compute_zernike_b3_weights(
        make_not_null(&zernike_b3_weights), mesh, target_points_dv, b3_ylm,
        number_of_target_points);
  }

  // Build the SH analysis matrix A[s * n_ang + i_ang] = A(s, i_ang) with one
  // phys_to_spec_all_offsets call on an identity-like arrangement: for "shell"
  // i_ang (stride = n_ang) the physical data is a unit vector at position
  // i_ang, so A_mat[s * n_ang + i_ang] is the s-th spectral coefficient of
  // that unit vector, i.e. A(s, i_ang).
  DataVector phys_id(n_ang * n_ang, 0.0);
  for (size_t i = 0; i < n_ang; ++i) {
    phys_id[i * n_ang + i] = 1.0;
  }
  const DataVector A_mat = b3_ylm.phys_to_spec_all_offsets(phys_id, n_ang);

  // result(k, s_grid) with s_grid = i_ang * n_r + i_r equals
  //   interp_r_even(k, i_r) * B_even(k, i_ang)
  // + interp_r_odd (k, i_r) * B_odd (k, i_ang)
  // where B_{even,odd}(k, i_ang) = sum_s zernike_b3_weights(2k{+1}, s)*A(s,
  // i_ang). This is the B3 analogue of compute_zernike_b2_interpolation.
  for (size_t k = 0; k < number_of_target_points; ++k) {
    for (size_t i_ang = 0; i_ang < n_ang; ++i_ang) {
      double b_even = 0.0;
      double b_odd = 0.0;
      for (size_t s = 0; s < n_spectral; ++s) {
        const double a_val = A_mat[s * n_ang + i_ang];
        b_even += zernike_b3_weights(2 * k, s) * a_val;
        b_odd += zernike_b3_weights(2 * k + 1, s) * a_val;
      }
      for (size_t i_r = 0; i_r < n_r; ++i_r) {
        (*result)(k, i_ang* n_r + i_r) =
            interpolation_matrices[0](k, i_r) * b_even +
            interpolation_matrices[0](k, i_r + n_r) * b_odd;
      }
    }
  }
}

template <size_t Order>
std::vector<double> fd_stencil(const DataVector& xi_source, double xi_target);

// potential future optimization: use the barycentric formula

// Just linear for now, can be extended to higher order...
// Future optimization: it might be more efficient to use Blaze's sparse
// matrices since the interpolation matrix is mostly zeros
template <>
std::vector<double> fd_stencil<1>(const DataVector& xi_source,
                                  const double xi_target) {
  ASSERT(std::is_sorted(std::begin(xi_source), std::end(xi_source)),
         "xi_source = " << xi_source);
  auto xi_u =
      std::upper_bound(std::begin(xi_source), std::end(xi_source), xi_target);
  if (std::end(xi_source) == xi_u) {
    std::advance(xi_u, -1);
  }
  if (std::begin(xi_source) == xi_u) {
    std::advance(xi_u, 1);
  }
  const auto xi_l = std::prev(xi_u);
  auto index = std::distance(std::begin(xi_source), xi_l);
  std::vector<double> result(xi_source.size(), 0.0);
  const auto result_l = std::next(std::begin(result), index);
  const auto result_u = std::next(result_l, 1);
  *result_l = (*xi_u - xi_target) / (*xi_u - *xi_l);
  *result_u = (xi_target - *xi_l) / (*xi_u - *xi_l);
  return result;
}

// Quadratic (three-point) finite-difference interpolation stencil.
// Returns a weight vector with exactly three non-zero entries corresponding
// to a chosen triplet of consecutive grid points. For interior targets we pick
// the more centered of the two possible triplets bracketing the target.
// Edge targets (or outside) use the first or last 3 points, yielding
// one-sided quadratic extrapolation.
template <>
std::vector<double> fd_stencil<2>(const DataVector& xi_source,
                                  const double xi_target) {
  ASSERT(std::is_sorted(std::begin(xi_source), std::end(xi_source)),
         "xi_source = " << xi_source);
  const size_t num_of_pts = xi_source.size();
  ASSERT(num_of_pts >= 3,
         "Need at least 3 points for quadratic interpolation (got "
             << num_of_pts << ")");

  auto xi_u =
      std::upper_bound(std::begin(xi_source), std::end(xi_source), xi_target);

  // Determine the three indices to use.
  size_t i0{0};
  size_t i1{0};
  size_t i2{0};

  if (xi_u == std::begin(xi_source)) {
    // Target at/below first point
    i0 = 0;
    i1 = 1;
    i2 = 2;
  } else if (xi_u == std::end(xi_source)) {
    // Target above last point
    i0 = num_of_pts - 3;
    i1 = num_of_pts - 2;
    i2 = num_of_pts - 1;
  } else {
    const auto iu =
        static_cast<size_t>(std::distance(std::begin(xi_source), xi_u));
    const size_t il = iu - 1;  // lower bracket index

    if (il == 0) {
      // Near lower boundary
      i0 = 0;
      i1 = 1;
      i2 = 2;
    } else if (iu == num_of_pts - 1) {
      // Near upper boundary (target between last two points)
      i0 = num_of_pts - 3;
      i1 = num_of_pts - 2;
      i2 = num_of_pts - 1;
    } else {
      // Interior: choose centered triplet
      // Option A: (il-1, il, iu)
      // Option B: (il, iu, iu+1)
      const double x_il = xi_source[il];
      const double x_iu = xi_source[iu];
      // Compare proximity to pick more centered set
      if (xi_target - x_il < x_iu - xi_target) {
        i0 = il - 1;
        i1 = il;
        i2 = iu;
      } else {
        i0 = il;
        i1 = iu;
        i2 = iu + 1;
      }
    }
  }

  // Compute Lagrange weights for the selected three nodes.
  const double x0 = xi_source[i0];
  const double x1 = xi_source[i1];
  const double x2 = xi_source[i2];

  const double denom0 = (x0 - x1) * (x0 - x2);
  const double denom1 = (x1 - x0) * (x1 - x2);
  const double denom2 = (x2 - x0) * (x2 - x1);

  std::vector<double> w(num_of_pts, 0.0);
  w[i0] = (xi_target - x1) * (xi_target - x2) / denom0;
  w[i1] = (xi_target - x0) * (xi_target - x2) / denom1;
  w[i2] = (xi_target - x0) * (xi_target - x1) / denom2;

  return w;
}

// Cubic (four-point) finite-difference interpolation stencil.
// Uses four consecutive grid points to build a degree-3 Lagrange interpolant.
// Boundary/out-of-domain targets use the first/last 4 points (one-sided cubic
// extrapolation). Interior targets: choose among candidate 4-point windows that
// contain the bracketing pair to best center the target (minimizing distance to
// the window's midpoint).
template <>
std::vector<double> fd_stencil<3>(const DataVector& xi_source,
                                  const double xi_target) {
  ASSERT(std::is_sorted(std::begin(xi_source), std::end(xi_source)),
         "xi_source = " << xi_source);
  const size_t num_of_pts = xi_source.size();
  ASSERT(num_of_pts >= 4, "Need at least 4 points for cubic interpolation (got "
                              << num_of_pts << ")");
  auto xi_u =
      std::upper_bound(std::begin(xi_source), std::end(xi_source), xi_target);

  size_t start{0};  // start index of the 4-point window
  if (xi_u == std::begin(xi_source)) {
    // Below first point
    start = 0;
  } else if (xi_u == std::end(xi_source)) {
    // Above last point
    start = num_of_pts - 4;
  } else {
    const auto iu =
        static_cast<size_t>(std::distance(std::begin(xi_source), xi_u));
    const size_t il = iu - 1;  // lower bracket index
    // Determine candidate window starts ensuring il >= s and iu <= s+3
    const size_t min_start = (il >= 2 ? il - 2 : 0);
    const size_t max_start = std::min(il, num_of_pts - 4);
    // Collect candidates
    double best_dist = std::numeric_limits<double>::max();
    size_t best_start = min_start;
    for (size_t s = min_start; s <= max_start; ++s) {
      // Midpoint of inner two nodes (s+1, s+2) gives a reasonable center
      const double mid = 0.5 * (xi_source[s + 1] + xi_source[s + 2]);
      const double dist = std::abs(xi_target - mid);
      if (dist < best_dist) {
        best_dist = dist;
        best_start = s;
      }
    }
    start = best_start;
  }

  const size_t i0 = start;
  const size_t i1 = start + 1;
  const size_t i2 = start + 2;
  const size_t i3 = start + 3;
  const double x0 = xi_source[i0];
  const double x1 = xi_source[i1];
  const double x2 = xi_source[i2];
  const double x3 = xi_source[i3];

  // Compute Lagrange weights for 4 points.
  auto lagrange_weight = [&](double xj, const std::array<double, 3>& others) {
    double num = 1.0;
    double denom = 1.0;
    for (const double xo : others) {
      num *= (xi_target - xo);
      denom *= (xj - xo);
    }
    return num / denom;
  };

  std::vector<double> w(num_of_pts, 0.0);
  w[i0] = lagrange_weight(x0, {x1, x2, x3});
  w[i1] = lagrange_weight(x1, {x0, x2, x3});
  w[i2] = lagrange_weight(x2, {x0, x1, x3});
  w[i3] = lagrange_weight(x3, {x0, x1, x2});
  return w;
}

template <size_t Dim, typename DataType>
Matrix interpolation_matrix(
    const Mesh<Dim>& mesh,
    const tnsr::I<DataType, Dim, Frame::ElementLogical>& points);

template <typename DataType>
Matrix interpolation_matrix(
    const Mesh<1>& mesh,
    const tnsr::I<DataType, 1, Frame::ElementLogical>& points,
    const std::optional<size_t>& fd_to_fd_interp_order) {
  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    auto source_xi = logical_coordinates(mesh);
    const auto number_of_source_points = mesh.number_of_grid_points();
    const DataVector xi_source(get<0>(source_xi).data(),
                               number_of_source_points);
    const size_t number_of_target_points = get_size(get<0>(points));
    Matrix result(number_of_target_points, number_of_source_points);
    constexpr size_t default_order = 1;  // linear interpolation by default
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get_element(get<0>(points), p);
      std::vector<double> stencil;
      switch (fd_to_fd_interp_order.value_or(default_order)) {
        case 1: {
          stencil = fd_stencil<1>(xi_source, xi_target);
          break;
        }
        case 2: {
          stencil = fd_stencil<2>(xi_source, xi_target);
          break;
        }
        case 3: {
          stencil = fd_stencil<3>(xi_source, xi_target);
          break;
        }
        default:
          ERROR("Unsupported fd_to_fd_interp_order: "
                << fd_to_fd_interp_order.value_or(default_order)
                << "; expected 1, 2, or 3.");
      }
      for (size_t i = 0; i < number_of_source_points; ++i) {
        result(p, i) = stencil[i];
      }
    }
    return result;
  }

  // Not FD, so use spectral interpolation
  ASSERT(
      fd_to_fd_interp_order == std::nullopt,
      "fd_to_fd_interp_order only applies to FD meshes. But Mesh is " << mesh);
  return Spectral::interpolation_matrix(mesh, get<0>(points));
}

template <typename DataType>
Matrix interpolation_matrix(
    const Mesh<2>& mesh,
    const tnsr::I<DataType, 2, Frame::ElementLogical>& points,
    const std::optional<size_t>& fd_to_fd_interp_order) {
  const auto number_of_target_points = get_size(get<0>(points));
  Matrix result(number_of_target_points, mesh.number_of_grid_points());

  ASSERT(
      mesh.basis()[0] == Spectral::Basis::FiniteDifference or
          fd_to_fd_interp_order == std::nullopt,
      "fd_to_fd_interp_order only applies to FD meshes. But Mesh is " << mesh);

  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::FiniteDifference,
           "Mixed FD and DG bases are not supported. Mesh = " << mesh);
    auto source_xi = logical_coordinates(mesh);
    DataVector xi_source{get<0>(source_xi).data(), mesh.extents(0)};
    DataVector eta_source;
    if (mesh.extents(1) == mesh.extents(0)) {
      eta_source.set_data_ref(&xi_source);
    } else {
      eta_source.destructive_resize(mesh.extents(1));
      for (size_t j = 0; j < mesh.extents(1); ++j) {
        eta_source[j] = get<1>(source_xi)[j * mesh.extents(0)];
      }
    }
    constexpr size_t default_order = 1;  // linear interpolation by default
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get_element(get<0>(points), p);
      const double eta_target = get_element(get<1>(points), p);
      std::vector<double> xi_stencil;
      std::vector<double> eta_stencil;
      switch (fd_to_fd_interp_order.value_or(default_order)) {
        case 1: {
          xi_stencil = fd_stencil<1>(xi_source, xi_target);
          eta_stencil = fd_stencil<1>(eta_source, eta_target);
          break;
        }
        case 2: {
          xi_stencil = fd_stencil<2>(xi_source, xi_target);
          eta_stencil = fd_stencil<2>(eta_source, eta_target);
          break;
        }
        case 3: {
          xi_stencil = fd_stencil<3>(xi_source, xi_target);
          eta_stencil = fd_stencil<3>(eta_source, eta_target);
          break;
        }
        default:
          ERROR("Unsupported fd_to_fd_interp_order: "
                << fd_to_fd_interp_order.value_or(default_order)
                << "; expected 1, 2, or 3.");
      }
      for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
        for (size_t i = 0; i < mesh.extents(0); ++i) {
          result(p, s) = xi_stencil[i] * eta_stencil[j];
          ++s;
        }
      }
    }
    return result;
  } else if (mesh.basis()[0] == Spectral::Basis::SphericalHarmonic) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::SphericalHarmonic,
           "Expected both dimensions to have spherical harmonic basis. Mesh = "
               << mesh);
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto [n_th, n_ph] = mesh.extents().indices();
    const auto& [m_th, m_ph] = cardinal_interpolator.interpolation_matrices();
    for (size_t i_ph = 0, s = 0; i_ph < n_ph; ++i_ph) {
      for (size_t i_th = 0; i_th < n_th; ++i_th) {
        for (size_t k = 0; k < number_of_target_points; ++k) {
          result(k, s) = m_th(k, i_th) * m_ph(2 * k, i_ph) +
                         m_th(k, i_th + n_th) * m_ph(2 * k + 1, i_ph);
        }
        ++s;
      }
    }
    return result;
  } else if (mesh.basis()[0] == Spectral::Basis::ZernikeB2) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::ZernikeB2,
           "Unexpected basis combination: " << mesh.basis());
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto& interpolation_matrices =
        cardinal_interpolator.interpolation_matrices();
    compute_zernike_b2_interpolation<2>(make_not_null(&result), mesh,
                                        interpolation_matrices,
                                        number_of_target_points);
    return result;
  }

  // Not FD or special basis, so use 1D spectral interpolation matrices
  const std::array<Matrix, 2> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points))}};

  // First dimension of DataVector varies fastest.
  for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
    for (size_t i = 0; i < mesh.extents(0); ++i) {
      for (size_t p = 0; p < number_of_target_points; ++p) {
        result(p, s) = matrices[0](p, i) * matrices[1](p, j);
      }
      ++s;
    }
  }
  return result;
}

template <typename DataType>
Matrix interpolation_matrix(
    const Mesh<3>& mesh,
    const tnsr::I<DataType, 3, Frame::ElementLogical>& points,
    const std::optional<size_t>& fd_to_fd_interp_order) {
  const auto number_of_target_points = get_size(get<0>(points));
  Matrix result(number_of_target_points, mesh.number_of_grid_points());

  ASSERT(
      mesh.basis()[0] == Spectral::Basis::FiniteDifference or
          fd_to_fd_interp_order == std::nullopt,
      "fd_to_fd_interp_order only applies to FD meshes. But Mesh is " << mesh);

  if (mesh.basis() == std::array{Spectral::Basis::FiniteDifference,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Basis::FiniteDifference}) {
    auto source_xi = logical_coordinates(mesh);
    DataVector xi_source{get<0>(source_xi).data(), mesh.extents(0)};
    DataVector eta_source;
    if (mesh.extents(1) == mesh.extents(0)) {
      eta_source.set_data_ref(&xi_source);
    } else {
      eta_source.destructive_resize(mesh.extents(1));
      for (size_t j = 0; j < mesh.extents(1); ++j) {
        eta_source[j] = get<1>(source_xi)[j * mesh.extents(0)];
      }
    }
    DataVector zeta_source;
    if (mesh.extents(2) == mesh.extents(0)) {
      zeta_source.set_data_ref(&xi_source);
    } else if (mesh.extents(2) == mesh.extents(1)) {
      zeta_source.set_data_ref(&eta_source);
    } else {
      zeta_source.destructive_resize(mesh.extents(2));
      for (size_t k = 0; k < mesh.extents(2); ++k) {
        zeta_source[k] =
            get<2>(source_xi)[k * mesh.extents(0) * mesh.extents(1)];
      }
    }
    constexpr size_t default_order = 1;  // linear interpolation by default
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get_element(get<0>(points), p);
      const double eta_target = get_element(get<1>(points), p);
      const double zeta_target = get_element(get<2>(points), p);
      std::vector<double> xi_stencil;
      std::vector<double> eta_stencil;
      std::vector<double> zeta_stencil;
      switch (fd_to_fd_interp_order.value_or(default_order)) {
        case 1: {
          xi_stencil = fd_stencil<1>(xi_source, xi_target);
          eta_stencil = fd_stencil<1>(eta_source, eta_target);
          zeta_stencil = fd_stencil<1>(zeta_source, zeta_target);
          break;
        }
        case 2: {
          xi_stencil = fd_stencil<2>(xi_source, xi_target);
          eta_stencil = fd_stencil<2>(eta_source, eta_target);
          zeta_stencil = fd_stencil<2>(zeta_source, zeta_target);
          break;
        }
        case 3: {
          xi_stencil = fd_stencil<3>(xi_source, xi_target);
          eta_stencil = fd_stencil<3>(eta_source, eta_target);
          zeta_stencil = fd_stencil<3>(zeta_source, zeta_target);
          break;
        }
        default:
          ERROR("Unsupported fd_to_fd_interp_order: "
                << fd_to_fd_interp_order.value_or(default_order)
                << "; expected 1, 2, or 3.");
      }
      for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
        for (size_t j = 0; j < mesh.extents(1); ++j) {
          for (size_t i = 0; i < mesh.extents(0); ++i) {
            result(p, s) = xi_stencil[i] * eta_stencil[j] * zeta_stencil[k];
            ++s;
          }
        }
      }
    }
    return result;
  } else if (mesh.basis() == std::array{Spectral::Basis::FiniteDifference,
                                        Spectral::Basis::FiniteDifference,
                                        Spectral::Basis::Cartoon}) {
    // Axial Symmetry -> call 2D implementation
    tnsr::I<DataType, 2, Frame::ElementLogical> points_2d{};
    if constexpr (std::is_same_v<DataType, DataVector>) {
      get<0>(points_2d).set_data_ref(
          make_not_null(const_cast<DataType*>(&get<0>(points))));  // NOLINT
      get<1>(points_2d).set_data_ref(
          make_not_null(const_cast<DataType*>(&get<1>(points))));  // NOLINT
    } else {
      get<0>(points_2d) = get<0>(points);
      get<1>(points_2d) = get<1>(points);
    }
    return interpolation_matrix(mesh.slice_through(0, 1), points_2d,
                                fd_to_fd_interp_order);
  } else if (mesh.basis() == std::array{Spectral::Basis::FiniteDifference,
                                        Spectral::Basis::Cartoon,
                                        Spectral::Basis::Cartoon}) {
    // Spherical Symmetry -> call 1D implementation
    tnsr::I<DataType, 1, Frame::ElementLogical> points_1d{};
    if constexpr (std::is_same_v<DataType, DataVector>) {
      get<0>(points_1d).set_data_ref(
          make_not_null(const_cast<DataType*>(&get<0>(points))));  // NOLINT
    } else {
      get<0>(points_1d) = get<0>(points);
    }
    return interpolation_matrix(mesh.slice_through(0), points_1d,
                                fd_to_fd_interp_order);
  } else if (mesh.basis()[1] == Spectral::Basis::SphericalHarmonic) {
    ASSERT(mesh.basis()[2] == Spectral::Basis::SphericalHarmonic,
           "Expected last two dimensions to each have spherical harmonic "
           "basis. Mesh = "
               << mesh);
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto [n_r, n_th, n_ph] = mesh.extents().indices();
    const auto& [m_r, m_th, m_ph] =
        cardinal_interpolator.interpolation_matrices();
    for (size_t i_ph = 0, s = 0; i_ph < n_ph; ++i_ph) {
      for (size_t i_th = 0; i_th < n_th; ++i_th) {
        for (size_t i_r = 0; i_r < n_r; ++i_r) {
          for (size_t k = 0; k < number_of_target_points; ++k) {
            result(k, s) =
                m_r(k, i_r) * (m_th(k, i_th) * m_ph(2 * k, i_ph) +
                               m_th(k, i_th + n_th) * m_ph(2 * k + 1, i_ph));
          }
          ++s;
        }
      }
    }
    return result;
  } else if (mesh.basis()[0] == Spectral::Basis::ZernikeB2) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::ZernikeB2,
           "Unexpected basis combination: " << mesh.basis());
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto& interpolation_matrices =
        cardinal_interpolator.interpolation_matrices();
    compute_zernike_b2_interpolation<3>(make_not_null(&result), mesh,
                                        interpolation_matrices,
                                        number_of_target_points);
    return result;
  } else if (mesh.basis()[0] == Spectral::Basis::ZernikeB3) {
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto& interpolation_matrices =
        cardinal_interpolator.interpolation_matrices();
    compute_zernike_b3_interpolation(make_not_null(&result), mesh,
                                     interpolation_matrices, points,
                                     number_of_target_points);
    return result;
  }
  ASSERT(mesh.basis(0) != Spectral::Basis::FiniteDifference and
             mesh.basis(1) != Spectral::Basis::FiniteDifference and
             mesh.basis(2) != Spectral::Basis::FiniteDifference,
         "Mixed FD and DG bases are not supported. Mesh = " << mesh);

  // Not FD or special basis, so use 1D spectral interpolation matrices
  const std::array<Matrix, 3> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(2), get<2>(points))}};

  // First dimension of DataVector varies fastest.
  for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
    for (size_t j = 0; j < mesh.extents(1); ++j) {
      for (size_t i = 0; i < mesh.extents(0); ++i) {
        for (size_t p = 0; p < number_of_target_points; ++p) {
          result(p, s) =
              matrices[0](p, i) * matrices[1](p, j) * matrices[2](p, k);
        }
        ++s;
      }
    }
  }
  return result;
}
}  // namespace

namespace intrp {

template <size_t Dim>
Irregular<Dim>::Irregular() = default;

template <size_t Dim>
Irregular<Dim>::Irregular(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points,
    const std::optional<size_t>& fd_to_fd_interp_order)
    : interpolation_matrix_(interpolation_matrix(source_mesh, target_points,
                                                 fd_to_fd_interp_order)) {}

template <size_t Dim>
Irregular<Dim>::Irregular(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<double, Dim, Frame::ElementLogical>& target_point,
    const std::optional<size_t>& fd_to_fd_interp_order)
    : interpolation_matrix_(interpolation_matrix(source_mesh, target_point,
                                                 fd_to_fd_interp_order)) {}

template <size_t Dim>
void Irregular<Dim>::pup(PUP::er& p) {
  p | interpolation_matrix_;
}

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<DataVector*> result,
                                 const DataVector& input) const {
  const size_t m = interpolation_matrix_.rows();
  const size_t k = interpolation_matrix_.columns();
  ASSERT(k == input.size(),
         "Number of points in 'input', "
             << input.size()
             << ",\n disagrees with the size of the source_mesh, " << k
             << ", that was passed into the constructor");
  if (result->size() != m) {
    result->destructive_resize(m);
  }
  dgemv_('n', m, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(), input.data(), 1, 0.0, result->data(),
         1);
}

template <size_t Dim>
DataVector Irregular<Dim>::interpolate(const DataVector& input) const {
  DataVector result{input.size()};
  interpolate(make_not_null(&result), input);
  return result;
}

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<ComplexDataVector*> result,
                                 const ComplexDataVector& input) const {
  const size_t m = interpolation_matrix_.rows();
  const size_t k = interpolation_matrix_.columns();
  ASSERT(k == input.size(),
         "Number of points in 'input', "
             << input.size()
             << ",\n disagrees with the size of the source_mesh, " << k
             << ", that was passed into the constructor");
  if (result->size() != m) {
    result->destructive_resize(m);
  }
  // Possible performance optimization: can possibly be written as a single
  // `dgemm` call, or might be faster using `zgemv`.
  // NOLINTBEGIN
  dgemv_('n', m, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(),
         reinterpret_cast<const double*>(input.data()), 2, 0.0,
         reinterpret_cast<double*>(result->data()), 2);
  dgemv_('n', m, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(),
         reinterpret_cast<const double*>(input.data()) + 1, 2, 0.0,
         reinterpret_cast<double*>(result->data()) + 1, 2);
  // NOLINTEND
}

template <size_t Dim>
ComplexDataVector Irregular<Dim>::interpolate(
    const ComplexDataVector& input) const {
  ComplexDataVector result{input.size()};
  interpolate(make_not_null(&result), input);
  return result;
}

namespace {

template <typename ValueType>
void span_interpolate_impl(const gsl::not_null<gsl::span<ValueType>*> result,
                           const gsl::span<const ValueType>& input,
                           const Matrix& interpolation_matrix) {
  const size_t m = interpolation_matrix.rows();
  const size_t k = interpolation_matrix.columns();
  ASSERT(input.size() % k == 0,
         "Number of points in 'input', "
             << input.size()
             << ",\n must be a multiple of the source grid points, " << k
             << ", that was passed into the constructor");
  const size_t number_of_components = input.size() / k;
  ASSERT(result->size() == number_of_components * m,
         "The result must be of size " << number_of_components * m
                                       << " but got " << result->size());
  if constexpr (std::is_same_v<ValueType, double>) {
    dgemm_<true>('N', 'N', m, number_of_components, k, 1.0,
                 interpolation_matrix.data(), interpolation_matrix.spacing(),
                 input.data(), k, 0.0, result->data(), m);
  } else if constexpr (std::is_same_v<ValueType, float>) {
    // No BLAS function exists for mixed precision so we do the matrix multiply
    // manually. A possible performance optimization would be to compute the
    // interpolation matrix in single precision and then use sgemm.
    for (size_t j = 0; j < number_of_components; ++j) {
      for (size_t i = 0; i < m; ++i) {
        float sum = interpolation_matrix.data()[i] * input[j * k];
        for (size_t l = 1; l < k; ++l) {
          sum += interpolation_matrix.data()[i + l * m] * input[l + j * k];
        }
        (*result)[i + j * m] = sum;
      }
    }
  } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
    // BLAS zgemm operates on complex matrices, so we need to copy the real
    // matrix to a complex matrix with zero imaginary part before calling zgemm.
    // Note by Nils Vu (Aug 2024): Profiling of partial derivatives showed that
    // this zgemm approach with a complex matrix is still faster than
    // transposing the complex input data and applying the real matrix with
    // dgemm (though maybe the transpose can be avoided with smarter dgemm
    // strides). Possible performance optimization: avoid the copy here by
    // caching the complex matrix.
    const blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>
        matrix_complex{interpolation_matrix};
    zgemm_<true>('N', 'N', m, number_of_components, k,
                 std::complex<double>{1.0, 0.0}, matrix_complex.data(),
                 matrix_complex.spacing(), input.data(), k,
                 std::complex<double>{0.0, 0.0}, result->data(), m);
  }
}
}  // namespace

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<gsl::span<double>*> result,
                                 const gsl::span<const double>& input) const {
  span_interpolate_impl(result, input, interpolation_matrix_);
}

template <size_t Dim>
void Irregular<Dim>::interpolate(
    const gsl::not_null<gsl::span<std::complex<double>>*> result,
    const gsl::span<const std::complex<double>>& input) const {
  span_interpolate_impl(result, input, interpolation_matrix_);
}

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<gsl::span<float>*> result,
                                 const gsl::span<const float>& input) const {
  span_interpolate_impl(result, input, interpolation_matrix_);
}

template <size_t Dim>
bool operator!=(const Irregular<Dim>& lhs, const Irregular<Dim>& rhs) {
  return not(lhs == rhs);
}

template class Irregular<1>;
template class Irregular<2>;
template class Irregular<3>;
template bool operator!=(const Irregular<1>& lhs, const Irregular<1>& rhs);
template bool operator!=(const Irregular<2>& lhs, const Irregular<2>& rhs);
template bool operator!=(const Irregular<3>& lhs, const Irregular<3>& rhs);

}  // namespace intrp
