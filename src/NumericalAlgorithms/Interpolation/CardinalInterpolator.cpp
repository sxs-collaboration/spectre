// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "CardinalInterpolator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationWeights.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace intrp {

namespace {

template <size_t Dim, typename DataType>
std::array<Matrix, Dim> interpolation_matrices_impl(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataType, Dim, Frame::ElementLogical>& target_points) {
  std::array<Matrix, Dim> interpolation_matrices{};
  const size_t n_target_points = get_size(get<0>(target_points));

  const auto spherical_harmonic_gauss = [&source_mesh, &target_points,
                                         &interpolation_matrices,
                                         n_target_points](const size_t dim) {
    const DataVector theta =
        get<0>(logical_coordinates(source_mesh.slice_through(dim)));
    const DataVector cos_theta_source = cos(theta);
    const DataType cos_theta_target = cos(target_points.get(dim));
    const Matrix theta_matrix = Spectral::fornberg_interpolation_matrix(
        cos_theta_target, cos_theta_source);
    const size_t n_th = source_mesh.extents(dim);
    gsl::at(interpolation_matrices, dim).resize(n_target_points, 2 * n_th);
    const DataVector csc_theta_source = 1.0 / sin(theta);
    const DataType sin_theta_target = sin(target_points.get(dim));
    for (size_t k = 0; k < n_target_points; ++k) {
      for (size_t i_th = 0; i_th < n_th; ++i_th) {
        const double factor =
            0.5 * get_element(sin_theta_target, k) * csc_theta_source[i_th];
        gsl::at(interpolation_matrices, dim)(k, i_th) =
            theta_matrix(k, i_th) * (0.5 + factor);
        gsl::at(interpolation_matrices, dim)(k, n_th + i_th) =
            theta_matrix(k, i_th) * (0.5 - factor);
      }
    }
  };

  const auto spherical_harmonic_equiangular =
      [&source_mesh, &target_points, &interpolation_matrices,
       n_target_points](const size_t dim) {
        const DataVector phi =
            get<0>(logical_coordinates(source_mesh.slice_through(dim)));
        const DataType& phi_target = target_points.get(dim);
        DataVector extended_phi_target{2 * n_target_points};
        for (size_t k = 0; k < n_target_points; ++k) {
          extended_phi_target[2 * k] = get_element(phi_target, k);
          extended_phi_target[2 * k + 1] = get_element(phi_target, k) + M_PI;
        }
        gsl::at(interpolation_matrices, dim) =
            Spectral::fourier_interpolation_matrix(extended_phi_target,
                                                   source_mesh.extents(dim));
      };

  const auto zernike_gauss_radau_upper = [&source_mesh, &target_points,
                                          &interpolation_matrices,
                                          n_target_points](const size_t dim) {
    const size_t n_r = source_mesh.extents(dim);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    auto buffer = cpp20::make_unique_for_overwrite<double[]>(
        2 * source_mesh.extents(dim) +
        (std::is_same_v<DataType, DataVector> ? 2 * n_target_points : 0));
    DataVector radius(&buffer[0], source_mesh.extents(dim));
    DataVector r_squared(&buffer[source_mesh.extents(dim)],
                         source_mesh.extents(dim));
    DataType r_target;
    DataType r_target_squared;
    if constexpr (std::is_same_v<DataType, DataVector>) {
      r_target.set_data_ref(&buffer[2 * source_mesh.extents(dim)],
                            n_target_points);
      r_target_squared.set_data_ref(
          &buffer[2 * source_mesh.extents(dim) + n_target_points],
          n_target_points);
    }
    // Need to transform from [-1, 1] logical to true [0,1]
    radius =
        0.5 *
        (get<0>(logical_coordinates(source_mesh.slice_through(dim))) + 1.0);
    r_squared = square(radius);
    r_target = 0.5 * (target_points.get(dim) + 1.0);
    r_target_squared = square(r_target);

    // Storing for both even and odd modes
    // First n_r of each column is even, second n_r is odd
    Matrix interpolation_matrix{n_target_points, 2 * n_r};
    Matrix even_matrix =
        Spectral::fornberg_interpolation_matrix(r_target_squared, r_squared);
    for (size_t k = 0; k < n_target_points; ++k) {
      for (size_t i_r = 0; i_r < n_r; ++i_r) {
        interpolation_matrix.at(k, i_r) = even_matrix.at(k, i_r);
        interpolation_matrix.at(k, i_r + n_r) =
            even_matrix.at(k, i_r) * get_element(r_target, k) / radius.at(i_r);
      }
    }
    gsl::at(interpolation_matrices, dim) = interpolation_matrix;
  };

  for (size_t d = 0; d < Dim; ++d) {
    switch (source_mesh.basis(d)) {
      case Spectral::Basis::Chebyshev:
        [[fallthrough]];
      case Spectral::Basis::Legendre: {
        const DataVector xi =
            get<0>(logical_coordinates(source_mesh.slice_through(d)));
        gsl::at(interpolation_matrices, d) =
            Spectral::fornberg_interpolation_matrix(target_points.get(d), xi);
        break;
      }
      case Spectral::Basis::SphericalHarmonic: {
        ASSERT(
            Dim > 1,
            "SphericalHarmonic interpolation got unexpected dimension, got dim "
            "= " << Dim);
        switch (source_mesh.quadrature(d)) {
          case Spectral::Quadrature::Gauss: {
            spherical_harmonic_gauss(d);
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            spherical_harmonic_equiangular(d);
            break;
          }
          default:
            ERROR(
                "Quadrature must be Gauss or Equiangular for Basis "
                "SphericalHarmonic, not "
                << source_mesh.quadrature(d));
        }
        break;
      }
      case Spectral::Basis::ZernikeB2: {
        ASSERT(Dim > 1,
               "ZernikeB2 interpolation got unexpected dimension, got dim = "
                   << Dim);
        switch (source_mesh.quadrature(d)) {
          case Spectral::Quadrature::GaussRadauUpper: {
            zernike_gauss_radau_upper(d);
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            const size_t n_phi = source_mesh.extents(d);
            const DataType& phi_target = target_points.get(d);
            Matrix fourier_basis{n_target_points, n_phi};
            for (size_t k = 0; k < n_target_points; ++k) {
              const double p = get_element(phi_target, k);
              for (size_t j = 0; j < n_phi; ++j) {
                fourier_basis(k, j) = Spectral::compute_basis_function_value<
                    Spectral::Basis::Fourier>(j, p);
              }
            }
            gsl::at(interpolation_matrices, d) = fourier_basis;
            break;
          }
          default:
            ERROR(
                "Quadrature must be GaussRadauUpper or Equiangular for Basis "
                "ZernikeB2, not "
                << source_mesh.quadrature(d));
        }
        break;
      }
      case Spectral::Basis::ZernikeB3: {
        ASSERT(Dim == 3,
               "ZernikeB3 interpolation got unexpected dimension, got dim = "
                   << Dim);
        switch (source_mesh.quadrature(d)) {
          case Spectral::Quadrature::GaussRadauUpper: {
            zernike_gauss_radau_upper(d);
            break;
          }
          case Spectral::Quadrature::Gauss: {
            spherical_harmonic_gauss(d);
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            spherical_harmonic_equiangular(d);
            break;
          }
          default:
            ERROR(
                "Quadrature must be GaussRadauUpper, Gauss, or Equiangular "
                "for Basis ZernikeB3, not "
                << source_mesh.quadrature(d));
        }
        break;
      }
      default:
        ERROR("Basis " << source_mesh.basis(d)
                       << " is not supported by the Cardinal interpolator.");
    }
  }
  return interpolation_matrices;
}
}  // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#elif defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <>
DataVector Cardinal<1>::interpolate_zernike_b2(
    const DataVector& /*f_source*/) const {
  ERROR("ZernikeB2 interpolation is not supported for 1D");
}

template <>
void Cardinal<1>::compute_zernike_b2_weights(
    gsl::not_null<Matrix*> /*weights*/, const Mesh<1>& /*source_mesh*/,
    const std::array<Matrix, 1>& /*interpolation_matrices*/,
    size_t /*n_target_points*/) {
  ERROR("ZernikeB2 interpolation is not supported for 1D");
}

template <>
void Cardinal<1>::set_zernike_b2_weights() {
  ERROR("ZernikeB2 interpolation is not supported for 1D");
}

template <>
void Cardinal<1>::compute_zernike_b3_weights(
    gsl::not_null<Matrix*> /*weights*/, const Mesh<1>& /*source_mesh*/,
    const tnsr::I<DataVector, 1, Frame::ElementLogical>& /*target_points*/,
    const ylm::Spherepack& /*b3_ylm*/, size_t /*n_target_points*/) {
  ERROR("ZernikeB3 interpolation is not supported for 1D");
}

template <>
DataVector Cardinal<1>::interpolate_zernike_b3(
    const DataVector& /*f_source*/) const {
  ERROR("ZernikeB3 interpolation is not supported for 1D");
}

template <>
void Cardinal<1>::set_zernike_b3_weights() {
  ERROR("ZernikeB3 interpolation is not supported for 1D");
}

template <>
void Cardinal<2>::compute_zernike_b3_weights(
    gsl::not_null<Matrix*> /*weights*/, const Mesh<2>& /*source_mesh*/,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& /*target_points*/,
    const ylm::Spherepack& /*b3_ylm*/, size_t /*n_target_points*/) {
  ERROR("ZernikeB3 interpolation is not supported for 2D");
}

template <>
DataVector Cardinal<2>::interpolate_zernike_b3(
    const DataVector& /*f_source*/) const {
  ERROR("ZernikeB3 interpolation is not supported for 2D");
}

template <>
void Cardinal<2>::set_zernike_b3_weights() {
  ERROR("ZernikeB3 interpolation is not supported for 2D");
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <size_t Dim>
void Cardinal<Dim>::compute_zernike_b2_weights(
    const gsl::not_null<Matrix*> weights, const Mesh<Dim>& source_mesh,
    const std::array<Matrix, Dim>& interpolation_matrices,
    const size_t n_target_points) {
  static_assert(Dim == 2 or Dim == 3,
                "ZernikeB2 interpolation only supports 2D and 3D");
  const size_t n_r = source_mesh.extents(0);
  const size_t n_phi = source_mesh.extents(1);
  const size_t n_z = Dim == 3 ? source_mesh.extents(2) : 1;

  ASSERT(n_phi % 2 == 1,
         "Need N_phi to be odd for stability, but got " << n_phi);

  const size_t combined_dim = n_phi * n_z;  // n_z=1 for 2D case

  const Matrix& nodal_to_modal =
      Spectral::nodal_to_modal_matrix<Spectral::Basis::Fourier,
                                      Spectral::Quadrature::Equiangular>(n_phi);
  auto& zernike_b2_weights = *weights;

  // Pre-compute combined angular+z weights in interleaved format:
  // [even_k0, odd_k0, even_k1, odd_k1, ... ]
  zernike_b2_weights.resize(2 * n_target_points, combined_dim);
  for (size_t k = 0; k < n_target_points; ++k) {
    for (size_t idx = 0; idx < combined_dim; ++idx) {
      zernike_b2_weights(2 * k, idx) = 0.0;      // even (radial_offset=0)
      zernike_b2_weights(2 * k + 1, idx) = 0.0;  // odd (radial_offset=n_r)
    }

    // m=0 mode (always uses radial_offset = 0)
    for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
      const double angular_weight =
          interpolation_matrices[1](k, 0) * nodal_to_modal(0, i_phi);
      if constexpr (Dim == 2) {
        zernike_b2_weights(2 * k, i_phi) += angular_weight;
      } else {  // Dim == 3
        for (size_t i_z = 0; i_z < n_z; ++i_z) {
          const size_t idx = i_z * n_phi + i_phi;
          zernike_b2_weights(2 * k, idx) +=
              angular_weight * interpolation_matrices[2](k, i_z);
        }
      }
    }

    // m>0 modes: group by radial_offset
    for (size_t i_m = 1; i_m < n_phi; ++i_m) {
      const size_t m = (i_m + 1) / 2;
      const size_t radial_offset = (m % 2) * n_r;

      for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
        const double angular_weight =
            interpolation_matrices[1](k, i_m) * nodal_to_modal(i_m, i_phi);
        if constexpr (Dim == 2) {
          if (radial_offset == 0) {
            zernike_b2_weights(2 * k, i_phi) += angular_weight;
          } else {
            zernike_b2_weights(2 * k + 1, i_phi) += angular_weight;
          }
        } else {  // Dim == 3
          for (size_t i_z = 0; i_z < n_z; ++i_z) {
            const size_t idx = i_z * n_phi + i_phi;
            const double combined_weight =
                angular_weight * interpolation_matrices[2](k, i_z);
            if (radial_offset == 0) {
              zernike_b2_weights(2 * k, idx) += combined_weight;
            } else {
              zernike_b2_weights(2 * k + 1, idx) += combined_weight;
            }
          }
        }
      }
    }
  }
}

template <size_t Dim>
void Cardinal<Dim>::set_zernike_b2_weights() {
  if (using_zernike_b2_) {
    Cardinal<Dim>::compute_zernike_b2_weights(
        make_not_null(&zernike_b2_weights_), source_mesh_,
        interpolation_matrices_, n_target_points_);
  }
}

template <size_t Dim>
DataVector Cardinal<Dim>::interpolate_zernike_b2(
    const DataVector& f_source) const {
  static_assert(Dim == 2 or Dim == 3,
                "ZernikeB2 interpolation only supports 2D and 3D");

  ASSERT(f_source.size() == source_mesh_.number_of_grid_points(),
         "Size of source data ("
             << f_source.size() << ") does not match size of source mesh ("
             << source_mesh_.number_of_grid_points() << ")");

  const size_t n_r = source_mesh_.extents(0);
  const size_t n_phi = source_mesh_.extents(1);
  const size_t n_z = Dim == 3 ? source_mesh_.extents(2) : 1;

  const size_t combined_dim = n_phi * n_z;  // n_z=1 for 2D case

  DataVector result{n_target_points_};
  DataVector intermediate{2 * n_r};

  for (size_t k = 0; k < n_target_points_; ++k) {
    // Angular (and z for 3D) contraction for even modes
    dgemv_('N', n_r, combined_dim, 1.0, f_source.data(), n_r,
           zernike_b2_weights_.data() + 2 * k, 2 * n_target_points_, 0.0,
           intermediate.data(), 1);
    // Angular (and z for 3D) contraction for odd modes
    dgemv_('N', n_r, combined_dim, 1.0, f_source.data(), n_r,
           zernike_b2_weights_.data() + (2 * k + 1), 2 * n_target_points_, 0.0,
           intermediate.data() + n_r, 1);
    // Radial interpolation for both even and odd contributions
    result[k] = ddot_(2 * n_r, interpolation_matrices_[0].data() + k,
                      n_target_points_, intermediate.data(), 1);
  }
  return result;
}

template <>
void Cardinal<3>::compute_zernike_b3_weights(
    const gsl::not_null<Matrix*> weights, const Mesh<3>& source_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& target_points,
    const ylm::Spherepack& b3_ylm, const size_t n_target_points) {
  const size_t l_max = source_mesh.extents(1) - 1;
  const size_t m_max = (source_mesh.extents(2) - 1) / 2;

  ASSERT(l_max < 2 * source_mesh.extents(0) - 1,
         "ZernikeB3 radial resolution is insufficient for the requested "
         "angular resolution. Need l_max < 2*n_r-1, but l_max="
             << l_max << " and n_r=" << source_mesh.extents(0));
  ASSERT(b3_ylm.l_max() == l_max and b3_ylm.m_max() == m_max,
         "Passed Spherepack is incorrect size for compute_zernike_b3_weights, "
         "got b3_ylm's l_max = "
             << b3_ylm.l_max() << " and m_max = " << b3_ylm.m_max()
             << " while the mesh has l_max = " << l_max
             << ", m_max = " << m_max);

  auto& zernike_b3_weights = *weights;
  const size_t n_spectral = b3_ylm.spectral_size();

  zernike_b3_weights.resize(2 * n_target_points, n_spectral);
  // zero out the matrix
  zernike_b3_weights.reset();

  // For each target point, evaluate every SPHEREPACK basis function using a
  // unit spectral vector, then store by l-parity (row 2k = even-l, 2k+1 =
  // odd-l).
  DataVector unit_spec(n_spectral, 0.0);
  ylm::SpherepackIterator iter{l_max, m_max};

  for (size_t k = 0; k < n_target_points; ++k) {
    const auto info =
        b3_ylm.set_up_interpolation_info<double>(std::array<double, 2>{
            get<1>(target_points)[k], get<2>(target_points)[k]});
    iter.reset();
    while (iter) {
      const size_t s = iter();
      const size_t l = iter.l();
      unit_spec[s] = 1.0;
      double val = 0.0;
      b3_ylm.interpolate_from_coefs(make_not_null(&val), unit_spec, info);
      unit_spec[s] = 0.0;
      zernike_b3_weights(2 * k + (l % 2), s) = val;
      ++iter;
    }
  }
}

template <>
void Cardinal<3>::set_zernike_b3_weights() {
  if (using_zernike_b3_) {
    const size_t l_max = source_mesh_.extents(1) - 1;
    const size_t m_max = (source_mesh_.extents(2) - 1) / 2;
    b3_ylm_.emplace(l_max, m_max);
    if (not target_points_.has_value()) {
      ERROR(
          "CardinalInterpolator does not have target_points_ member variable "
          "set, required for B3 interpolation.");
    }
    Cardinal<3>::compute_zernike_b3_weights(make_not_null(&zernike_b3_weights_),
                                            source_mesh_, *target_points_,
                                            *b3_ylm_, n_target_points_);
  }
}

template <>
DataVector Cardinal<3>::interpolate_zernike_b3(
    const DataVector& f_source) const {
  ASSERT(f_source.size() == source_mesh_.number_of_grid_points(),
         "Size of source data ("
             << f_source.size() << ") does not match size of source mesh ("
             << source_mesh_.number_of_grid_points() << ")");

  const size_t n_r = source_mesh_.extents(0);
  const auto& ylm = *b3_ylm_;
  const size_t n_spectral = ylm.spectral_size();

  // SH analysis for all radial shells at once.
  // Layout after transform: spec_buf[s * n_r + i_r] = SPHEREPACK coefficient
  // for mode s at radial shell i_r.
  DataVector spec_buf(n_spectral * n_r);
  ylm.phys_to_spec_all_offsets(make_not_null(spec_buf.data()),
                               make_not_null(f_source.data()), n_r);

  DataVector result{n_target_points_};
  // intermediate[0:n_r] = angular contraction of even-l modes
  // intermediate[n_r:2*n_r] = angular contraction of odd-l modes
  DataVector intermediate{2 * n_r};

  for (size_t k = 0; k < n_target_points_; ++k) {
    // Contract spec_buf (viewed as n_r x n_spectral column-major) with
    // zernike_b3_weights_ row 2k (even-l) -> intermediate[0:n_r].
    // spec_buf[s * n_r + i_r]: BLAS sees this as an n_r x n_spectral matrix
    // in column-major (each column = one mode's radial strip of length n_r).
    dgemv_('N', n_r, n_spectral, 1.0, spec_buf.data(), n_r,
           zernike_b3_weights_.data() + 2 * k, 2 * n_target_points_, 0.0,
           intermediate.data(), 1);
    // Same for odd-l modes -> intermediate[n_r:2*n_r].
    dgemv_('N', n_r, n_spectral, 1.0, spec_buf.data(), n_r,
           zernike_b3_weights_.data() + (2 * k + 1), 2 * n_target_points_, 0.0,
           intermediate.data() + n_r, 1);
    // Radial interpolation across even and odd contributions (same as B2).
    result[k] = ddot_(2 * n_r, interpolation_matrices_[0].data() + k,
                      n_target_points_, intermediate.data(), 1);
  }
  return result;
}

template <size_t Dim>
Cardinal<Dim>::Cardinal() = default;

template <size_t Dim>
Cardinal<Dim>::Cardinal(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points)
    : n_target_points_(get<0>(target_points).size()),
      source_mesh_(source_mesh),
      interpolation_matrices_(
          interpolation_matrices_impl(source_mesh_, target_points)),
      using_spherical_harmonics_(
          alg::any_of(source_mesh_.basis(),
                      [](const Spectral::Basis basis) {
                        return basis == Spectral::Basis::SphericalHarmonic;
                      })),
      using_zernike_b2_(alg::any_of(source_mesh_.basis(),
                                    [](const Spectral::Basis basis) {
                                      return basis ==
                                             Spectral::Basis::ZernikeB2;
                                    })),
      using_zernike_b3_(
          alg::any_of(source_mesh_.basis(), [](const Spectral::Basis basis) {
            return basis == Spectral::Basis::ZernikeB3;
          })) {
  if constexpr (Dim > 1) {
    if (using_zernike_b2_) {
      set_zernike_b2_weights();
    }
    if constexpr (Dim == 3) {
      if (using_zernike_b3_) {
        target_points_ = target_points;
        set_zernike_b3_weights();
      }
    }
  }
}

template <size_t Dim>
Cardinal<Dim>::Cardinal(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<double, Dim, Frame::ElementLogical>& target_point)
    : n_target_points_(1),
      source_mesh_(source_mesh),
      interpolation_matrices_(
          interpolation_matrices_impl(source_mesh_, target_point)),
      using_spherical_harmonics_(
          alg::any_of(source_mesh_.basis(),
                      [](const Spectral::Basis basis) {
                        return basis == Spectral::Basis::SphericalHarmonic;
                      })),
      using_zernike_b2_(alg::any_of(source_mesh_.basis(),
                                    [](const Spectral::Basis basis) {
                                      return basis ==
                                             Spectral::Basis::ZernikeB2;
                                    })),
      using_zernike_b3_(
          alg::any_of(source_mesh_.basis(), [](const Spectral::Basis basis) {
            return basis == Spectral::Basis::ZernikeB3;
          })) {
  if constexpr (Dim > 1) {
    if (using_zernike_b2_) {
      set_zernike_b2_weights();
    }
    if constexpr (Dim == 3) {
      if (using_zernike_b3_) {
        target_points_ = tnsr::I<DataVector, 3, Frame::ElementLogical>{1_st};
        get<0>(*target_points_) = get<0>(target_point);
        get<1>(*target_points_) = get<1>(target_point);
        get<2>(*target_points_) = get<2>(target_point);
        set_zernike_b3_weights();
      }
    }
  }
}

template <>
DataVector Cardinal<1>::interpolate(const DataVector& f_source) const {
  const size_t n_source_points = source_mesh_.number_of_grid_points();
  ASSERT(f_source.size() == n_source_points,
         "Size of source data ("
             << f_source.size() << ") does not match size of source mesh ("
             << n_source_points
             << ") that this interpolator was constructed with.");
  DataVector result{n_target_points_};
  dgemv_('N', n_target_points_, n_source_points, 1.0,
         interpolation_matrices_[0].data(),
         interpolation_matrices_[0].spacing(), f_source.data(), 1, 0.0,
         result.data(), 1);
  return result;
}

template <>
DataVector Cardinal<2>::interpolate(const DataVector& f_source) const {
  ASSERT(f_source.size() == source_mesh_.number_of_grid_points(),
         "Size of source data ("
             << f_source.size() << ") does not match size of source mesh ("
             << source_mesh_.number_of_grid_points()
             << ") that this interpolator was constructed with.");
  DataVector result{n_target_points_};
  if (using_spherical_harmonics_) {
    const auto [n_th, n_ph] = source_mesh_.extents().indices();
    DataVector intermediate_result{2 * n_th};
    for (size_t k = 0; k < n_target_points_; ++k) {
      dgemv_('N', n_th, n_ph, 1.0, f_source.data(), n_th,
             interpolation_matrices_[1].data() + 2 * k, 2 * n_target_points_,
             0.0, intermediate_result.data(), 1);
      dgemv_('N', n_th, n_ph, 1.0, f_source.data(), n_th,
             interpolation_matrices_[1].data() + (2 * k + 1),
             2 * n_target_points_, 0.0, intermediate_result.data() + n_th, 1);
      result[k] = ddot_(2 * n_th, interpolation_matrices_[0].data() + k,
                        n_target_points_, intermediate_result.data(), 1);
    }
    return result;
  } else if (using_zernike_b2_) {
    return interpolate_zernike_b2(f_source);
  }
  const auto [n_xi, n_eta] = source_mesh_.extents().indices();
  Matrix intermediate_result{n_target_points_, n_eta};
  dgemm_('N', 'N', n_target_points_, n_eta, n_xi, 1.0,
         interpolation_matrices_[0].data(),
         interpolation_matrices_[0].spacing(), f_source.data(), n_xi, 0.0,
         intermediate_result.data(), n_target_points_);
  for (size_t k = 0; k < n_target_points_; ++k) {
    result[k] =
        ddot_(n_eta, interpolation_matrices_[1].data() + k, n_target_points_,
              intermediate_result.data() + k, n_target_points_);
  }
  return result;
}

template <>
DataVector Cardinal<3>::interpolate(const DataVector& f_source) const {
  ASSERT(f_source.size() == source_mesh_.number_of_grid_points(),
         "Size of source data ("
             << f_source.size() << ") does not match size of source mesh ("
             << source_mesh_.number_of_grid_points()
             << ") that this interpolator was constructed with.");
  const auto [n0, n1, n2] = source_mesh_.extents().indices();
  Matrix intermediate_matrix{n_target_points_, n1 * n2};
  dgemm_('N', 'N', n_target_points_, n1 * n2, n0, 1.0,
         interpolation_matrices_[0].data(),
         interpolation_matrices_[0].spacing(), f_source.data(), n0, 0.0,
         intermediate_matrix.data(), n_target_points_);
  auto transpose = intermediate_matrix.transpose();
  DataVector result{n_target_points_};
  if (using_spherical_harmonics_) {
    DataVector intermediate_vector{2 * n1};
    for (size_t k = 0; k < n_target_points_; ++k) {
      dgemv_('N', n1, n2, 1.0, transpose.data() + k * n1 * n2, n1,
             interpolation_matrices_[2].data() + 2 * k, 2 * n_target_points_,
             0.0, intermediate_vector.data(), 1);
      dgemv_('N', n1, n2, 1.0, transpose.data() + k * n1 * n2, n1,
             interpolation_matrices_[2].data() + (2 * k + 1),
             2 * n_target_points_, 0.0, intermediate_vector.data() + n1, 1);
      result[k] = ddot_(2 * n1, interpolation_matrices_[1].data() + k,
                        n_target_points_, intermediate_vector.data(), 1);
    }
    return result;
  } else if (using_zernike_b2_) {
    return interpolate_zernike_b2(f_source);
  } else if (using_zernike_b3_) {
    return interpolate_zernike_b3(f_source);
  }
  DataVector intermediate_vector{n2};
  for (size_t k = 0; k < n_target_points_; ++k) {
    dgemv_('T', n1, n2, 1.0, transpose.data() + k * n1 * n2, n1,
           interpolation_matrices_[1].data() + k, n_target_points_, 0.0,
           intermediate_vector.data(), 1);
    result[k] = ddot_(n2, interpolation_matrices_[2].data() + k,
                      n_target_points_, intermediate_vector.data(), 1);
  }
  return result;
}

template <size_t Dim>
const std::array<Matrix, Dim>& Cardinal<Dim>::interpolation_matrices() const {
  return interpolation_matrices_;
}

template <size_t Dim>
void Cardinal<Dim>::pup(PUP::er& p) {
  p | n_target_points_;
  p | target_points_;
  p | source_mesh_;
  p | interpolation_matrices_;
  p | using_spherical_harmonics_;
  p | using_zernike_b2_;
  p | using_zernike_b3_;
  if (p.isUnpacking()) {
    if constexpr (Dim > 1) {
      set_zernike_b2_weights();
      if constexpr (Dim == 3) {
        set_zernike_b3_weights();
      }
    }
  }
}

template <size_t Dim>
bool operator==(const Cardinal<Dim>& lhs, const Cardinal<Dim>& rhs) {
  // Not all member variables are required:
  //  - using_spherical_harmonics_, using_zernike_b2_, using_zernike_b3_ are
  //    from source_mesh_
  //  - zernike_b2_weights_ is computable from n_target_points_ and source_mesh_
  //  - zernike_b3_weights_ is computable from n_target_points_, source_mesh_,
  //    and target_points_
  return lhs.n_target_points_ == rhs.n_target_points_ and
         lhs.source_mesh_ == rhs.source_mesh_ and
         lhs.interpolation_matrices_ == rhs.interpolation_matrices_ and
         lhs.target_points_ == rhs.target_points_;
}

template <size_t Dim>
bool operator!=(const Cardinal<Dim>& lhs, const Cardinal<Dim>& rhs) {
  return not(lhs == rhs);
}

template class Cardinal<1>;
template class Cardinal<2>;
template class Cardinal<3>;
template bool operator==(const Cardinal<1>& lhs, const Cardinal<1>& rhs);
template bool operator==(const Cardinal<2>& lhs, const Cardinal<2>& rhs);
template bool operator==(const Cardinal<3>& lhs, const Cardinal<3>& rhs);
template bool operator!=(const Cardinal<1>& lhs, const Cardinal<1>& rhs);
template bool operator!=(const Cardinal<2>& lhs, const Cardinal<2>& rhs);
template bool operator!=(const Cardinal<3>& lhs, const Cardinal<3>& rhs);

}  // namespace intrp
