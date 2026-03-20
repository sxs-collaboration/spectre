// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "CardinalInterpolator.hpp"

#include <array>
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
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

namespace {

template <size_t Dim, typename DataType>
std::array<Matrix, Dim> interpolation_matrices_impl(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataType, Dim, Frame::ElementLogical>& target_points) {
  std::array<Matrix, Dim> interpolation_matrices{};
  const size_t n_target_points = get_size(get<0>(target_points));
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
        switch (source_mesh.quadrature(d)) {
          case Spectral::Quadrature::Gauss: {
            const DataVector theta =
                get<0>(logical_coordinates(source_mesh.slice_through(d)));
            const DataVector cos_theta_source = cos(theta);
            const DataType cos_theta_target = cos(target_points.get(d));
            const Matrix theta_matrix = Spectral::fornberg_interpolation_matrix(
                cos_theta_target, cos_theta_source);
            const size_t n_th = source_mesh.extents(d);
            gsl::at(interpolation_matrices, d)
                .resize(n_target_points, 2 * n_th);
            const DataVector csc_theta_source = 1.0 / sin(theta);
            const DataType sin_theta_target = sin(target_points.get(d));
            for (size_t k = 0; k < n_target_points; ++k) {
              for (size_t i_th = 0; i_th < n_th; ++i_th) {
                const double factor = 0.5 * get_element(sin_theta_target, k) *
                                      csc_theta_source[i_th];
                gsl::at(interpolation_matrices, d)(k, i_th) =
                    theta_matrix(k, i_th) * (0.5 + factor);
                gsl::at(interpolation_matrices, d)(k, n_th + i_th) =
                    theta_matrix(k, i_th) * (0.5 - factor);
              }
            }
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            const DataVector phi =
                get<0>(logical_coordinates(source_mesh.slice_through(d)));
            const DataType& phi_target = target_points.get(d);
            DataVector extended_phi_target{2 * n_target_points};
            for (size_t k = 0; k < n_target_points; ++k) {
              extended_phi_target[2 * k] = get_element(phi_target, k);
              extended_phi_target[2 * k + 1] =
                  get_element(phi_target, k) + M_PI;
            }
            gsl::at(interpolation_matrices, d) =
                Spectral::fourier_interpolation_matrix(extended_phi_target,
                                                       source_mesh.extents(d));
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
        switch (source_mesh.quadrature(d)) {
          case Spectral::Quadrature::GaussRadauUpper: {
            const size_t n_r = source_mesh.extents(d);

            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            auto buffer = cpp20::make_unique_for_overwrite<double[]>(
                2 * source_mesh.extents(d) +
                (std::is_same_v<DataType, DataVector> ? 2 * n_target_points
                                                      : 0));
            DataVector radius(&buffer[0], source_mesh.extents(d));
            DataVector r_squared(&buffer[source_mesh.extents(d)],
                                 source_mesh.extents(d));
            DataType r_target;
            DataType r_target_squared;
            if constexpr (std::is_same_v<DataType, DataVector>) {
              r_target.set_data_ref(&buffer[2 * source_mesh.extents(d)],
                                    n_target_points);
              r_target_squared.set_data_ref(
                  &buffer[2 * source_mesh.extents(d) + n_target_points],
                  n_target_points);
            }
            // Need to transform from [-1, 1] logical to true [0,1]
            radius =
                0.5 *
                (get<0>(logical_coordinates(source_mesh.slice_through(d))) +
                 1.0);
            r_squared = square(radius);
            r_target = 0.5 * (target_points.get(d) + 1.0);
            r_target_squared = square(r_target);

            // Storing for both even and odd modes
            // First n_r of each column is even, second n_r is odd
            Matrix interpolation_matrix{n_target_points, 2 * n_r};
            Matrix even_matrix = Spectral::fornberg_interpolation_matrix(
                r_target_squared, r_squared);
            for (size_t k = 0; k < n_target_points; ++k) {
              for (size_t i_r = 0; i_r < n_r; ++i_r) {
                interpolation_matrix.at(k, i_r) = even_matrix.at(k, i_r);
                interpolation_matrix.at(k, i_r + n_r) =
                    even_matrix.at(k, i_r) * get_element(r_target, k) /
                    radius.at(i_r);
              }
            }
            gsl::at(interpolation_matrices, d) = interpolation_matrix;
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
void Cardinal<1>::set_zernike_b2_weights() {
  ERROR("ZernikeB2 interpolation is not supported for 1D");
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <size_t Dim>
void Cardinal<Dim>::set_zernike_b2_weights() {
  static_assert(Dim == 2 or Dim == 3,
                "ZernikeB2 interpolation only supports 2D and 3D");
  if (not using_zernike_b2_) {
    return;
  }
  const size_t n_r = source_mesh_.extents(0);
  const size_t n_phi = source_mesh_.extents(1);
  const size_t n_z = Dim == 3 ? source_mesh_.extents(2) : 1;

  ASSERT(n_phi % 2 == 1,
         "Need N_phi to be odd for stability, but got " << n_phi);

  const size_t combined_dim = n_phi * n_z;  // n_z=1 for 2D case

  const Matrix& nodal_to_modal =
      Spectral::nodal_to_modal_matrix<Spectral::Basis::Fourier,
                                      Spectral::Quadrature::Equiangular>(n_phi);

  // Pre-compute combined angular+z weights in interleaved format: [even_k0,
  // odd_k0, even_k1, odd_k1, ...]
  zernike_weights_.resize(2 * n_target_points_, combined_dim);

  for (size_t k = 0; k < n_target_points_; ++k) {
    for (size_t idx = 0; idx < combined_dim; ++idx) {
      zernike_weights_(2 * k, idx) = 0.0;      // even (radial_offset=0)
      zernike_weights_(2 * k + 1, idx) = 0.0;  // odd (radial_offset=n_r)
    }

    // m=0 mode (always uses radial_offset = 0)
    for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
      const double angular_weight =
          interpolation_matrices_[1](k, 0) * nodal_to_modal(0, i_phi);
      if constexpr (Dim == 2) {
        zernike_weights_(2 * k, i_phi) += angular_weight;
      } else {  // Dim == 3
        for (size_t i_z = 0; i_z < n_z; ++i_z) {
          const size_t idx = i_z * n_phi + i_phi;
          zernike_weights_(2 * k, idx) +=
              angular_weight * interpolation_matrices_[2](k, i_z);
        }
      }
    }

    // m>0 modes: group by radial_offset
    for (size_t i_m = 1; i_m < n_phi; ++i_m) {
      const size_t m = (i_m + 1) / 2;
      const size_t radial_offset = (m % 2) * n_r;

      for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
        const double angular_weight =
            interpolation_matrices_[1](k, i_m) * nodal_to_modal(i_m, i_phi);
        if constexpr (Dim == 2) {
          if (radial_offset == 0) {
            zernike_weights_(2 * k, i_phi) += angular_weight;
          } else {
            zernike_weights_(2 * k + 1, i_phi) += angular_weight;
          }
        } else {  // Dim == 3
          for (size_t i_z = 0; i_z < n_z; ++i_z) {
            const size_t idx = i_z * n_phi + i_phi;
            const double combined_weight =
                angular_weight * interpolation_matrices_[2](k, i_z);
            if (radial_offset == 0) {
              zernike_weights_(2 * k, idx) += combined_weight;
            } else {
              zernike_weights_(2 * k + 1, idx) += combined_weight;
            }
          }
        }
      }
    }
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
           zernike_weights_.data() + 2 * k, 2 * n_target_points_, 0.0,
           intermediate.data(), 1);
    // Angular (and z for 3D) contraction for odd modes
    dgemv_('N', n_r, combined_dim, 1.0, f_source.data(), n_r,
           zernike_weights_.data() + (2 * k + 1), 2 * n_target_points_, 0.0,
           intermediate.data() + n_r, 1);
    // Radial interpolation for both even and odd contributions
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
      using_zernike_b2_(
          alg::any_of(source_mesh_.basis(), [](const Spectral::Basis basis) {
            return basis == Spectral::Basis::ZernikeB2;
          })) {
  if (using_zernike_b2_) {
    set_zernike_b2_weights();
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
      using_zernike_b2_(
          alg::any_of(source_mesh_.basis(), [](const Spectral::Basis basis) {
            return basis == Spectral::Basis::ZernikeB2;
          })) {
  if (using_zernike_b2_) {
    set_zernike_b2_weights();
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
  p | source_mesh_;
  p | interpolation_matrices_;
  p | using_spherical_harmonics_;
  p | using_zernike_b2_;
  if (p.isUnpacking()) {
    set_zernike_b2_weights();
  }
}

template <size_t Dim>
bool operator==(const Cardinal<Dim>& lhs, const Cardinal<Dim>& rhs) {
  // Not all member variables are required:
  //  - using_spherical_haromincs_ and using_zernike_b2_ are from source_mesh_
  //  - zernike_weights_ is computable from n_target_points_ and source_mesh_
  return lhs.n_target_points_ == rhs.n_target_points_ and
         lhs.source_mesh_ == rhs.source_mesh_ and
         lhs.interpolation_matrices_ == rhs.interpolation_matrices_;
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
