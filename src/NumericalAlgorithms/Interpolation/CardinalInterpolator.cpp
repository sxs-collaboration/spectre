// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "CardinalInterpolator.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationWeights.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
            fornberg_interpolation_matrix(target_points.get(d), xi);
        break;
      }
      case Spectral::Basis::SphericalHarmonic: {
        switch (source_mesh.quadrature(d)) {
          case Spectral::Quadrature::Gauss: {
            const DataVector theta =
                get<0>(logical_coordinates(source_mesh.slice_through(d)));
            const DataVector cos_theta_source = cos(theta);
            const DataType cos_theta_target = cos(target_points.get(d));
            const Matrix theta_matrix = fornberg_interpolation_matrix(
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
            gsl::at(interpolation_matrices, d) = fourier_interpolation_matrix(
                extended_phi_target, source_mesh.extents(d));
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
      default:
        ERROR("Basis " << source_mesh.basis(d)
                       << " is not supported by the Cardinal interpolator.");
    }
  }
  return interpolation_matrices;
}

}  // namespace

template <size_t Dim>
Cardinal<Dim>::Cardinal(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points)
    : n_target_points_(get<0>(target_points).size()),
      source_mesh_(source_mesh),
      interpolation_matrices_(
          interpolation_matrices_impl(source_mesh_, target_points)),
      using_spherical_harmonics_(
          alg::any_of(source_mesh_.basis(), [](const Spectral::Basis basis) {
            return basis == Spectral::Basis::SphericalHarmonic;
          })) {}

template <size_t Dim>
Cardinal<Dim>::Cardinal(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<double, Dim, Frame::ElementLogical>& target_point)
    : n_target_points_(1),
      source_mesh_(source_mesh),
      interpolation_matrices_(
          interpolation_matrices_impl(source_mesh_, target_point)),
      using_spherical_harmonics_(
          alg::any_of(source_mesh_.basis(), [](const Spectral::Basis basis) {
            return basis == Spectral::Basis::SphericalHarmonic;
          })) {}

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
bool operator==(const Cardinal<Dim>& lhs, const Cardinal<Dim>& rhs) {
  return lhs.interpolation_matrices() == rhs.interpolation_matrices();
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
