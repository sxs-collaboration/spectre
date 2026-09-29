// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace {
// For a spherical-harmonic (Spherepack) angular basis, the angular logical
// derivative is the Pfaffian gradient (d/dtheta, csc(theta) d/dphi), which
// couples the two angular dimensions and is therefore not a per-dimension
// differentiation matrix. The weak divergence applies the transpose of the
// strong logical derivative (see `differentiation_matrix_transpose`). Here we
// build, once per `l_max`, the explicit transposes of the two angular gradient
// operators acting on the combined angular grid. Row `s` of each transpose is
// the corresponding component of the Pfaffian gradient of the `s`-th angular
// unit vector, which makes the matrices the (Euclidean) transposes of the
// strong gradient used in `logical_partial_derivative`.
const std::pair<Matrix, Matrix>&
spherical_harmonic_weak_divergence_transpose_matrices(const size_t l_max) {
  static const auto cache =
      make_static_cache<CacheRange<2_st, 151_st>>([](const size_t local_l_max) {
        const auto& ylm = ylm::get_spherepack_cache(local_l_max);
        const size_t num_angular_points = ylm.physical_size();
        Matrix theta_transpose(num_angular_points, num_angular_points);
        Matrix phi_transpose(num_angular_points, num_angular_points);
        DataVector unit_vector(num_angular_points, 0.0);
        for (size_t s = 0; s < num_angular_points; ++s) {
          unit_vector[s] = 1.0;
          const auto gradient = ylm.gradient(unit_vector);
          for (size_t b = 0; b < num_angular_points; ++b) {
            theta_transpose(s, b) = get<0>(gradient)[b];
            phi_transpose(s, b) = get<1>(gradient)[b];
          }
          unit_vector[s] = 0.0;
        }
        return std::pair{std::move(theta_transpose), std::move(phi_transpose)};
      });
  return cache(l_max);
}
}  // namespace

namespace weak_divergence_detail {
// The weak differentiation matrix in a dimension is `M^{-1} D^T M`, where `M`
// is the diagonal mass matrix (quadrature weights) and `D` is the strong
// differentiation matrix (see `weak_flux_differentiation_matrix`). For the two
// coupled angular directions of a spherical-harmonic basis the analog is
// `M_ang^{-1} G^T M_ang`, where `G` is the strong Pfaffian gradient (whose
// transpose is built above) and `M_ang` are the angular integration weights.
// These are obtained from the transpose matrices by scaling row `i` and column
// `j` with `1 / w[i]` and `w[j]`, respectively.
const std::pair<Matrix, Matrix>&
spherical_harmonic_weak_flux_differentiation_matrices(const size_t l_max) {
  static const auto cache =
      make_static_cache<CacheRange<2_st, 151_st>>([](const size_t local_l_max) {
        const auto& [theta_transpose, phi_transpose] =
            spherical_harmonic_weak_divergence_transpose_matrices(local_l_max);
        const std::vector<double>& weights =
            ylm::get_spherepack_cache(local_l_max).integration_weights();
        const size_t num_angular_points = theta_transpose.rows();
        Matrix theta_weak_div = theta_transpose;
        Matrix phi_weak_div = phi_transpose;
        for (size_t i = 0; i < num_angular_points; ++i) {
          for (size_t j = 0; j < num_angular_points; ++j) {
            const double factor = weights[j] / weights[i];
            theta_weak_div(i, j) *= factor;
            phi_weak_div(i, j) *= factor;
          }
        }
        return std::pair{std::move(theta_weak_div), std::move(phi_weak_div)};
      });
  return cache(l_max);
}
}  // namespace weak_divergence_detail

template <typename ResultTensor, typename FluxTensor, size_t Dim>
void logical_weak_divergence(const gsl::not_null<ResultTensor*> div_flux,
                             const FluxTensor& flux, const Mesh<Dim>& mesh) {
  // Note: This function hasn't been optimized much at all. Feel free to
  // optimize if needed!
  static const Matrix identity_matrix{};
  if constexpr (Dim == 3) {
    if (mesh.basis(1) == Spectral::Basis::SphericalHarmonic) {
      if constexpr (std::is_same_v<typename FluxTensor::type, DataVector>) {
        // Radial direction (xi) is a standard Legendre slice; the two angular
        // directions (theta, phi) are handled together with the transpose
        // Spherepack gradient (see above). We apply the angular operators with
        // a combined angular index of size `num_angular_points`.
        const size_t num_radial_points = mesh.extents(0);
        const auto& [theta_transpose, phi_transpose] =
            spherical_harmonic_weak_divergence_transpose_matrices(
                mesh.extents(1) - 1);
        const size_t num_angular_points = theta_transpose.rows();
        const Index<2> angular_extents{num_radial_points, num_angular_points};
        const Matrix& xi_transpose =
            Spectral::differentiation_matrix_transpose(mesh.slice_through(0));
        for (size_t storage_index = 0; storage_index < div_flux->size();
             ++storage_index) {
          const auto div_flux_index = div_flux->get_tensor_index(storage_index);
          // Radial (xi) contribution
          div_flux->get(div_flux_index) += apply_matrices(
              std::array<std::reference_wrapper<const Matrix>, 3>{
                  {std::cref(xi_transpose), std::cref(identity_matrix),
                   std::cref(identity_matrix)}},
              flux.get(prepend(div_flux_index, 0_st)), mesh.extents());
          // Angular (theta) contribution
          div_flux->get(div_flux_index) += apply_matrices(
              std::array<std::reference_wrapper<const Matrix>, 2>{
                  {std::cref(identity_matrix), std::cref(theta_transpose)}},
              flux.get(prepend(div_flux_index, 1_st)), angular_extents);
          // Angular (phi) contribution
          div_flux->get(div_flux_index) += apply_matrices(
              std::array<std::reference_wrapper<const Matrix>, 2>{
                  {std::cref(identity_matrix), std::cref(phi_transpose)}},
              flux.get(prepend(div_flux_index, 2_st)), angular_extents);
        }
        return;
      } else {
        ERROR(
            "logical_weak_divergence with a spherical-harmonic basis is not "
            "implemented for complex data.");
      }
    }
  }
  for (size_t d = 0; d < Dim; ++d) {
    auto matrices = make_array<Dim>(std::cref(identity_matrix));
    gsl::at(matrices, d) =
        Spectral::differentiation_matrix_transpose(mesh.slice_through(d));
    for (size_t storage_index = 0; storage_index < div_flux->size();
         ++storage_index) {
      const auto div_flux_index = div_flux->get_tensor_index(storage_index);
      const auto flux_index = prepend(div_flux_index, d);
      div_flux->get(div_flux_index) +=
          apply_matrices(matrices, flux.get(flux_index), mesh.extents());
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TENSOR(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION_SCALAR(r, data)                                     \
  template void logical_weak_divergence(                                  \
      const gsl::not_null<Scalar<DTYPE(data)>*> div_flux,                 \
      const tnsr::I<DTYPE(data), DIM(data), Frame::ElementLogical>& flux, \
      const Mesh<DIM(data)>& mesh);
#define INSTANTIATION_TENSOR(r, data)                                    \
  template void logical_weak_divergence(                                 \
      const gsl::not_null<tnsr::TENSOR(data) < DTYPE(data), DIM(data),   \
                          Frame::Inertial>* > div_flux,                  \
      const TensorMetafunctions::prepend_spatial_index<                  \
          tnsr::TENSOR(data) < DTYPE(data), DIM(data), Frame::Inertial>, \
      DIM(data), UpLo::Up, Frame::ElementLogical > &flux,                \
      const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION_SCALAR, (DataVector, ComplexDataVector),
                        (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATION_TENSOR, (DataVector, ComplexDataVector),
                        (1, 2, 3), (i, I, aa))

template void logical_weak_divergence(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3, Frame::Inertial>*>
        div_flux,
    const TensorMetafunctions::prepend_spatial_index<
        tnsr::aa<ComplexDataVector, 3, Frame::Inertial>, 2, UpLo::Up,
        Frame::ElementLogical>& flux,
    const Mesh<2>& mesh);

#undef INSTANTIATION_SCALAR
#undef INSTANTIATION_TENSOR
#undef DTYPE
#undef DIM
#undef TENSOR
