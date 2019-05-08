// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperGr.hpp"

#include <array>
#include <cmath>  // IWYU pragma: keep
#include <cstddef>
#include <utility>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindGeneralizedEigenvalues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

// IWYU pragma: no_include <complex>

// IWYU pragma: no_forward_declare Strahlkorper
// IWYU pragma: no_forward_declare Tensor

/// \cond
// Functions used by StrahlkorperGr::dimensionful_spin_magnitude
namespace {
// Find the 2D surface metric by inserting the tangents \f$\partial_\theta\f$
// and \f$\partial_\phi\f$ into the slots of the 3D spatial metric
template <typename Fr>
tnsr::ii<DataVector, 2, Frame::Spherical<Fr>> get_surface_metric(
    const tnsr::ii<DataVector, 3, Fr>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Fr>& tangents,
    const Scalar<DataVector>& sin_theta) noexcept {
  auto surface_metric =
      make_with_value<tnsr::ii<DataVector, 2, Frame::Spherical<Fr>>>(
          get<0, 0>(spatial_metric), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get<0, 1>(surface_metric) += spatial_metric.get(i, j) *
                                   tangents.get(i, 0) * tangents.get(j, 1) *
                                   get(sin_theta);
    }
    // Use symmetry to sum over fewer terms for the 0,0 and 1,1 components
    get<0, 0>(surface_metric) +=
        spatial_metric.get(i, i) * square(tangents.get(i, 0));
    get<1, 1>(surface_metric) += spatial_metric.get(i, i) *
                                 square(tangents.get(i, 1)) *
                                 square(get(sin_theta));
    for (size_t j = i + 1; j < 3; ++j) {
      get<0, 0>(surface_metric) += 2.0 * spatial_metric.get(i, j) *
                                   tangents.get(i, 0) * tangents.get(j, 0);
      get<1, 1>(surface_metric) += 2.0 * spatial_metric.get(i, j) *
                                   tangents.get(i, 1) * tangents.get(j, 1) *
                                   square(get(sin_theta));
    }
  }
  return surface_metric;
}

// Compute the trace of Christoffel 2nd kind on the horizon
template <typename Fr>
tnsr::I<DataVector, 2, Frame::Spherical<Fr>> get_trace_christoffel_second_kind(
    const tnsr::ii<DataVector, 2, Frame::Spherical<Fr>>& surface_metric,
    const tnsr::II<DataVector, 2, Frame::Spherical<Fr>>& inverse_surface_metric,
    const Scalar<DataVector>& sin_theta, const YlmSpherepack& ylm) noexcept {
  const Scalar<DataVector> cos_theta{cos(ylm.theta_phi_points()[0])};

  // Because the surface metric components are not representable in terms
  // of scalar spherical harmonics, you can't naively take first derivatives.
  // To avoid potentially large numerical errors, actually differentiate
  // square(sin_theta) * the metric component, then
  // compute from that the gradient of just the metric component itself.
  //
  // Note: the method implemented here works with YlmSpherepack but will fail
  // for other expansions that, unlike Spherepack, include a collocation
  // point at theta = 0. Before switching to such an expansion, first
  // reimplement this code to avoid dividing by sin(theta).
  //
  // Note: YlmSpherepack gradients are flat-space Pfaffian derivatives.
  auto grad_surface_metric_theta_theta =
      ylm.gradient(square(get(sin_theta)) * get<0, 0>(surface_metric));
  get<0>(grad_surface_metric_theta_theta) /= square(get(sin_theta));
  get<1>(grad_surface_metric_theta_theta) /= square(get(sin_theta));
  get<0>(grad_surface_metric_theta_theta) -=
      2.0 * get<0, 0>(surface_metric) * get(cos_theta) / get(sin_theta);

  auto grad_surface_metric_theta_phi =
      ylm.gradient(get(sin_theta) * get<0, 1>(surface_metric));

  get<0>(grad_surface_metric_theta_phi) /= get(sin_theta);
  get<1>(grad_surface_metric_theta_phi) /= get(sin_theta);
  get<0>(grad_surface_metric_theta_phi) -=
      get<0, 1>(surface_metric) * get(cos_theta) / get(sin_theta);

  auto grad_surface_metric_phi_phi = ylm.gradient(get<1, 1>(surface_metric));

  auto deriv_surface_metric =
      make_with_value<tnsr::ijj<DataVector, 2, Frame::Spherical<Fr>>>(
          get<0, 0>(surface_metric), 0.0);
  // Get the partial derivative of the metric from the Pfaffian derivative
  get<0, 0, 0>(deriv_surface_metric) = get<0>(grad_surface_metric_theta_theta);
  get<1, 0, 0>(deriv_surface_metric) =
      get(sin_theta) * get<1>(grad_surface_metric_theta_theta);
  get<0, 0, 1>(deriv_surface_metric) = get<0>(grad_surface_metric_theta_phi);
  get<1, 0, 1>(deriv_surface_metric) =
      get(sin_theta) * get<1>(grad_surface_metric_theta_phi);
  get<0, 1, 1>(deriv_surface_metric) = get<0>(grad_surface_metric_phi_phi);
  get<1, 1, 1>(deriv_surface_metric) =
      get(sin_theta) * get<1>(grad_surface_metric_phi_phi);

  return trace_last_indices(
      raise_or_lower_first_index(
          gr::christoffel_first_kind(deriv_surface_metric),
          inverse_surface_metric),
      inverse_surface_metric);
}

// I'm going to solve a general eigenvalue problem of the form
// A x = lambda B x, where A and B are NxN, where N is the
// number of elements with l > 0 and l < ntheta - 2,
// i.e. l < l_max + 1 - 2 = l_max - 1. This function computes N.
size_t get_matrix_dimension(const YlmSpherepack& ylm) noexcept {
  // If l_max == m_max, there are square(l_max+1) Ylms total
  size_t matrix_dimension = square(ylm.l_max() + 1);
  // If l_max > m_max, there are
  // (l_max - m_max) * (l_max - m_max + 1) fewer Ylms total
  matrix_dimension -=
      (ylm.l_max() - ylm.m_max()) * (ylm.l_max() - ylm.m_max() + 1);
  // The actual matrix dimension is smaller, because we do not count
  // Ylms with l == 0, l == l_max, or l == l_max - 1.
  matrix_dimension -= 4 * ylm.m_max() + 3;
  if (ylm.l_max() == ylm.m_max()) {
    matrix_dimension += 2;
  }
  return matrix_dimension;
}

// Get left matrix A and right matrix B for eigenproblem A x = lambda B x.
template <typename Fr>
void get_left_and_right_eigenproblem_matrices(
    const gsl::not_null<Matrix*> left_matrix,
    const gsl::not_null<Matrix*> right_matrix,
    const tnsr::II<DataVector, 2, Frame::Spherical<Fr>>& inverse_surface_metric,
    const tnsr::I<DataVector, 2, Frame::Spherical<Fr>>&
        trace_christoffel_second_kind,
    const Scalar<DataVector>& sin_theta, const Scalar<DataVector>& ricci_scalar,
    const YlmSpherepack& ylm) noexcept {
  const auto grad_ricci_scalar = ylm.gradient(get(ricci_scalar));
  // loop over all terms with 0<l<l_max-1: each makes a column of
  // the matrices for the eigenvalue problem
  size_t column = 0;  // number which column of the matrix we are filling
  for (auto iter_i = SpherepackIterator(ylm.l_max(), ylm.m_max()); iter_i;
       ++iter_i) {
    if (iter_i.l() > 0 and iter_i.l() < ylm.l_max() - 1 and
        iter_i.m() <= iter_i.l()) {
      // Make a spectral vector that's all zeros except for one element,
      // which is 1. This corresponds to the ith Ylm, which I call yi.
      DataVector yi_spectral(ylm.spectral_size(), 0.0);
      yi_spectral[iter_i()] = 1.0;

      // Transform column vector corresponding to
      // a specific Y_lm to physical space.
      const DataVector yi_physical = ylm.spec_to_phys(yi_spectral);

      // In physical space, numerically compute the
      // linear differential operators acting on the
      // ith Y_lm.

      // \nabla^2 Y_lm
      const auto derivs_yi = ylm.first_and_second_derivative(yi_physical);
      auto laplacian_yi =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(laplacian_yi) +=
          get<0, 0>(derivs_yi.second) * get<0, 0>(inverse_surface_metric);
      get(laplacian_yi) += 2.0 * get<1, 0>(derivs_yi.second) *
                           get<1, 0>(inverse_surface_metric) * get(sin_theta);
      get(laplacian_yi) += get<1, 1>(derivs_yi.second) *
                           get<1, 1>(inverse_surface_metric) *
                           square(get(sin_theta));
      get(laplacian_yi) -=
          get<0>(derivs_yi.first) * get<0>(trace_christoffel_second_kind);
      get(laplacian_yi) -= get<1>(derivs_yi.first) * get(sin_theta) *
                           get<1>(trace_christoffel_second_kind);

      // \nabla^4 Y_lm
      const auto derivs_laplacian_yi =
          ylm.first_and_second_derivative(get(laplacian_yi));
      auto laplacian_squared_yi =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(laplacian_squared_yi) += get<0, 0>(derivs_laplacian_yi.second) *
                                   get<0, 0>(inverse_surface_metric);
      get(laplacian_squared_yi) += 2.0 * get<1, 0>(derivs_laplacian_yi.second) *
                                   get<1, 0>(inverse_surface_metric) *
                                   get(sin_theta);
      get(laplacian_squared_yi) += get<1, 1>(derivs_laplacian_yi.second) *
                                   get<1, 1>(inverse_surface_metric) *
                                   square(get(sin_theta));
      get(laplacian_squared_yi) -= get<0>(derivs_laplacian_yi.first) *
                                   get<0>(trace_christoffel_second_kind);
      get(laplacian_squared_yi) -= get<1>(derivs_laplacian_yi.first) *
                                   get(sin_theta) *
                                   get<1>(trace_christoffel_second_kind);

      // \nabla R \cdot \nabla Y_lm
      auto grad_ricci_scalar_dot_grad_yi =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(grad_ricci_scalar_dot_grad_yi) += get<0>(derivs_yi.first) *
                                            get<0>(grad_ricci_scalar) *
                                            get<0, 0>(inverse_surface_metric);
      get(grad_ricci_scalar_dot_grad_yi) +=
          get<0>(derivs_yi.first) * get<1>(grad_ricci_scalar) *
          get<1, 0>(inverse_surface_metric) * get(sin_theta);
      get(grad_ricci_scalar_dot_grad_yi) +=
          get<1>(derivs_yi.first) * get<0>(grad_ricci_scalar) *
          get<1, 0>(inverse_surface_metric) * get(sin_theta);
      get(grad_ricci_scalar_dot_grad_yi) +=
          get<1>(derivs_yi.first) * get<1>(grad_ricci_scalar) *
          get<1, 1>(inverse_surface_metric) * square(get(sin_theta));

      // Assemble the operator making up the eigenproblem's left-hand-side
      auto left_matrix_yi_physical =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(left_matrix_yi_physical) = get(laplacian_squared_yi) +
                                     get(ricci_scalar) * get(laplacian_yi) +
                                     get(grad_ricci_scalar_dot_grad_yi);

      // Transform back to spectral space, to get one column each for the left
      // and right matrices for the eigenvalue problem.
      const DataVector left_matrix_yi_spectral =
          ylm.phys_to_spec(get(left_matrix_yi_physical));
      const DataVector right_matrix_yi_spectral =
          ylm.phys_to_spec(get(laplacian_yi));

      // Set the current column of the left and right matrices
      // for the eigenproblem.
      size_t row = 0;
      for (auto iter_j = SpherepackIterator(ylm.l_max(), ylm.m_max()); iter_j;
           ++iter_j) {
        if (iter_j.l() > 0 and iter_j.l() < ylm.l_max() - 1) {
          (*left_matrix)(row, column) = left_matrix_yi_spectral[iter_j()];
          (*right_matrix)(row, column) = right_matrix_yi_spectral[iter_j()];
          ++row;
        }
      }  // loop over rows
      ++column;
    }
  }  // loop over columns
}

// Find the eigenvectors corresponding to the three smallest-magnitude
// eigenvalues.
// Note: uses the fact that eigenvalues should be real
std::array<DataVector, 3> get_eigenvectors_for_3_smallest_magnitude_eigenvalues(
    const DataVector& eigenvalues_real_part, const Matrix& eigenvectors,
    const YlmSpherepack& ylm) noexcept {
  size_t index_smallest = 0;
  size_t index_second_smallest = 0;
  size_t index_third_smallest = 0;

  // Simple algorithm that loops over all elements to
  // find indexes of 3 smallest-magnitude eigenvalues
  for (size_t i = 1; i < eigenvalues_real_part.size(); ++i) {
    if (abs(eigenvalues_real_part[i]) <
        abs(eigenvalues_real_part[index_smallest])) {
      index_third_smallest = index_second_smallest;
      index_second_smallest = index_smallest;
      index_smallest = i;
    } else if (i < 2 or abs(eigenvalues_real_part[i]) <
                            abs(eigenvalues_real_part[index_second_smallest])) {
      index_third_smallest = index_second_smallest;
      index_second_smallest = i;
    } else if (i < 3 or abs(eigenvalues_real_part[i]) <
                            abs(eigenvalues_real_part[index_third_smallest])) {
      index_third_smallest = i;
    }
  }

  DataVector smallest_eigenvector(ylm.spectral_size(), 0.0);
  DataVector second_smallest_eigenvector(ylm.spectral_size(), 0.0);
  DataVector third_smallest_eigenvector(ylm.spectral_size(), 0.0);

  size_t row = 0;

  for (auto iter_i = SpherepackIterator(ylm.l_max(), ylm.m_max()); iter_i;
       ++iter_i) {
    if (iter_i.l() > 0 and iter_i.l() < ylm.l_max() - 1) {
      smallest_eigenvector[iter_i()] = eigenvectors(row, index_smallest);
      second_smallest_eigenvector[iter_i()] =
          eigenvectors(row, index_second_smallest);
      third_smallest_eigenvector[iter_i()] =
          eigenvectors(row, index_third_smallest);
      ++row;
    }
  }

  return {{smallest_eigenvector, second_smallest_eigenvector,
           third_smallest_eigenvector}};
}

// This function converts the three eigenvectors with smallest-magnitude
// eigenvalues to physical space to get the spin potentials corresponding to
// the approximate Killing vectors. The potentials are normalized using the
// "Kerr normalization:" the integral of (potential - the potential average)^2
// is set to (horizon area)^3/(48*pi), as it is for Kerr.
std::array<DataVector, 3> get_normalized_spin_potentials(
    const std::array<DataVector, 3>& eigenvectors_for_potentials,
    const YlmSpherepack& ylm, const Scalar<DataVector>& area_element) noexcept {
  const double area = ylm.definite_integral(get(area_element).data());

  std::array<DataVector, 3> potentials;

  DataVector temp_integrand(get(area_element));
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(potentials, i) =
        ylm.spec_to_phys(gsl::at(eigenvectors_for_potentials, i));

    temp_integrand = gsl::at(potentials, i) * get(area_element);
    const double potential_average =
        ylm.definite_integral(temp_integrand.data()) / area;

    temp_integrand =
        square(gsl::at(potentials, i) - potential_average) * get(area_element);
    const double potential_norm = ylm.definite_integral(temp_integrand.data());
    gsl::at(potentials, i) *=
        sqrt(cube(area) / (48.0 * square(M_PI) * potential_norm));
  }
  return potentials;
}

// Get the spin magnitude. There are three potentials, each corresponding to an
// approximate Killing vector. The spin for each potential is the surface
// integral of the potential times the spin function. The spin magnitude
// is the Euclidean norm of the spin for each potential.
double get_spin_magnitude(const std::array<DataVector, 3>& potentials,
                          const Scalar<DataVector>& spin_function,
                          const Scalar<DataVector>& area_element,
                          const YlmSpherepack& ylm) noexcept {
  double spin_magnitude_squared = 0.0;

  DataVector spin_density(get(area_element));
  for (size_t i = 0; i < 3; ++i) {
    spin_density =
        gsl::at(potentials, i) * get(spin_function) * get(area_element);
    spin_magnitude_squared +=
        square(ylm.definite_integral(spin_density.data()) / (8.0 * M_PI));
  }
  return sqrt(spin_magnitude_squared);
}
}  // namespace

namespace StrahlkorperGr {

template <typename Frame>
tnsr::i<DataVector, 3, Frame> unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) noexcept {
  auto unit_normal_one_form = normal_one_form;
  for (size_t i = 0; i < 3; ++i) {
    unit_normal_one_form.get(i) *= one_over_one_form_magnitude;
  }
  return unit_normal_one_form;
}

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> grad_unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& r_hat, const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) noexcept {
  const DataVector one_over_radius = 1.0 / radius;
  tnsr::ii<DataVector, 3, Frame> grad_normal(radius.size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // symmetry
      grad_normal.get(i, j) = -one_over_one_form_magnitude *
                              (r_hat.get(i) * r_hat.get(j) * one_over_radius +
                               d2x_radius.get(i, j));
      for (size_t k = 0; k < 3; ++k) {
        grad_normal.get(i, j) -=
            unit_normal_one_form.get(k) * christoffel_2nd_kind.get(k, i, j);
      }
    }
    grad_normal.get(i, i) += one_over_radius * one_over_one_form_magnitude;
  }
  return grad_normal;
}

template <typename Frame>
tnsr::II<DataVector, 3, Frame> inverse_surface_metric(
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept {
  auto inv_surf_metric = upper_spatial_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      inv_surf_metric.get(i, j) -=
          unit_normal_vector.get(i) * unit_normal_vector.get(j);
    }
  }
  return inv_surf_metric;
}

template <typename Frame>
Scalar<DataVector> expansion(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) noexcept {
  // If you want the future *ingoing* null expansion,
  // the formula is the same as here except you
  // change the sign on grad_normal just before you
  // subtract the extrinsic curvature.
  // That is, if GsBar is the value of grad_normal
  // at this point in the code, and S^i is the unit
  // spatial normal to the surface,
  // the outgoing expansion is
  // (g^ij - S^i S^j) (GsBar_ij - K_ij)
  // and the ingoing expansion is
  // (g^ij - S^i S^j) (-GsBar_ij - K_ij)

  Scalar<DataVector> expansion(get<0, 0>(grad_normal).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(expansion) += inverse_surface_metric.get(i, j) *
                        (grad_normal.get(i, j) - extrinsic_curvature.get(i, j));
    }
  }

  return expansion;
}

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector) noexcept {
  Scalar<DataVector> nI_nJ_gradnij(get<0, 0>(grad_normal).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(nI_nJ_gradnij) += unit_normal_vector.get(i) *
                            unit_normal_vector.get(j) * grad_normal.get(i, j);
    }
  }

  auto extrinsic_curvature(grad_normal);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      extrinsic_curvature.get(i, j) += unit_normal_one_form.get(i) *
                                       unit_normal_one_form.get(j) *
                                       get(nI_nJ_gradnij);
      for (size_t k = 0; k < 3; ++k) {
        extrinsic_curvature.get(i, j) -=
            unit_normal_vector.get(k) *
            (unit_normal_one_form.get(i) * grad_normal.get(j, k) +
             unit_normal_one_form.get(j) * grad_normal.get(i, k));
      }
    }
  }
  return extrinsic_curvature;
}

template <typename Frame>
Scalar<DataVector> ricci_scalar(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept {
  auto ricci_scalar = trace(spatial_ricci_tensor, upper_spatial_metric);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(ricci_scalar) -= 2.0 * spatial_ricci_tensor.get(i, j) *
                           unit_normal_vector.get(i) *
                           unit_normal_vector.get(j);

      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          // K^{ij} K_{ij} = g^{ik} g^{jl} K_{kl} K_{ij}
          get(ricci_scalar) -=
              upper_spatial_metric.get(i, k) * upper_spatial_metric.get(j, l) *
              extrinsic_curvature.get(k, l) * extrinsic_curvature.get(i, j);
        }
      }
    }
  }

  get(ricci_scalar) +=
      square(get(trace(extrinsic_curvature, upper_spatial_metric)));

  return ricci_scalar;
}

template <typename Frame>
Scalar<DataVector> area_element(
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept {
  auto cap_theta = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);
  auto cap_phi = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    cap_theta.get(i) = jacobian.get(i, 0);
    cap_phi.get(i) = jacobian.get(i, 1);
    for (size_t j = 0; j < 3; ++j) {
      cap_theta.get(i) += r_hat.get(i) *
                          (r_hat.get(j) - normal_one_form.get(j)) *
                          jacobian.get(j, 0);
      cap_phi.get(i) += r_hat.get(i) * (r_hat.get(j) - normal_one_form.get(j)) *
                        jacobian.get(j, 1);
    }
  }

  auto area_element = Scalar<DataVector>{square(radius)};
  get(area_element) *=
      sqrt(get(dot_product(cap_theta, cap_theta, spatial_metric)) *
               get(dot_product(cap_phi, cap_phi, spatial_metric)) -
           square(get(dot_product(cap_theta, cap_phi, spatial_metric))));
  return area_element;
}

template <typename Frame>
Scalar<DataVector> euclidean_area_element(
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept {
  auto cap_theta = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);
  auto cap_phi = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    cap_theta.get(i) = jacobian.get(i, 0);
    cap_phi.get(i) = jacobian.get(i, 1);
    for (size_t j = 0; j < 3; ++j) {
      cap_theta.get(i) += r_hat.get(i) *
                          (r_hat.get(j) - normal_one_form.get(j)) *
                          jacobian.get(j, 0);
      cap_phi.get(i) += r_hat.get(i) * (r_hat.get(j) - normal_one_form.get(j)) *
                        jacobian.get(j, 1);
    }
  }

  auto area_element = Scalar<DataVector>{square(radius)};
  get(area_element) *= sqrt(get(dot_product(cap_theta, cap_theta)) *
                                get(dot_product(cap_phi, cap_phi)) -
                            square(get(dot_product(cap_theta, cap_phi))));
  return area_element;
}

template <typename Frame>
double surface_integral_of_scalar(
    const Scalar<DataVector>& area_element, const Scalar<DataVector>& scalar,
    const Strahlkorper<Frame>& strahlkorper) noexcept {
  const DataVector integrand = get(area_element) * get(scalar);
  return strahlkorper.ylm_spherepack().definite_integral(integrand.data());
}

template <typename Frame>
Scalar<DataVector> spin_function(
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const YlmSpherepack& ylm,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const Scalar<DataVector>& area_element,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) noexcept {
  auto temp = make_with_value<Scalar<DataVector>>(area_element, 0.0);
  auto extrinsic_curvature_theta_normal_sin_theta =
      make_with_value<Scalar<DataVector>>(area_element, 0.0);
  auto extrinsic_curvature_phi_normal =
      make_with_value<Scalar<DataVector>>(area_element, 0.0);

  DataVector& extrinsic_curvature_dot_normal = get(temp);
  for (size_t i = 0; i < 3; ++i) {
    extrinsic_curvature_dot_normal =
        extrinsic_curvature.get(i, 0) * get<0>(unit_normal_vector);
    for (size_t j = 1; j < 3; ++j) {
      extrinsic_curvature_dot_normal +=
          extrinsic_curvature.get(i, j) * unit_normal_vector.get(j);
    }

    // Note: I must multiply by sin_theta because
    // I take the phi derivative of this term by using
    // the spherepack gradient, which includes a
    // sin_theta in the denominator of the phi derivative.
    // Will do this outside the i,j loops.
    get(extrinsic_curvature_theta_normal_sin_theta) +=
        extrinsic_curvature_dot_normal * tangents.get(i, 0);

    // Note: I must multiply by sin_theta because tangents.get(i,1)
    // actually contains \partial_\phi / sin(theta), but I want just
    //\partial_\phi. Will do this outside the i,j loops.
    get(extrinsic_curvature_phi_normal) +=
        extrinsic_curvature_dot_normal * tangents.get(i, 1);
  }

  DataVector& sin_theta = get(temp);
  sin_theta = sin(ylm.theta_phi_points()[0]);
  get(extrinsic_curvature_theta_normal_sin_theta) *= sin_theta;
  get(extrinsic_curvature_phi_normal) *= sin_theta;

  Scalar<DataVector>& spin_function = temp;
  get(spin_function) =
      (get<0>(ylm.gradient(get(extrinsic_curvature_phi_normal))) -
       get<1>(ylm.gradient(get(extrinsic_curvature_theta_normal_sin_theta)))) /
      (sin_theta * get(area_element));

  return temp;
}

template <typename Frame>
double dimensionful_spin_magnitude(
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const YlmSpherepack& ylm, const Scalar<DataVector>& area_element) noexcept {
  const Scalar<DataVector> sin_theta{sin(ylm.theta_phi_points()[0])};

  const auto& surface_metric =
      get_surface_metric(spatial_metric, tangents, sin_theta);
  const auto& inverse_surface_metric =
      determinant_and_inverse(surface_metric).second;
  const auto& trace_christoffel_second_kind = get_trace_christoffel_second_kind(
      surface_metric, inverse_surface_metric, sin_theta, ylm);

  const size_t matrix_dimension = get_matrix_dimension(ylm);
  Matrix left_matrix(matrix_dimension, matrix_dimension, 0.0);
  Matrix right_matrix(matrix_dimension, matrix_dimension, 0.0);
  get_left_and_right_eigenproblem_matrices(
      &left_matrix, &right_matrix, inverse_surface_metric,
      trace_christoffel_second_kind, sin_theta, ricci_scalar, ylm);

  DataVector eigenvalues_real_part(matrix_dimension, 0.0);
  DataVector eigenvalues_im_part(matrix_dimension, 0.0);
  Matrix eigenvectors(matrix_dimension, matrix_dimension, 0.0);
  find_generalized_eigenvalues(&eigenvalues_real_part, &eigenvalues_im_part,
                               &eigenvectors, left_matrix, right_matrix);

  const std::array<DataVector, 3> smallest_eigenvectors =
      get_eigenvectors_for_3_smallest_magnitude_eigenvalues(
          eigenvalues_real_part, eigenvectors, ylm);

  // Get normalized potentials (Kerr normalization) corresponding to the
  // eigenvectors with three smallest-magnitude eigenvalues.
  const auto& potentials =
      get_normalized_spin_potentials(smallest_eigenvectors, ylm, area_element);

  return get_spin_magnitude(potentials, spin_function, area_element, ylm);
}

template <typename Frame>
std::array<double, 3> spin_vector(const double spin_magnitude,
                                  const Scalar<DataVector>& area_element,
                                  const Scalar<DataVector>& radius,
                                  const tnsr::i<DataVector, 3, Frame>& r_hat,
                                  const Scalar<DataVector>& ricci_scalar,
                                  const Scalar<DataVector>& spin_function,
                                  const YlmSpherepack& ylm) noexcept {
  std::array<double, 3> spin_vector = {{0.0, 0.0, 0.0}};
  auto integrand = make_with_value<Scalar<DataVector>>(get(radius), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    // Compute horizon coordinates with a coordinate center such that
    // the mass dipole moment vanishes.
    get(integrand) =
        get(area_element) * get(ricci_scalar) * r_hat.get(i) * get(radius);
    get(integrand) =
        ylm.definite_integral(get(integrand).data()) / (-8.0 * M_PI);
    get(integrand) += r_hat.get(i) * get(radius);

    // Get a component of a vector in the direction of the spin
    get(integrand) *= get(area_element) * get(spin_function);
    gsl::at(spin_vector, i) = ylm.definite_integral(get(integrand).data());
  }

  // Normalize spin_vector so its magnitude is the magnitude of the spin
  return spin_vector * (spin_magnitude / magnitude(spin_vector));
}

double irreducible_mass(const double area) noexcept {
  ASSERT(area > 0.0,
         "The area of the horizon must be greater than zero but is " << area);
  return sqrt(area / (16.0 * M_PI));
}

double christodoulou_mass(const double dimensionful_spin_magnitude,
                          const double irreducible_mass) noexcept {
  return sqrt(square(irreducible_mass) + (square(dimensionful_spin_magnitude) /
                                          (4.0 * square(irreducible_mass))));
}
}  // namespace StrahlkorperGr

template tnsr::i<DataVector, 3, Frame::Inertial>
StrahlkorperGr::unit_normal_one_form<Frame::Inertial>(
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) noexcept;

template tnsr::ii<DataVector, 3, Frame::Inertial>
StrahlkorperGr::grad_unit_normal_one_form<Frame::Inertial>(
    const tnsr::i<DataVector, 3, Frame::Inertial>& r_hat,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame::Inertial>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame::Inertial>&
        christoffel_2nd_kind) noexcept;

template tnsr::II<DataVector, 3, Frame::Inertial>
StrahlkorperGr::inverse_surface_metric<Frame::Inertial>(
    const tnsr::I<DataVector, 3, Frame::Inertial>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        upper_spatial_metric) noexcept;

template Scalar<DataVector> StrahlkorperGr::expansion<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& grad_normal,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inverse_surface_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature) noexcept;

template tnsr::ii<DataVector, 3, Frame::Inertial>
StrahlkorperGr::extrinsic_curvature<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& grad_normal,
    const tnsr::i<DataVector, 3, Frame::Inertial>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame::Inertial>& unit_normal_vector) noexcept;

template Scalar<DataVector> StrahlkorperGr::ricci_scalar<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        upper_spatial_metric) noexcept;

template Scalar<DataVector> StrahlkorperGr::area_element<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame::Inertial>& jacobian,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame::Inertial>& r_hat) noexcept;

template Scalar<DataVector>
StrahlkorperGr::euclidean_area_element<Frame::Inertial>(
    const StrahlkorperTags::aliases::Jacobian<Frame::Inertial>& jacobian,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame::Inertial>& r_hat) noexcept;

template double StrahlkorperGr::surface_integral_of_scalar(
    const Scalar<DataVector>& area_element, const Scalar<DataVector>& scalar,
    const Strahlkorper<Frame::Inertial>& strahlkorper) noexcept;

template Scalar<DataVector> StrahlkorperGr::spin_function<Frame::Inertial>(
    const StrahlkorperTags::aliases::Jacobian<Frame::Inertial>& tangents,
    const YlmSpherepack& ylm,
    const tnsr::I<DataVector, 3, Frame::Inertial>& unit_normal_vector,
    const Scalar<DataVector>& area_element,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature) noexcept;

template double StrahlkorperGr::dimensionful_spin_magnitude<Frame::Inertial>(
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame::Inertial>& tangents,
    const YlmSpherepack& ylm, const Scalar<DataVector>& area_element) noexcept;

template std::array<double, 3> StrahlkorperGr::spin_vector<Frame::Inertial>(
    const double spin_magnitude, const Scalar<DataVector>& area_element,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame::Inertial>& r_hat,
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function, const YlmSpherepack& ylm) noexcept;
/// \endcond
