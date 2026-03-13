// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Jacobi.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationWeights.hpp"
#include "NumericalAlgorithms/Spectral/InverseWeightFunctionValues.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"

namespace Spectral {
namespace {
template <size_t Dim, typename T>
T Q(const size_t n, const size_t m, const T& r) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(n >= m, "m, " << m << ", must be at most n, " << n);
  ASSERT((n + m) % 2 == 0,
         "n, " << n << ", plus m, " << m << ", must be even.");
  const auto mm = static_cast<double>(m);

  // Implemented from Matsushima1995
  // Matsushima and Marcus paper notation varries from usual standards
  // Note: their \alpha and \beta are NOT Jacobi polynomial \alpha and \beta
  // For 1D, their \alpha = 1, \beta = 0
  // For 2D, their \alpha = 1, \beta = 1
  // For 3D, their \alpha = 1, \beta = 2
  // This means their \gamma := 2\alpha + \beta is either 3 for polar or
  // 4 for spherical, which in SpEC's mBeta = \gamma - 2
  constexpr double betaM = Dim == 1 ? 0.0 : (Dim == 2 ? 1.0 : 2.0);
  // Qnm is Q_n^m, p2/m2 being plus2 / minus2
  // using betaM to indicate Marcus Beta
  T Qmm = integer_pow(r, m);
  if (n == m) {
    return Qmm;
  }
  T Qmp2m =
      0.5 * Qmm * ((2. * mm + betaM + 3.) * square(r) - (2. * mm + betaM + 1.));
  if (n == (m + 2)) {
    return Qmp2m;
  }
  T Qnm = Qmp2m;
  T Qnm2m = Qmm;
  for (size_t n_loop = m + 2; n_loop < n; n_loop += 2) {
    const auto nn = static_cast<double>(n_loop);
    const T Qnp2m =
        ((2. * nn + betaM + 1.) *
             ((2. * nn + betaM - 1.) * (2. * nn + betaM + 3.) * square(r) -
              2. * nn * (nn + betaM + 1.) - 2. * mm * (mm + betaM - 1.) -
              (betaM - 1.) * (betaM + 1.)) *
             Qnm -
         (nn - mm) * (nn + mm + betaM - 1.) * (2. * nn + betaM + 3.) * Qnm2m) /
        ((nn - mm + 2.) * (nn + mm + betaM + 1.) * (2. * nn + betaM - 1.));
    Qnm2m = Qnm;
    Qnm = Qnp2m;
  }
  return Qnm;
}

// Pi(r) := Q^0_m(r)
template <size_t Dim, typename T>
T Pi(const size_t m, const T& r) {
  ASSERT(m > 1, "Pi: m must be at least two, got m = " << m);
  return Q<Dim>(m, 0, r) - Q<Dim>(m - 2, 0, r);
}

template <size_t Dim>
double I(const size_t n, const size_t m) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(n >= m, "n = " << n << "; m = " << m);
  ASSERT((n + m) % 2 == 0,
         "n and m must have same parity, got n = " << n << " m = " << m);
  constexpr double betaM = Dim == 1 ? 0.0 : (Dim == 2 ? 1.0 : 2.0);
  double result = 0.5 / (static_cast<double>(m) + 0.5 + 0.5 * betaM);
  for (size_t i = m + 2; i <= n; i += 2) {
    const auto ii = static_cast<double>(i);
    result *= (2. * ii + betaM - 3.) / (2. * ii + betaM + 1.);
  }
  return result;
}

template <size_t Dim>
std::pair<DataVector, DataVector>
compute_collocation_points_and_weights_gauss_radau_impl(
    const size_t num_points) {
  // Implementing for true logical coordinates: [0,1]
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(
      num_points >= 1,
      "Zernike-Gauss-Radau quadrature requires at least one collocation point");
  const size_t poly_degree = num_points - 1;
  DataVector x(num_points);
  DataVector w(num_points);
  switch (poly_degree) {
    case 0:
      x[0] = 1.0;
      w[0] = I<Dim>(0, 0);
      break;
    default:
      size_t root_index = 0;
      // Find the collocation points by finding the roots of pi(m, r)
      // We know there is a root at r = 1 (and don't want one at 0)
      const size_t M = 2 * num_points;
      const size_t num_intervals = M * M;
      const double delta = 0.1 / static_cast<double>(num_intervals);
      const double dr = (1. - 2. * delta) / static_cast<double>(num_intervals);

      double r_left = delta;
      double f_left = Pi<Dim>(M, r_left);
      for (size_t i = 0; i < num_intervals; ++i) {
        const double r_right = r_left + dr;
        const double f_right = Pi<Dim>(M, r_right);

        // Checking if root has been bracketed
        if (f_left * f_right <= 0) {
          const double root =
              RootFinder::toms748([M](const double r) { return Pi<Dim>(M, r); },
                                  r_left, r_right, f_left, f_right, 0.0,
                                  4.0 * std::numeric_limits<double>::epsilon());
          x[root_index] = root;
          ++root_index;
        }
        r_left = r_right;
        f_left = f_right;
      }
      ASSERT(num_points - 1 == root_index,
             "Did not find all the roots of Pi, expect "
                 << num_points - 1 << ", but got " << root_index);
      x[num_points - 1] = 1.0;
      // Computing quadrature weights
      const auto mm = static_cast<double>(M);
      constexpr double betaM = Dim == 1 ? 0.0 : (Dim == 2 ? 1.0 : 2.0);
      for (size_t i = 0; i < num_points - 1; ++i) {
        w[i] = 2. * (2. * mm + betaM - 3.) * square(x[i]) * I<Dim>(M - 2, 0) /
               (mm * (mm + betaM - 1.) * square(Q<Dim>(M - 2, 0, x[i])));
      }
      w[num_points - 1] =
          (2. * (2. * mm + betaM - 3.) * I<Dim>(M - 2, 0) /
           (mm * (mm + betaM - 1.) * square(Q<Dim>(M - 2, 0, 1.0))));
      break;
  }
  return std::make_pair(std::move(x), std::move(w));
}
}  // namespace

template <size_t Dim>
template <typename T>
T Zernike<Dim>::basis_function_value(const size_t n, const size_t m,
                                     const T& xi) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  // the sqrt(I) normalization is for normalizing "with respect to the weight
  // w(r)," which is the integrating weight in the orthogonality condition
  // Note that xi is logical coordinate, so in [-1, 1]. Need to shift back to
  // true [0, 1] range
  return Q<Dim>(n, m, static_cast<T>(0.5 * (xi + 1.0))) / sqrt(I<Dim>(n, m));
}

template <size_t Dim>
std::pair<DataVector, DataVector>
Zernike<Dim>::compute_collocation_points_and_weights(const size_t num_points) {
  // Manually adding an affine coordinate map to shift logical coordinates
  // from [0,1] -> [-1,1] (simpler to keep track of this than generalize the
  // [-1,1] assumption scattered through the code)
  auto [points, weights] =
      compute_collocation_points_and_weights_gauss_radau_impl<Dim>(num_points);
  points = 2.0 * points - 1.0;
  // Don't need to modify weights as the affine shift is only external, you
  // will only ever use weights on "real" [0,1] coordinates
  return std::make_pair(points, weights);
}

template <size_t Dim>
Matrix Zernike<Dim>::differentiation_matrix(const size_t num_points,
                                            const Parity parity) {
  ASSERT(parity != Parity::Uninitialized, "Passed parity must set");
  // "missing" factor of 1/2 because logical coordinates have a Jacobian
  // factor to go from [-1,1] -> [0,1]
  const DataVector collocation_pts =
      Zernike<Dim>::compute_collocation_points_and_weights(num_points).first +
      1.0;
  const size_t max_deriv = 1;
  std::array<DataVector, max_deriv + 1> fornberg_weights{};
  DataVector extended_collocation_pts(2 * num_points, 0.0);
  Matrix extended_diff_matrix(num_points, 2 * num_points);
  Matrix projector(2 * num_points, num_points, 0.0);
  for (size_t i = 0; i < num_points; ++i) {
    extended_collocation_pts[i] = collocation_pts[i];
    extended_collocation_pts[i + num_points] = -collocation_pts[i];
    projector(i, i) = 1.0;
    projector(i + num_points, i) = parity == Parity::Odd ? -1.0 : 1.0;
  }
  for (size_t i = 0; i < num_points; ++i) {
    Spectral::fornberg_derivative_interpolation_weights<max_deriv>(
        make_not_null(&fornberg_weights), collocation_pts[i],
        extended_collocation_pts);
    for (size_t j = 0; j < 2 * num_points; ++j) {
      extended_diff_matrix(i, j) = fornberg_weights[1][j];
    }
  }
  return extended_diff_matrix * projector;
}

// Specializations of function templates defined in the Spectral directory

template <>
DataVector compute_basis_function_value<Basis::ZernikeB1>(
    const size_t k, const size_t m, const DataVector& xi) {
  return Zernike<1>::basis_function_value(k, m, xi);
}

template <>
DataVector compute_basis_function_value<Basis::ZernikeB2>(
    const size_t k, const size_t m, const DataVector& xi) {
  return Zernike<2>::basis_function_value(k, m, xi);
}

template <>
DataVector compute_basis_function_value<Basis::ZernikeB3>(
    const size_t k, const size_t m, const DataVector& xi) {
  return Zernike<3>::basis_function_value(k, m, xi);
}

// The bases orthonormal wrt n
template <>
double compute_basis_function_normalization_square<Basis::ZernikeB1>(
    const size_t /*k*/) {
  return 1.;
}

template <>
double compute_basis_function_normalization_square<Basis::ZernikeB2>(
    const size_t /*k*/) {
  return 1.;
}

template <>
double compute_basis_function_normalization_square<Basis::ZernikeB3>(
    const size_t /*k*/) {
  return 1.;
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::ZernikeB1, Quadrature::GaussRadauUpper>(const size_t num_points) {
  return Zernike<1>::compute_collocation_points_and_weights(num_points);
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::ZernikeB2, Quadrature::GaussRadauUpper>(const size_t num_points) {
  return Zernike<2>::compute_collocation_points_and_weights(num_points);
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::ZernikeB3, Quadrature::GaussRadauUpper>(const size_t num_points) {
  return Zernike<3>::compute_collocation_points_and_weights(num_points);
}


#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
// The following one-index basis function values need to be created so that
// GetSpectralQuantityForMesh.hpp can correctly instantiate all things called
// from SPECTRAL_QUANTITY_FOR_MESH that are genuinetly one-indexed
// (e.g. collocation points and weights)
template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

template <>
Matrix spectral_indefinite_integral_matrix<Basis::ZernikeB1>(
    const size_t /*num_points*/) {
  ERROR("Indefinite integral matrix is not defined for ZernikeB1 basis");
}
template <>
Matrix spectral_indefinite_integral_matrix<Basis::ZernikeB2>(
    const size_t /*num_points*/) {
  ERROR("Indefinite integral matrix is not defined for ZernikeB2 basis");
}
template <>
Matrix spectral_indefinite_integral_matrix<Basis::ZernikeB3>(
    const size_t /*num_points*/) {
  ERROR("Indefinite integral matrix is not defined for ZernikeB3 basis");
}

template <>
DataVector compute_basis_function_value<Basis::ZernikeB1>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("Calling one-index Zernike basis function");
}

template <>
DataVector compute_basis_function_value<Basis::ZernikeB2>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("Calling one-index Zernike basis function");
}

template <>
DataVector compute_basis_function_value<Basis::ZernikeB3>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("Calling one-index Zernike basis function");
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_TYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_BASIS_FUNCTION_VALUE(r, data)                       \
  template GET_TYPE(data) Zernike<GET_DIM(data)>::basis_function_value( \
      const size_t n, const size_t m, const GET_TYPE(data) & xi);

GENERATE_INSTANTIATIONS(INSTANTIATE_BASIS_FUNCTION_VALUE, (1, 2, 3),
                        (double, DataVector))

#undef INSTANTIATE_BASIS_FUNCTION_VALUE
#undef GET_TYPE

#define INSTANTIATE_DIFF_MATRICES(r, data)                        \
  template Matrix Zernike<GET_DIM(data)>::differentiation_matrix( \
      const size_t num_points, const Parity parity);

GENERATE_INSTANTIATIONS(INSTANTIATE_DIFF_MATRICES, (1, 2, 3))

#undef INSTANTIATE_DIFF_MATRICES
#undef GET_DIM
}  // namespace Spectral
