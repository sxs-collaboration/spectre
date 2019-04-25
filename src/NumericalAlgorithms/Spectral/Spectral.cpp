// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <algorithm>
#include <ostream>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {

std::ostream& operator<<(std::ostream& os,
                         const Basis& basis) noexcept {
  switch (basis) {
    case Basis::Legendre:
      return os << "Legendre";
    case Basis::Chebyshev:
      return os << "Chebyshev";
    default: ERROR("Invalid basis");
  }
}

std::ostream& operator<<(std::ostream& os,
                         const Quadrature& quadrature) noexcept {
  switch (quadrature) {
    case Quadrature::Gauss:
      return os << "Gauss";
    case Quadrature::GaussLobatto:
      return os << "GaussLobatto";
    default: ERROR("Invalid quadrature");
  }
}

template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points) noexcept;

namespace {

// Caching mechanism

template <Basis BasisType, Quadrature QuadratureType,
          typename SpectralQuantityGenerator>
const auto& precomputed_spectral_quantity(const size_t num_points) noexcept {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  // We compute the quantity for all possible `num_point`s the first time this
  // function is called and keep the data around for the lifetime of the
  // program. The computation is handled by the call operator of the
  // `SpectralQuantityType` instance.
  static const auto precomputed_data =
      make_static_cache<CacheRange<min_num_points, max_num_points + 1>>(
          SpectralQuantityGenerator{});
  return precomputed_data(num_points);
}

template <Basis BasisType, Quadrature QuadratureType>
struct CollocationPointsAndWeightsGenerator {
  std::pair<DataVector, DataVector> operator()(const size_t num_points) const
      noexcept {
    return compute_collocation_points_and_weights<BasisType, QuadratureType>(
        num_points);
  }
};

// Computation of basis-agnostic quantities

template <Basis BasisType, Quadrature QuadratureType>
struct QuadratureWeightsGenerator {
  DataVector operator()(const size_t num_points) const noexcept {
    const auto& pts_and_weights = precomputed_spectral_quantity<
        BasisType, QuadratureType,
        CollocationPointsAndWeightsGenerator<BasisType, QuadratureType>>(
        num_points);
    return pts_and_weights.second *
           compute_inverse_weight_function_values<BasisType>(
               pts_and_weights.first);
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct BarycentricWeightsGenerator {
  DataVector operator()(const size_t num_points) const noexcept {
    // Algorithm 30 in Kopriva, p. 75
    // This is valid for any collocation points.
    const DataVector& x =
        collocation_points<BasisType, QuadratureType>(num_points);
    DataVector bary_weights(num_points, 1.);
    for (size_t j = 1; j < num_points; j++) {
      for (size_t k = 0; k < j; k++) {
        bary_weights[k] *= x[k] - x[j];
        bary_weights[j] *= x[j] - x[k];
      }
    }
    for (size_t j = 0; j < num_points; j++) {
      bary_weights[j] = 1. / bary_weights[j];
    }
    return bary_weights;
  }
};

// We don't need this as part of the public interface, but precompute it since
// `interpolation_matrix` needs it at runtime.
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& barycentric_weights(const size_t num_points) noexcept {
  return precomputed_spectral_quantity<
      BasisType, QuadratureType,
      BarycentricWeightsGenerator<BasisType, QuadratureType>>(num_points);
}

template <Basis BasisType, Quadrature QuadratureType>
struct DifferentiationMatrixGenerator {
  Matrix operator()(const size_t num_points) const noexcept {
    // Algorithm 37 in Kopriva, p. 82
    // It is valid for any collocation points and barycentric weights.
    const DataVector& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    const DataVector& bary_weights =
        barycentric_weights<BasisType, QuadratureType>(num_points);
    Matrix diff_matrix(num_points, num_points);
    for (size_t i = 0; i < num_points; ++i) {
      double& diagonal = diff_matrix(i, i) = 0.0;
      for (size_t j = 0; j < num_points; ++j) {
        if (LIKELY(i != j)) {
          diff_matrix(i, j) =
              bary_weights[j] /
              (bary_weights[i] * (collocation_pts[i] - collocation_pts[j]));
          diagonal -= diff_matrix(i, j);
        }
      }
    }
    return diff_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct IntegrationMatrixGenerator {
  Matrix operator()(const size_t num_points) const noexcept {
    return Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(
               num_points) *
           spectral_indefinite_integral_matrix<BasisType>(num_points) *
           Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(
               num_points);
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct ModalToNodalMatrixGenerator {
  Matrix operator()(const size_t num_points) const noexcept {
    // To obtain the Vandermonde matrix we need to compute the basis function
    // values at the collocation points. Constructing the matrix proceeds
    // the same for any basis.
    const DataVector& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    Matrix vandermonde_matrix(num_points, num_points);
    for (size_t j = 0; j < num_points; j++) {
      const auto& basis_function_values =
          compute_basis_function_value<BasisType>(j, collocation_pts);
      for (size_t i = 0; i < num_points; i++) {
        vandermonde_matrix(i, j) = basis_function_values[i];
      }
    }
    return vandermonde_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct NodalToModalMatrixGenerator {
  Matrix operator()(const size_t num_points) const noexcept {
    // Numerically invert the matrix for this generic case
    return inv(modal_to_nodal_matrix<BasisType, QuadratureType>(num_points));
  }
};

template <Basis BasisType>
struct NodalToModalMatrixGenerator<BasisType, Quadrature::Gauss> {
  using data_type = Matrix;
  Matrix operator()(const size_t num_points) const noexcept {
    // For Gauss quadrature we implement the analytic expression
    // \f$\mathcal{V}^{-1}_{ij}=\mathcal{V}_{ji}\frac{w_j}{\gamma_i}\f$
    // (see description of `nodal_to_modal_matrix`).
    const DataVector& weights =
        precomputed_spectral_quantity<
            BasisType, Quadrature::Gauss,
            CollocationPointsAndWeightsGenerator<BasisType, Quadrature::Gauss>>(
            num_points)
            .second;
    const Matrix& vandermonde_matrix =
        modal_to_nodal_matrix<BasisType, Quadrature::Gauss>(num_points);
    Matrix vandermonde_inverse(num_points, num_points);
    // This should be vectorized when the functionality is implemented.
    for (size_t i = 0; i < num_points; i++) {
      for (size_t j = 0; j < num_points; j++) {
        vandermonde_inverse(i, j) =
            vandermonde_matrix(j, i) * weights[j] /
            compute_basis_function_normalization_square<BasisType>(i);
      }
    }
    return vandermonde_inverse;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct LinearFilterMatrixGenerator {
  Matrix operator()(const size_t num_points) const noexcept {
    // We implement the expression
    // \f$\mathcal{V}^{-1}\cdot\mathrm{diag}(1,1,0,0,...)\cdot\mathcal{V}\f$
    // (see description of `linear_filter_matrix`)
    // which multiplies the first two columns of
    // `nodal_to_modal_matrix` with the first two rows of
    // `modal_to_nodal_matrix`.
    Matrix lin_filter(num_points, num_points);
    dgemm_('N', 'N', num_points, num_points, std::min(size_t{2}, num_points),
           1.0,
           modal_to_nodal_matrix<BasisType, QuadratureType>(num_points).data(),
           num_points,
           nodal_to_modal_matrix<BasisType, QuadratureType>(num_points).data(),
           num_points, 0.0, lin_filter.data(), num_points);
    return lin_filter;
  }
};

}  // namespace

// Public interface

template <Basis BasisType, Quadrature QuadratureType>
const DataVector& collocation_points(const size_t num_points) noexcept {
  return precomputed_spectral_quantity<
             BasisType, QuadratureType,
             CollocationPointsAndWeightsGenerator<BasisType, QuadratureType>>(
             num_points)
      .first;
}

template <Basis BasisType, Quadrature QuadratureType>
const DataVector& quadrature_weights(const size_t num_points) noexcept {
  return precomputed_spectral_quantity<
      BasisType, QuadratureType,
      QuadratureWeightsGenerator<BasisType, QuadratureType>>(num_points);
}

/// \cond
// clang-tidy: Macro arguments should be in parentheses, but we want to append
// template parameters here.
#define PRECOMPUTED_SPECTRAL_QUANTITY(function_name, return_type,      \
                                      generator_name)                  \
  template <Basis BasisType, Quadrature QuadratureType>                \
  const return_type& function_name(const size_t num_points) noexcept { \
    return precomputed_spectral_quantity<                              \
        BasisType, QuadratureType,                                     \
        generator_name<BasisType, QuadratureType>>(/* NOLINT */        \
                                                   num_points);        \
  }

PRECOMPUTED_SPECTRAL_QUANTITY(differentiation_matrix, Matrix,
                              DifferentiationMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(integration_matrix, Matrix,
                              IntegrationMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(modal_to_nodal_matrix, Matrix,
                              ModalToNodalMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(nodal_to_modal_matrix, Matrix,
                              NodalToModalMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(linear_filter_matrix, Matrix,
                              LinearFilterMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY
/// \endcond

template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(const size_t num_points,
                            const T& target_points) noexcept {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  const DataVector& collocation_pts =
      collocation_points<BasisType, QuadratureType>(num_points);
  const DataVector& bary_weights =
      barycentric_weights<BasisType, QuadratureType>(num_points);
  const size_t num_target_points = get_size(target_points);
  Matrix interp_matrix(num_target_points, num_points);
  // Algorithm 32 in Kopriva, p. 76
  // This is valid for any collocation points.
  for (size_t k = 0; k < num_target_points; k++) {
    // Check where no interpolation is necessary since a target point
    // matches the original collocation points
    bool row_has_match = false;
    for (size_t j = 0; j < num_points; j++) {
      interp_matrix(k, j) = 0.0;
      if (equal_within_roundoff(get_element(target_points, k),
                                collocation_pts[j])) {
        interp_matrix(k, j) = 1.0;
        row_has_match = true;
      }
    }
    // Perform interpolation for non-matching points
    if (not row_has_match) {
      double sum = 0.0;
      for (size_t j = 0; j < num_points; j++) {
        interp_matrix(k, j) = bary_weights[j] / (get_element(target_points, k) -
                                                 collocation_pts[j]);
        sum += interp_matrix(k, j);
      }
      for (size_t j = 0; j < num_points; j++) {
        interp_matrix(k, j) /= sum;
      }
    }
  }
  return interp_matrix;
}

namespace {

template <typename F>
decltype(auto) get_spectral_quantity_for_mesh(F&& f,
                                              const Mesh<1>& mesh) noexcept {
  const auto num_points = mesh.extents(0);
  // Switch on runtime values of basis and quadrature to select
  // corresponding template specialization. For basis functions spanning
  // multiple dimensions we can generalize this function to take a
  // higher-dimensional Mesh.
  switch (mesh.basis(0)) {
    case Basis::Legendre:
      switch (mesh.quadrature(0)) {
        case Quadrature::Gauss:
          return f(std::integral_constant<Basis, Basis::Legendre>{},
                   std::integral_constant<Quadrature, Quadrature::Gauss>{},
                   num_points);
        case Quadrature::GaussLobatto:
          return f(
              std::integral_constant<Basis, Basis::Legendre>{},
              std::integral_constant<Quadrature, Quadrature::GaussLobatto>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::Chebyshev:
      switch (mesh.quadrature(0)) {
        case Quadrature::Gauss:
          return f(std::integral_constant<Basis, Basis::Chebyshev>{},
                   std::integral_constant<Quadrature, Quadrature::Gauss>{},
                   num_points);
        case Quadrature::GaussLobatto:
          return f(
              std::integral_constant<Basis, Basis::Chebyshev>{},
              std::integral_constant<Quadrature, Quadrature::GaussLobatto>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    default:
      ERROR("Missing basis case for spectral quantity");
  }
}

}  // namespace

/// \cond
// clang-tidy: Macro arguments should be in parentheses, but we want to append
// template parameters here.
#define SPECTRAL_QUANTITY_FOR_MESH(function_name, return_type)           \
  const return_type& function_name(const Mesh<1>& mesh) noexcept {       \
    return get_spectral_quantity_for_mesh(                               \
        [](const auto basis, const auto quadrature,                      \
           const size_t num_points) noexcept->const return_type& {       \
          return function_name</* NOLINT */ decltype(basis)::value,      \
                               decltype(quadrature)::value>(num_points); \
        },                                                               \
        mesh);                                                           \
  }

SPECTRAL_QUANTITY_FOR_MESH(collocation_points, DataVector)
SPECTRAL_QUANTITY_FOR_MESH(quadrature_weights, DataVector)
SPECTRAL_QUANTITY_FOR_MESH(differentiation_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(integration_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(modal_to_nodal_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(nodal_to_modal_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(linear_filter_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH
/// \endcond

template <typename T>
Matrix interpolation_matrix(const Mesh<1>& mesh,
                            const T& target_points) noexcept {
  return get_spectral_quantity_for_mesh(
      [target_points](const auto basis, const auto quadrature,
                      const size_t num_points) noexcept->Matrix {
        return interpolation_matrix<decltype(basis)::value,
                                    decltype(quadrature)::value>(num_points,
                                                                 target_points);
      },
      mesh);
}

}  // namespace Spectral

/// \cond HIDDEN_SYMBOLS
#define BASIS(data) BOOST_PP_TUPLE_ELEM(0, data)
#define QUAD(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                  \
  template const DataVector&                                                  \
      Spectral::collocation_points<BASIS(data), QUAD(data)>(size_t) noexcept; \
  template const DataVector&                                                  \
      Spectral::quadrature_weights<BASIS(data), QUAD(data)>(size_t) noexcept; \
  template const Matrix&                                                      \
      Spectral::differentiation_matrix<BASIS(data), QUAD(data)>(              \
          size_t) noexcept;                                                   \
  template const Matrix&                                                      \
      Spectral::integration_matrix<BASIS(data), QUAD(data)>(size_t) noexcept; \
  template const Matrix&                                                      \
      Spectral::nodal_to_modal_matrix<BASIS(data), QUAD(data)>(               \
          size_t) noexcept;                                                   \
  template const Matrix&                                                      \
      Spectral::modal_to_nodal_matrix<BASIS(data), QUAD(data)>(               \
          size_t) noexcept;                                                   \
  template const Matrix&                                                      \
      Spectral::linear_filter_matrix<BASIS(data), QUAD(data)>(                \
          size_t) noexcept;                                                   \
  template Matrix Spectral::interpolation_matrix<BASIS(data), QUAD(data)>(    \
      size_t, const DataVector&) noexcept;                                    \
  template Matrix Spectral::interpolation_matrix<BASIS(data), QUAD(data)>(    \
      size_t, const std::vector<double>&) noexcept;
template Matrix Spectral::interpolation_matrix(const Mesh<1>&,
                                               const DataVector&) noexcept;
template Matrix Spectral::interpolation_matrix(
    const Mesh<1>&, const std::vector<double>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Spectral::Basis::Chebyshev, Spectral::Basis::Legendre),
                        (Spectral::Quadrature::Gauss,
                         Spectral::Quadrature::GaussLobatto))

#undef BASIS
#undef QUAD
#undef INSTANTIATE
/// \endcond
