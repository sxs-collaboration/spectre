// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <ostream>

#include "DataStructures/Variables.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Index;
// IWYU pragma: no_forward_declare Variables
/// \endcond

namespace apply_matrices_detail {
template <typename ElementType, size_t Dim, bool... DimensionIsIdentity>
struct Impl {
  template <typename MatrixType>
  static void apply(gsl::not_null<ElementType*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const ElementType* data, const Index<Dim>& extents,
                    size_t number_of_independent_components);
};

template <typename MatrixType, size_t Dim>
size_t result_size(const std::array<MatrixType, Dim>& matrices,
                   const Index<Dim>& extents) {
  size_t num_points_result = 1;
  for (size_t d = 0; d < Dim; ++d) {
    const size_t cols = dereference_wrapper(gsl::at(matrices, d)).columns();
    if (cols == 0) {
      // An empty matrix is treated as the identity.
      num_points_result *= extents[d];
    } else {
      ASSERT(cols == extents[d],
             "Matrix " << d << " has wrong number of columns: " << cols
                       << " (expected " << extents[d] << ")");
      num_points_result *= dereference_wrapper(gsl::at(matrices, d)).rows();
    }
  }
  return num_points_result;
}
}  // namespace apply_matrices_detail

/// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Multiply by matrices in each dimension
///
/// Multiplies each stripe in the first dimension of `u` by
/// `matrices[0]`, each stripe in the second dimension of `u` by
/// `matrices[1]`, and so on.  If any of the matrices are empty they
/// will be treated as the identity, but the matrix multiplications
/// will be skipped for increased efficiency.
///
/// \note The element type stored in the vectors to be transformed may be either
/// `double` or `std::complex<double>`. The matrix, however, must be real. In
/// the case of acting on a vector of complex values, the matrix is treated as
/// having zero imaginary part. This is chosen for efficiency in all
/// use-cases for spectral matrix arithmetic so far encountered.
template <typename VariableTags, typename MatrixType, size_t Dim>
void apply_matrices(const gsl::not_null<Variables<VariableTags>*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const Variables<VariableTags>& u,
                    const Index<Dim>& extents) {
  ASSERT(u.number_of_grid_points() == extents.product(),
         "Mismatch between extents (" << extents.product()
                                      << ") and variables ("
                                      << u.number_of_grid_points() << ").");
  ASSERT(result->number_of_grid_points() ==
             apply_matrices_detail::result_size(matrices, extents),
         "result has wrong size.  Expected "
             << apply_matrices_detail::result_size(matrices, extents)
             << ", received " << result->number_of_grid_points());
  apply_matrices_detail::Impl<typename Variables<VariableTags>::value_type,
                              Dim>::apply(result->data(), matrices, u.data(),
                                          extents,
                                          u.number_of_independent_components);
}

template <typename VariableTags, typename MatrixType, size_t Dim>
Variables<VariableTags> apply_matrices(
    const std::array<MatrixType, Dim>& matrices,
    const Variables<VariableTags>& u, const Index<Dim>& extents) {
  Variables<VariableTags> result(
      apply_matrices_detail::result_size(matrices, extents));
  apply_matrices(make_not_null(&result), matrices, u, extents);
  return result;
}

// clang tidy mistakenly fails to identify this as a function definition
template <typename ResultType, typename MatrixType, typename VectorType,
          size_t Dim>
void apply_matrices(const gsl::not_null<ResultType*> result,  // NOLINT
                    const std::array<MatrixType, Dim>& matrices,
                    const VectorType& u, const Index<Dim>& extents) {
  const size_t number_of_independent_components = u.size() / extents.product();
  ASSERT(u.size() == number_of_independent_components * extents.product(),
         "The size of the vector u ("
             << u.size()
             << ") must be a multiple of the number of grid points ("
             << extents.product() << ").");
  ASSERT(result->size() ==
             number_of_independent_components *
                 apply_matrices_detail::result_size(matrices, extents),
         "result has wrong size.  Expected "
             << number_of_independent_components *
                    apply_matrices_detail::result_size(matrices, extents)
             << ", received " << result->size());
  apply_matrices_detail::Impl<typename VectorType::ElementType, Dim>::apply(
      result->data(), matrices, u.data(), extents,
      number_of_independent_components);
}

template <typename MatrixType, typename VectorType, size_t Dim>
VectorType apply_matrices(const std::array<MatrixType, Dim>& matrices,
                          const VectorType& u, const Index<Dim>& extents) {
  const size_t number_of_independent_components = u.size() / extents.product();
  VectorType result(number_of_independent_components *
                    apply_matrices_detail::result_size(matrices, extents));
  apply_matrices(make_not_null(&result), matrices, u, extents);
  return result;
}

template <typename ResultType, typename MatrixType, typename VectorType,
          size_t Dim>
ResultType apply_matrices(const std::array<MatrixType, Dim>& matrices,
                          const VectorType& u, const Index<Dim>& extents) {
  const size_t number_of_independent_components = u.size() / extents.product();
  ResultType result(number_of_independent_components *
                    apply_matrices_detail::result_size(matrices, extents));
  apply_matrices(make_not_null(&result), matrices, u, extents);
  return result;
}
/// @}
