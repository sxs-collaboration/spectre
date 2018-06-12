// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Index;
// IWYU pragma: no_forward_declare Variables
/// \endcond

namespace apply_matrices_detail {
template <size_t Dim, bool... DimensionIsIdentity>
struct Impl {
  template <typename MatrixType>
  static void apply(gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* data, const Index<Dim>& extents,
                    size_t number_of_independent_components) noexcept;
};

template <typename MatrixType, size_t Dim>
size_t result_size(const std::array<MatrixType, Dim>& matrices,
                   const Index<Dim>& extents) noexcept {
  size_t num_points_result = 1;
  for (size_t d = 0; d < Dim; ++d) {
    const size_t cols = dereference_wrapper(gsl::at(matrices, d)).columns();
    if (cols == 0) {
      // An empty matrix is treated as the identity.
      num_points_result *= extents[d];
    } else {
      ASSERT(cols == extents[d],
             "Matrix " << d << " has wrong number of columns: "
             << cols << " (expected " << extents[d] << ")");
      num_points_result *= dereference_wrapper(gsl::at(matrices, d)).rows();
    }
  }
  return num_points_result;
}
}  // namespace apply_matrices_detail

/// \ingroup NumericalAlgorithmsGroup
/// \brief Multiply by matrices in each dimension
///
/// Multiplies each stripe in the first dimension of `u` by
/// `matrices[0]`, each stripe in the second dimension of `u` by
/// `matrices[1]`, and so on.  If any of the matrices are empty they
/// will be treated as the identity, but the matrix multiplications
/// will be skipped for increased efficiency.
//@{
template <typename VariableTags, typename MatrixType, size_t Dim>
void apply_matrices(const gsl::not_null<Variables<VariableTags>*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const Variables<VariableTags>& u,
                    const Index<Dim>& extents) noexcept {
  ASSERT(u.number_of_grid_points() == extents.product(),
         "Mismatch between extents (" << extents.product()
         << ") and variables (" << u.number_of_grid_points() << ").");
  ASSERT(result->number_of_grid_points() ==
         apply_matrices_detail::result_size(matrices, extents),
         "result has wrong size.  Expected "
         << apply_matrices_detail::result_size(matrices, extents)
         << ", received " << result->number_of_grid_points());
  apply_matrices_detail::Impl<Dim>::apply(result->data(), matrices, u.data(),
                                          extents,
                                          u.number_of_independent_components);
}

template <typename VariableTags, typename MatrixType, size_t Dim>
Variables<VariableTags> apply_matrices(
    const std::array<MatrixType, Dim>& matrices,
    const Variables<VariableTags>& u, const Index<Dim>& extents) noexcept {
  Variables<VariableTags> result(
      apply_matrices_detail::result_size(matrices, extents));
  apply_matrices(make_not_null(&result), matrices, u, extents);
  return result;
}

template <typename MatrixType, size_t Dim>
void apply_matrices(const gsl::not_null<DataVector*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const DataVector& u, const Index<Dim>& extents) noexcept {
  ASSERT(u.size() == extents.product(),
         "Mismatch between extents (" << extents.product() << ") and size ("
         << u.size() << ").");
  ASSERT(result->size() ==
         apply_matrices_detail::result_size(matrices, extents),
         "result has wrong size.  Expected "
         << apply_matrices_detail::result_size(matrices, extents)
         << ", received " << result->size());
  apply_matrices_detail::Impl<Dim>::apply(result->data(), matrices, u.data(),
                                          extents, 1);
}

template <typename MatrixType, size_t Dim>
DataVector apply_matrices(const std::array<MatrixType, Dim>& matrices,
                          const DataVector& u,
                          const Index<Dim>& extents) noexcept {
  DataVector result(apply_matrices_detail::result_size(matrices, extents));
  apply_matrices(make_not_null(&result), matrices, u, extents);
  return result;
}
//@}
