// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/Blaze.hpp"
#include "Utilities/Gsl.hpp"

/// @{
/// \ingroup UtilitiesGroup
/// \brief Computes the outer product between two vectors.
/// \details For vectors \f$A\f$ and \f$B\f$, the resulting outer product is
/// \f$\{A_1 B_1,\, A_2 B_1\, \dots\, A_N B_1,\, A_1 B_2\, \dots\, A_N B_M\}\f$.
/// This is useful for generating separable volume data from its constituent
/// inputs.
template <typename LhsVectorType, typename RhsVectorType,
          typename ResultVectorType =
              typename blaze::MultTrait<LhsVectorType, RhsVectorType>::Type>
void outer_product(const gsl::not_null<ResultVectorType*> result,
                   const LhsVectorType& lhs,
                   const RhsVectorType& rhs) noexcept {
  result->destructive_resize(lhs.size() * rhs.size());
  for (size_t i = 0; i < rhs.size(); ++i) {
    ResultVectorType view{result->data() + i * lhs.size(), lhs.size()};
    view = rhs[i] * lhs;
  }
}

template <typename LhsVectorType, typename RhsVectorType,
          typename ResultVectorType =
              typename blaze::MultTrait<LhsVectorType, RhsVectorType>::Type>
ResultVectorType outer_product(const LhsVectorType& lhs,
                               const RhsVectorType& rhs) noexcept {
  auto result = ResultVectorType{lhs.size() * rhs.size()};
  outer_product(make_not_null(&result), lhs, rhs);
  return result;
}
/// @}

/// @{
/// \ingroup UtilitiesGroup
/// \brief Creates or fills a vector with data from `to_repeat` copied
/// `times_to_repeat`  times in sequence.
///
/// \details This can be useful for generating data that consists of the same
/// block of values duplicated a number of times. For instance, this can be used
/// to create a vector representing three-dimensional volume data from a
/// corresponding two-dimensional vector data, if the two-dimensional data
/// corresponds to the two fastest-varying directions of the desired
/// three-dimensional representation. The result would then be uniform in the
/// slowest-varying direction of the three dimensional grid.
template <typename VectorType>
void fill_with_n_copies(
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    const gsl::not_null<VectorType*> result, const VectorType& to_copy,
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    const size_t times_to_copy) noexcept {
  result->destructive_resize(to_copy.size() * times_to_copy);
  for (size_t i = 0; i < times_to_copy; ++i) {
    VectorType view{result->data() + i * to_copy.size(), to_copy.size()};
    view = to_copy;
  }
}

// clang-tidy incorrectly believes this to be a forward-declaration
template <typename VectorType>
VectorType create_vector_of_n_copies(
    const VectorType& to_copy,
    const size_t times_to_copy) noexcept {  // NOLINT
  auto result = VectorType{to_copy.size() * times_to_copy};
  fill_with_n_copies(make_not_null(&result), to_copy, times_to_copy);
  return result;
}
/// @}
