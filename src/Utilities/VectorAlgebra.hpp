// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/Gsl.hpp"

// @{
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
  check_and_resize(result, lhs.size() * rhs.size());
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
// @}

// @{
/// \ingroup UtilitiesGroup
/// \brief Repeats a vector `times_to_repeat` times.
/// \details This can be useful for generating data that consists of the same
/// block of values duplicated a number of times. For instance, this can be used
/// to create a vector representing three-dimensional volume data from a
/// corresponding two-dimensional vector data, if the two-dimensional data
/// corresponds to the two fastest-varying directions of the desired
/// three-dimensional representation. The result would then be uniform in the
/// slowest-varying direction of the three dimensional grid.
template <typename VectorType>
void repeat(const gsl::not_null<VectorType*> result,
            const VectorType& to_repeat,
            const size_t times_to_repeat) noexcept {
  check_and_resize(result, to_repeat.size() * times_to_repeat);
  for (size_t i = 0; i < times_to_repeat; ++i) {
    VectorType view{result->data() + i * to_repeat.size(), to_repeat.size()};
    view = to_repeat;
  }
}

template <typename VectorType>
VectorType repeat(const VectorType& to_repeat,
                  const size_t times_to_repeat) noexcept {
  auto result = VectorType{to_repeat.size() * times_to_repeat};
  repeat(make_not_null(&result), to_repeat, times_to_repeat);
  return result;
}
// @}
