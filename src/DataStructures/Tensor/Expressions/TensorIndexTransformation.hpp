// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions useful for transforming tensor multi-indices according to
/// a different generic index order

#pragma once

#include <array>
#include <cstddef>
#include <iterator>

#include "Utilities/Algorithm.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace TensorExpressions {
/// \brief Computes a transformation from one generic tensor index order to
/// another
///
/// \details
/// The elements of the transformation are the positions of the second list of
/// generic indices in the first list of generic indices. Put another way, for
/// some `i`, `tensorindices2[i] == tensorindices1[index_transformation[i]]`.
///
/// Here is an example of what the algorithm does:
///
/// Transformation between (1) \f$R_{cab}\f$ and (2) \f$S_{abc}\f$
/// `tensorindices1`:
/// \code
/// {2, 0, 1} // TensorIndex values for {c, a, b}
/// \endcode
/// `tensorindices2`:
/// \code
/// {0, 1, 2} // TensorIndex values for {a, b, c}
/// \endcode
/// returned `tensorindex_transformation`:
/// \code
/// {1, 2, 0} // positions of S' indices {a, b, c} in R's indices {c, a, b}
/// \endcode
///
/// \tparam NumIndices the number of indices
/// \param tensorindices1 the TensorIndex values of the first generic index
/// order
/// \param tensorindices2 the TensorIndex values of the second generic index
/// order
/// \return a transformation from the first generic index order to the second
template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices>
compute_tensorindex_transformation(
    const std::array<size_t, NumIndices>& tensorindices1,
    const std::array<size_t, NumIndices>& tensorindices2) noexcept {
  std::array<size_t, NumIndices> tensorindex_transformation{};
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(tensorindex_transformation, i) = static_cast<size_t>(
        std::distance(tensorindices1.begin(),
                      alg::find(tensorindices1, gsl::at(tensorindices2, i))));
  }
  return tensorindex_transformation;
}

/// \brief Computes the tensor multi-index that is equivalent to a given tensor
/// multi-index, according to the differences in their generic index orders
///
/// \details
/// Here is an example of what the algorithm does:
///
/// Transform (input) multi-index of \f$R_{cab}\f$ to the equivalent (output)
/// multi-index of \f$S_{abc}\f$
/// `tensorindex_transformation`:
/// \code
/// {1, 2, 0} // positions of S' indices {a, b, c} in R's indices {c, a, b}
/// \endcode
/// `input_multi_index`:
/// \code
/// {3, 4, 5} // i.e. c = 3, a = 4, b = 5
/// \endcode
/// returned equivalent `output_multi_index`:
/// \code
/// {4, 5, 3} // i.e. a = 4, b = 5, c = 3
/// \endcode
///
/// \tparam NumIndices the number of indices
/// \param input_multi_index the input tensor multi-index to transform
/// \param tensorindex_transformation the positions of the output's generic
/// indices in the input's generic indices (see example in details)
/// \return the output tensor multi-index that is equivalent to
/// `input_multi_index`, according to generic index order differences
// (`tensorindex_transformation`)
template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices>
transform_multi_index(
    const std::array<size_t, NumIndices>& input_multi_index,
    const std::array<size_t, NumIndices>& tensorindex_transformation) noexcept {
  std::array<size_t, NumIndices> output_multi_index{};
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(output_multi_index, i) =
        gsl::at(input_multi_index, gsl::at(tensorindex_transformation, i));
  }
  return output_multi_index;
}
}  // namespace TensorExpressions
