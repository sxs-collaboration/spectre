// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent tensors

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
namespace detail {
/// @{
/// \brief Given a tensor symmetry and the positions of indices where a generic
/// spatial index is used for a spacetime index and the positions where a
/// concrete time index is used, this returns the symmetry after making those
/// indices nonsymmetric with others that do not do the same
///
/// \details
/// Example: Let `symmetry` be `{1, 2, 2, 2, 1}`. Let
/// `spatial_spacetime_index_positions` be `{1, 2}`: the 2nd and 3rd indices
/// of the tensor are spacetime indices where a generic spatial index is used.
/// Let `time_index_positions` be `{4}`: the 5th (last) index is a spacetime
/// index where a concrete time index is used. The resulting symmetry will
/// reflect that the 2nd and 3rd indices are still symmetric with each other,
/// but no longer symmetric with the 4th index, and the 5th index will no
/// longer be symmetric with the 1st index. Therefore, the returned symmetry
/// will be equivalent to the form of `{4, 3, 3, 2, 1}`.
///
/// Note: the symmetry returned by this function is not necessarily in the
/// canonical form specified by ::Symmetry. For example, the actual array
/// returned in the example above is `{1, 4, 4, 2, 5}`. As such, this array
/// still encodes which indices are symmetric, but it is not in the canonical
/// form specified by ::Symmetry, which is `{4, 3, 3, 2, 1}`.
///
/// \param symmetry the input tensor symmetry to transform
/// \param spatial_spacetime_index_positions the positions of the indices of the
/// tensor where a generic spatial index is used for a spacetime index
/// \param time_index_positions the positions of the indices of the tensor where
/// a concrete time index is used for a spacetime index
/// \return the symmetry after making the `spatial_spacetime_index_positions`
/// and `time_index_positions` of `symmetry` nonsymmetric with indices at other
/// positions
template <
    size_t NumIndices, size_t NumSpatialSpacetimeIndices,
    size_t NumConcreteTimeIndices,
    Requires<(NumIndices >= 2 and (NumSpatialSpacetimeIndices != 0 or
                                   NumConcreteTimeIndices != 0))> = nullptr>
constexpr std::array<std::int32_t, NumIndices>
get_transformed_spacetime_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
        spatial_spacetime_index_positions,
    const std::array<size_t, NumConcreteTimeIndices>& time_index_positions) {
  std::array<std::int32_t, NumIndices> transformed_spacetime_symmetry{};
  const std::int32_t max_symm_value =
      static_cast<std::int32_t>(*alg::max_element(symmetry));
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(transformed_spacetime_symmetry, i) = gsl::at(symmetry, i);
  }
  for (size_t i = 0; i < NumSpatialSpacetimeIndices; i++) {
    gsl::at(transformed_spacetime_symmetry,
            gsl::at(spatial_spacetime_index_positions, i)) += max_symm_value;
  }
  for (size_t i = 0; i < NumConcreteTimeIndices; i++) {
    gsl::at(transformed_spacetime_symmetry, gsl::at(time_index_positions, i)) +=
        2 * max_symm_value;
  }

  return transformed_spacetime_symmetry;
}

template <size_t NumIndices, size_t NumSpatialSpacetimeIndices,
          size_t NumConcreteTimeIndices,
          Requires<(NumIndices < 2 or (NumSpatialSpacetimeIndices == 0 and
                                       NumConcreteTimeIndices == 0))> = nullptr>
constexpr std::array<std::int32_t, NumIndices>
get_transformed_spacetime_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
    /*spatial_spacetime_index_positions*/,
    const std::array<size_t, NumConcreteTimeIndices>&
    /*time_index_positions*/) {
  return symmetry;
}
/// @}

/// \ingroup TensorExpressionsGroup
/// \brief Helper struct for computing the new canonical symmetry of a tensor
/// after generic spatial indices are used for any of the tensor's spacetime
/// indices
///
/// \details This is relevant in cases where a tensor has spacetime indices that
/// are symmetric but generic spatial indices are used for a non-empty subset of
/// those symmetric spacetime indices. For example, if we have some rank 3
/// tensor with the first index being spatial and the 2nd and third indices
/// spacetime and symmetric, but a generic spatial index is used for the 2nd
/// index, the "result" of the single tensor expression, \f$R_{ija}\f$, is a
/// rank 3 tensor whose 2nd and 3rd indices are no longer symmetric.

/// Given that some `Tensor` named `R` that represents the tensor in the above
/// example, the symmetry of the `Tensor` is `[2, 1, 1]`, but the computed
/// symmetry of the `TensorAsExpression` that represents it will have symmetry
/// `[3, 2, 1]` to reflect this loss of symmetry.
///
/// Evaluating the "result" symmetry here in `TensorAsExpression`, at the leaves
/// of the expression tree, enables the propagation of this symmetry up the tree
/// to the other expression types. By determining each tensor's "result"
/// symmetry at the leaves, the expressions at internal nodes of the tree can
/// have their individual symmetries determined without having to each consider
/// whether their operand(s) are expression(s) that have spacetime indices where
/// generic spatial indices were used.
///
/// \tparam SymmList the ::Symmetry of the Tensor represented by the expression
/// \tparam TensorIndexTypeList the \ref SpacetimeIndex "TensorIndexType"'s of
/// the Tensor represented by the expression
/// \tparam TensorIndexList the generic indices of the Tensor represented by the
/// expression
template <typename SymmList, typename TensorIndexTypeList,
          typename TensorIndexList,
          size_t NumIndices = tmpl::size<SymmList>::value,
          typename IndexSequence = std::make_index_sequence<NumIndices>>
struct TensorAsExpressionSymm;

template <template <typename...> class SymmList, typename... Symm,
          typename TensorIndexTypeList, typename TensorIndexList,
          size_t NumIndices, size_t... Ints>
struct TensorAsExpressionSymm<SymmList<Symm...>, TensorIndexTypeList,
                              TensorIndexList, NumIndices,
                              std::index_sequence<Ints...>> {
  static constexpr auto symm = get_transformed_spacetime_symmetry<NumIndices>(
      {{Symm::value...}},
      get_spatial_spacetime_index_positions<TensorIndexTypeList,
                                            TensorIndexList>(),
      get_time_index_positions<TensorIndexList>());
  using type = Symmetry<symm[Ints]...>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a Tensor
///
/// \details
/// In order to represent a tensor as an expression, instead of having Tensor
/// derive off of TensorExpression, a TensorAsExpression derives off of
/// TensorExpression and contains a pointer to a Tensor. The reason having
/// Tensor derive off of TensorExpression is problematic is that the index
/// structure is part of the type of the TensorExpression, so every possible
/// permutation and combination of indices must be derived from. For a rank 3
/// tensor, this is already over 500 base classes, which the Intel compiler
/// takes too long to compile.
///
/// \tparam T the type of Tensor being represented as an expression
/// \tparam ArgsList the tensor indices, e.g. `_a` and `_b` in `F(_a, _b)`
template <typename T, typename ArgsList>
struct TensorAsExpression;

template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorAsExpression<Tensor<X, Symm, IndexList<Indices...>>,
                          ArgsList<Args...>>
    : public TensorExpression<
          TensorAsExpression<Tensor<X, Symm, IndexList<Indices...>>,
                             ArgsList<Args...>>,
          X,
          typename detail::TensorAsExpressionSymm<Symm, IndexList<Indices...>,
                                                  ArgsList<Args...>>::type,
          IndexList<Indices...>, ArgsList<Args...>> {
  using type = X;
  using symmetry =
      typename detail::TensorAsExpressionSymm<Symm, IndexList<Indices...>,
                                              ArgsList<Args...>>::type;
  using index_list = IndexList<Indices...>;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = ArgsList<Args...>;

  /// Construct an expression from a Tensor
  explicit TensorAsExpression(const Tensor<X, Symm, IndexList<Indices...>>& t)
      : t_(&t) {}
  ~TensorAsExpression() override = default;

  /// \brief Returns the value of the contained tensor's multi-index
  ///
  /// \param multi_index the multi-index of the tensor component to retrieve
  /// \return the value of the component at `multi_index` in the tensor
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return t_->get(multi_index);
  }

  /// Retrieve the i'th entry of the Tensor being held
  SPECTRE_ALWAYS_INLINE type operator[](const size_t i) const {
    return t_->operator[](i);
  }

 private:
  const Tensor<X, Symm, IndexList<Indices...>>* t_ = nullptr;
};
}  // namespace TensorExpressions
