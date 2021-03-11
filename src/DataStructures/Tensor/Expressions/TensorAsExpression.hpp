// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent tensors

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace TensorExpressions {
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
          X, Symm, IndexList<Indices...>, ArgsList<Args...>> {
  using type = X;
  using symmetry = Symm;
  using index_list = IndexList<Indices...>;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = ArgsList<Args...>;

  /// Construct an expression from a Tensor
  explicit TensorAsExpression(const Tensor<X, Symm, IndexList<Indices...>>& t)
      : t_(&t) {}
  ~TensorAsExpression() override = default;

  /// \brief Computes a transformation from the LHS tensor's multi-indices to
  /// the equivalent RHS tensor's multi-indices, according to the differences in
  /// the orderings of their generic indices
  ///
  /// \details
  /// The elements of the transformation are the positions of the RHS generic
  /// indices in the LHS generic indices. Put another way, for some `i`,
  /// `rhs_tensorindices[i] == lhs_tensorindices[index_transformation[i]]`.
  ///
  /// Here is an example of what the algorithm does:
  ///
  /// Tensor equation: \f$L_{cab} = R_{abc}\f$
  /// `lhs_tensorindices`:
  /// \code
  /// {2, 0, 1} // i.e. {c, a, b}
  /// \endcode
  /// `rhs_tensorindices`:
  /// \code
  /// {0, 1, 2} // i.e. {a, b, c}
  /// \endcode
  /// returned `index_transformation`:
  /// \code
  /// {1, 2, 0} // positions of RHS indices {c, a, b} in LHS indices {a, b, c}
  /// \endcode
  ///
  /// \param lhs_tensorindices the TensorIndexs of the LHS tensor
  /// \return a transformation from the LHS tensor's multi-indices to the
  /// equivalent RHS tensor's multi-indices
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, num_tensor_indices>
  compute_index_transformation(const std::array<size_t, num_tensor_indices>&
                                   lhs_tensorindices) noexcept {
    constexpr std::array<size_t, num_tensor_indices> rhs_tensorindices = {
        {Args::value...}};
    std::array<size_t, num_tensor_indices> index_transformation{};
    for (size_t i = 0; i < num_tensor_indices; i++) {
      gsl::at(index_transformation, i) = static_cast<size_t>(std::distance(
          lhs_tensorindices.begin(),
          alg::find(lhs_tensorindices, gsl::at(rhs_tensorindices, i))));
    }
    return index_transformation;
  }

  /// \brief Computes the RHS tensor multi-index that is equivalent to a given
  /// LHS tensor multi-index, according to the differences in the orderings of
  /// their generic indices
  ///
  /// \details
  /// Here is an example of what the algorithm does:
  ///
  /// Tensor equation: \f$L_{cab} = R_{abc}\f$
  /// `index_transformation`:
  /// \code
  /// {1, 2, 0} // positions of RHS indices {c, a, b} in LHS indices {a, b, c}
  /// \endcode
  /// `lhs_multi_index`:
  /// \code
  /// {3, 4, 5} // i.e. c = 3, a = 4, b = 5
  /// \endcode
  /// returned equivalent `rhs_multi_index`:
  /// \code
  /// {4, 5, 3} // i.e. a = 4, b = 5, c = 3
  /// \endcode
  ///
  /// \param lhs_multi_index the multi-index of the LHS tensor
  /// \param index_transformation the list of the positions of the RHS indices
  /// in the LHS indices
  /// \return the RHS tensor multi-index that is equivalent to
  /// `lhs_tensor_index`
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, num_tensor_indices>
  compute_rhs_multi_index(
      const std::array<size_t, num_tensor_indices>& lhs_multi_index,
      const std::array<size_t, num_tensor_indices>&
          index_transformation) noexcept {
    std::array<size_t, num_tensor_indices> rhs_multi_index{};
    for (size_t i = 0; i < num_tensor_indices; i++) {
      gsl::at(rhs_multi_index, i) =
          gsl::at(lhs_multi_index, gsl::at(index_transformation, i));
    }
    return rhs_multi_index;
  }

  /// \brief Returns the value of a left hand side tensor's multi-index
  ///
  /// \details
  /// One big challenge with TensorExpression implementation is the reordering
  /// of the indices on the left hand side (LHS) and right hand side (RHS) of
  /// the expression. The algorithms implemented in
  /// `compute_index_transformation` and `compute_rhs_multi_index` handle the
  /// index sorting by mapping between the generic index orders of the LHS and
  /// RHS tensors.
  ///
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the LHS of the tensor
  /// expression
  /// \param lhs_multi_index the multi-index of the LHS tensor component to
  /// retrieve
  /// \return the value of the DataType of the component at `lhs_multi_index` in
  /// the LHS tensor
  template <typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& lhs_multi_index)
      const noexcept {
    if constexpr (std::is_same_v<tmpl::list<LhsIndices...>,
                                 tmpl::list<Args...>>) {
      return t_->get(lhs_multi_index);
    } else {
      constexpr std::array<size_t, num_tensor_indices> index_transformation =
          compute_index_transformation({{LhsIndices::value...}});
      return t_->get(
          compute_rhs_multi_index(lhs_multi_index, index_transformation));
    }
  }

  /// Retrieve the i'th entry of the Tensor being held
  SPECTRE_ALWAYS_INLINE type operator[](const size_t i) const {
    return t_->operator[](i);
  }

 private:
  const Tensor<X, Symm, IndexList<Indices...>>* t_ = nullptr;
};
}  // namespace TensorExpressions
