// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details The expression can have a non-zero number of indices as long as
/// all indices are concrete time indices, as this represents a rank 0 tensor.
///
/// \tparam T the type of the tensor expression of which to take the square
/// root
/// \tparam Args the TensorIndexs of the expression
template <typename T, typename... Args>
struct SquareRoot
    : public TensorExpression<SquareRoot<T, Args...>, typename T::type,
                              typename T::symmetry, typename T::index_list,
                              tmpl::list<Args...>> {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only take the square root of a tensor expression that evaluates to "
      "a rank 0 tensor.");

  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = typename T::type;
  /// The ::Symmetry of the result of the expression
  using symmetry = typename T::symmetry;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = typename T::index_list;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = tmpl::list<Args...>;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices = sizeof...(Args);

  // === Arithmetic tensor operations properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand
  static constexpr size_t num_ops_left_child = T::num_ops_subtree;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand. This is 0 because this expression represents a unary
  /// operation.
  static constexpr size_t num_ops_right_child = 0;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree
  static constexpr size_t num_ops_subtree = num_ops_left_child + 1;

  SquareRoot(T t) : t_(std::move(t)) {}
  ~SquareRoot() override = default;

  /// \brief Returns the square root of the component of the tensor evaluated
  /// from the contained tensor expression
  ///
  /// \details
  /// SquareRoot only supports tensor expressions that evaluate to a rank 0
  /// Tensor This is why `multi_index` is always an array of size 0.
  ///
  /// \param multi_index the multi-index of the component of which to take the
  /// square root
  /// \return the square root of the component of the tensor evaluated from the
  /// contained tensor expression
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return sqrt(t_.get(multi_index));
  }

 private:
  /// Operand expression
  T t_;
};
}  // namespace tenex

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t` could take:
/// - `R()`
/// - `R(ti::A, ti::a)`
/// - `(R(ti::A, ti::B) * S(ti::a, ti::b))`
/// - `R(ti::t, ti::t) + 1.0`
///
/// \param t the tensor expression of which to take the square root
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto sqrt(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t) {
  return tenex::SquareRoot<T, Args...>(~t);
}
