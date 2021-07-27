// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \tparam T the type of the tensor expression of which to take the square
/// root
template <typename T>
struct SquareRoot
    : public TensorExpression<SquareRoot<T>, typename T::type, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>> {
  static_assert(
      std::is_base_of<TensorExpression<T, typename T::type, tmpl::list<>,
                                       tmpl::list<>, tmpl::list<>>,
                      T>::value,
      "Can only take the square root of a tensor expression that evaluates to "
      "a rank 0 tensor.");
  using type = typename T::type;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  static constexpr auto num_tensor_indices = 0;

  SquareRoot(T t) : t_(std::move(t)) {}
  ~SquareRoot() override = default;

  /// \brief Returns the square root of the component of the tensor evaluated
  /// from the contained tensor expression
  ///
  /// \details
  /// SquareRoot only supports tensor expressions that evaluate to a rank 0
  /// Tensor. This is why `multi_index` is always an array of size 0.
  ///
  /// \param multi_index the multi-index of the component of which to take the
  /// square root
  /// \return the square root of the component of the tensor evaluated from the
  /// contained tensor expression
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index)
      const noexcept {
    return sqrt(t_.get(multi_index));
  }

 private:
  T t_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t` could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
///
/// \param t the type of the tensor expression of which to take the square root
template <typename T>
SPECTRE_ALWAYS_INLINE auto sqrt(
    const TensorExpression<T, typename T::type, tmpl::list<>, tmpl::list<>,
                           tmpl::list<>>& t) {
  return TensorExpressions::SquareRoot(~t);
}
