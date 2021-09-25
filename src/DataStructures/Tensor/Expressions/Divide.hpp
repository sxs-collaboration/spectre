// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor division by scalars

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the quotient of one tensor
/// expression divided by another tensor expression that evaluates to a rank 0
/// tensor
///
/// \tparam T1 the numerator operand expression of the division expression
/// \tparam T2 the denominator operand expression of the division expression
/// \tparam Args2 the generic indices of the denominator expression
template <typename T1, typename T2, typename... Args2>
struct Divide : public TensorExpression<
                    Divide<T1, T2, Args2...>,
                    typename std::conditional_t<
                        std::is_same<typename T1::type, DataVector>::value or
                            std::is_same<typename T2::type, DataVector>::value,
                        DataVector, double>,
                    typename T1::symmetry, typename T1::index_list,
                    typename T1::args_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value or
                    std::is_same<T1, NumberAsExpression>::value,
                "Cannot divide TensorExpressions holding different data types");
  static_assert((... and tt::is_time_index<Args2>::value),
                "Can only divide a tensor expression by a double or a tensor "
                "expression that evaluates to "
                "a rank 0 tensor.");

  using type =
      std::conditional_t<std::is_same<typename T1::type, DataVector>::value or
                             std::is_same<typename T2::type, DataVector>::value,
                         DataVector, double>;
  using symmetry = typename T1::symmetry;
  using index_list = typename T1::index_list;
  using args_list = typename T1::args_list;
  static constexpr auto num_tensor_indices =
      tmpl::size<typename T1::index_list>::value;
  static constexpr auto op2_num_tensor_indices =
      tmpl::size<typename T2::index_list>::value;
  // the denominator has no indices or all time indices
  static constexpr auto op2_multi_index =
      make_array<op2_num_tensor_indices, size_t>(0);

  Divide(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~Divide() override = default;

  /// \brief Return the value of the component of the quotient tensor at a given
  /// multi-index
  ///
  /// \param result_multi_index the multi-index of the component of the quotient
  //// tensor to retrieve
  /// \return the value of the component in the quotient tensor at
  /// `result_multi_index`
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    return t1_.get(result_multi_index) / t2_.get(op2_multi_index);
  }

 private:
  T1 t1_;
  T2 t2_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of one tensor
/// expression over another tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t2` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t2` could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
/// - `R(ti_t, ti_t) + 1.0`
///
/// \param t1 the tensor expression numerator
/// \param t2 the rank 0 tensor expression denominator
template <typename T1, typename T2, typename... Args2>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T1, typename T1::type, typename T1::symmetry,
                           typename T1::index_list, typename T1::args_list>& t1,
    const TensorExpression<T2, typename T2::type, typename T2::symmetry,
                           typename T2::index_list, tmpl::list<Args2...>>& t2) {
  return TensorExpressions::Divide<T1, T2, Args2...>(~t1, ~t2);
}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a tensor
/// expression over a `double`
///
/// \note The implementation instead uses the operation, `t * (1.0 / number)`
///
/// \param t the tensor expression operand of the quotient
/// \param number the `double` operand of the quotient
/// \return the tensor expression representing the quotient of a tensor
/// expression and a `double`
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const double number) {
  return t * TensorExpressions::NumberAsExpression(1.0 / number);
}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a `double`
/// over a tensor expression that evaluates to a rank 0 tensor
///
/// \param number the `double` numerator of the quotient
/// \param t the tensor expression denominator of the quotient
/// \return the tensor expression representing the quotient of a `double` over a
/// tensor expression that evaluates to a rank 0 tensor
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator/(
    const double number,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return TensorExpressions::NumberAsExpression(number) / t;
}
