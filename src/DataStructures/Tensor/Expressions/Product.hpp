// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor inner and outer products

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
namespace detail {
template <typename T1, typename T2, typename SymmList1 = typename T1::symmetry,
          typename SymmList2 = typename T2::symmetry>
struct OuterProductType;

template <typename T1, typename T2, template <typename...> class SymmList1,
          typename... Symm1, template <typename...> class SymmList2,
          typename... Symm2>
struct OuterProductType<T1, T2, SymmList1<Symm1...>, SymmList2<Symm2...>> {
  using type =
      std::conditional_t<std::is_same<typename T1::type, DataVector>::value or
                             std::is_same<typename T2::type, DataVector>::value,
                         DataVector, double>;
  using symmetry =
      Symmetry<(Symm1::value + sizeof...(Symm2))..., Symm2::value...>;
  using index_list =
      tmpl::append<typename T1::index_list, typename T2::index_list>;
  using tensorindex_list =
      tmpl::append<typename T1::args_list, typename T2::args_list>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the outer product of two
/// tensor expressions
///
/// \tparam T1 the first operand expression of the outer product expression
/// \tparam T2 the second operand expression of the outer product expression
template <typename T1, typename T2,
          typename IndexList1 = typename T1::index_list,
          typename IndexList2 = typename T2::index_list,
          typename ArgsList1 = typename T1::args_list,
          typename ArgsList2 = typename T2::args_list>
struct OuterProduct;

template <typename T1, typename T2, template <typename...> class IndexList1,
          typename... Indices1, template <typename...> class IndexList2,
          typename... Indices2, template <typename...> class ArgsList1,
          typename... Args1, template <typename...> class ArgsList2,
          typename... Args2>
struct OuterProduct<T1, T2, IndexList1<Indices1...>, IndexList2<Indices2...>,
                    ArgsList1<Args1...>, ArgsList2<Args2...>>
    : public TensorExpression<
          OuterProduct<T1, T2>, typename detail::OuterProductType<T1, T2>::type,
          typename detail::OuterProductType<T1, T2>::symmetry,
          typename detail::OuterProductType<T1, T2>::index_list,
          typename detail::OuterProductType<T1, T2>::tensorindex_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value or
                    std::is_same<T1, NumberAsExpression>::value or
                    std::is_same<T2, NumberAsExpression>::value,
                "Cannot product Tensors holding different data types.");

  using type = typename detail::OuterProductType<T1, T2>::type;
  using symmetry = typename detail::OuterProductType<T1, T2>::symmetry;
  using index_list = typename detail::OuterProductType<T1, T2>::index_list;
  using args_list = typename detail::OuterProductType<T1, T2>::tensorindex_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  static constexpr auto op1_num_tensor_indices =
      tmpl::size<typename T1::index_list>::value;
  static constexpr auto op2_num_tensor_indices =
      num_tensor_indices - op1_num_tensor_indices;

  OuterProduct(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~OuterProduct() override = default;

  /// \brief Return the value of the component of the outer product tensor at a
  /// given multi-index
  ///
  /// \details
  /// This function takes the multi-index of some component of the resultant
  /// outer product to compute. The function first computes the multi-indices of
  /// the pair of components in the two operand expressions, then multiplies the
  /// values at these multi-indices to obtain the value of the resultant outer
  /// product component. For example, say we are evaluating
  /// \f$L_abc = R_{b} * S_{ca}\f$. Let `result_multi_index == {0, 1, 2}`, which
  /// refers to the component \f$L_{012}\f$, the component we wish to compute.
  /// This function will compute the multi-indices of the operands that
  /// correspond to \f$R_{1}\f$ and \f$S_{20}\f$, retrieve their values, and
  /// return their product.
  ///
  /// \param result_multi_index the multi-index of the component of the outer
  /// product tensor to retrieve
  /// \return the value of the component at `result_multi_index` in the outer
  /// product tensor
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    std::array<size_t, op1_num_tensor_indices> op1_multi_index{};
    for (size_t i = 0; i < op1_num_tensor_indices; i++) {
      gsl::at(op1_multi_index, i) = gsl::at(result_multi_index, i);
    }

    std::array<size_t, op2_num_tensor_indices> op2_multi_index{};
    for (size_t i = 0; i < op2_num_tensor_indices; i++) {
      gsl::at(op2_multi_index, i) =
          gsl::at(result_multi_index, op1_num_tensor_indices + i);
    }

    return t1_.get(op1_multi_index) * t2_.get(op2_multi_index);
  }

 private:
  T1 t1_;
  T2 t2_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the product of two tensor
/// expressions
///
/// \details
/// If the two operands have N pairs of indices that need to be contracted, the
/// returned expression will be an `OuterProduct` expression nested inside N
/// `TensorContract` expressions. This represents computing the inner product
/// of the outer product of the two operands. If the operands do not have any
/// indices to be contracted, the returned expression will be an `OuterProduct`.
///
/// The two arguments are expressions that contain the two operands of the
/// product, where the types of the operands are `T1` and `T2`.
///
/// \tparam T1 the derived TensorExpression type of the first operand of the
/// product
/// \tparam T2 the derived TensorExpression type of the second operand of the
/// product
/// \tparam ArgsList1 the TensorIndexs of the first operand
/// \tparam ArgsList2 the TensorIndexs of the second operand
/// \param t1 first operand expression of the product
/// \param t2 the second operand expression of the product
/// \return the tensor expression representing the product of two tensor
/// expressions
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T1, typename T1::type, typename T1::symmetry,
                           typename T1::index_list, ArgsList1>& t1,
    const TensorExpression<T2, typename T2::type, typename T2::symmetry,
                           typename T2::index_list, ArgsList2>& t2) {
  return TensorExpressions::contract(
      TensorExpressions::OuterProduct<T1, T2>(~t1, ~t2));
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the product of a tensor
/// expression and a `double`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the product
/// \tparam X the type of data stored in the tensor expression operand of the
/// product
/// \tparam ArgsList the TensorIndexs of the tensor expression operand of the
/// product
/// \param t the tensor expression operand of the product
/// \param number the `double` operand of the product
/// \return the tensor expression representing the product of a tensor
/// expression and a `double`
template <typename T, typename X, typename ArgsList>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T, X, typename T::symmetry, typename T::index_list,
                           ArgsList>& t,
    const double number) {
  return t * TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X, typename ArgsList>
SPECTRE_ALWAYS_INLINE auto operator*(
    const double number,
    const TensorExpression<T, X, typename T::symmetry, typename T::index_list,
                           ArgsList>& t) {
  return TensorExpressions::NumberAsExpression(number) * t;
}
/// @}
