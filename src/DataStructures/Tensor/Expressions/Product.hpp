// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor inner and outer products

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
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
  static constexpr auto num_tensor_indices_first_operand =
      tmpl::size<typename T1::index_list>::value;
  static constexpr auto num_tensor_indices_second_operand =
      num_tensor_indices - num_tensor_indices_first_operand;

  OuterProduct(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~OuterProduct() override = default;

  /// \ingroup TensorExpressionsGroup
  /// \brief Helper struct for computing the multi-index of a component of an
  /// operand of the outer product from the multi-index of a component of the
  /// outer product
  ///
  /// \details
  /// While `OperandTensorIndexList` will either be the first operand's generic
  /// indices, `ArgsList1<Args1...>`, or the second operand's generic indices,
  /// `ArgsList2<Args2...>`, and both of these are accessible within the class,
  /// this struct wraps the core functionality and is templated with
  /// `OperandTensorIndexList` so that the functionality can be reused for each
  /// operand's generic index list.
  ///
  /// \tparam OperandTensorIndexList the operand's list of TensorIndexs
  template <typename OperandTensorIndexList>
  struct GetOpTensorMultiIndex;

  template <typename... OperandTensorIndices>
  struct GetOpTensorMultiIndex<tmpl::list<OperandTensorIndices...>> {
    /// \ingroup TensorExpressionsGroup
    /// \brief Computes the multi-index of a component of an operand of the
    /// outer product from the multi-index of a component of the outer product
    ///
    /// \details
    /// Example: Let's say we are evaluating \f$L_abc = R_{b} * S_{ca}\f$. Let
    /// `ti_a_t` denote the type of `ti_a`, and apply the same convention for
    /// other generic indices. Let `LhsTensorIndices == ti_a_t, ti_b_t, ti_c_t`,
    /// and `OperandTensorIndices` is either `ti_b_t` or `ti_c_t, ti_a_t`. Let
    /// `lhs_tensor_multi_index == [0, 1, 2]`, representing the multi-index of
    /// the component \f$L_{012}\f$. If
    /// `OperandTensorIndices == ti_c_t, ti_a_t`, this function will return the
    /// tensor multi-index representing the component \f$S_{20}\f$, which is
    /// `[2, 0]`.
    ///
    /// \tparam LhsTensorIndices the TensorIndexs of the outer product tensor
    /// \param lhs_tensor_multi_index the tensor multi-index of a component in
    /// the outer product tensor
    /// \return the tensor multi-index of an operand of the outer product
    template <typename... LhsTensorIndices>
    static SPECTRE_ALWAYS_INLINE constexpr std::array<
        size_t, sizeof...(OperandTensorIndices)>
    apply(
        const std::array<size_t, num_tensor_indices>& lhs_tensor_multi_index) {
      constexpr size_t operand_num_tensor_indices =
          sizeof...(OperandTensorIndices);
      constexpr std::array<size_t, sizeof...(LhsTensorIndices)>
          lhs_tensorindex_vals = {{LhsTensorIndices::value...}};
      constexpr std::array<size_t, operand_num_tensor_indices>
          operand_tensorindex_vals = {{OperandTensorIndices::value...}};

      // the operand component's tensor multi-index to compute
      std::array<size_t, operand_num_tensor_indices>
          operand_tensor_multi_index{};
      for (size_t i = 0; i < operand_num_tensor_indices; i++) {
        gsl::at(operand_tensor_multi_index, i) =
            gsl::at(lhs_tensor_multi_index,
                    static_cast<size_t>(std::distance(
                        lhs_tensorindex_vals.begin(),
                        alg::find(lhs_tensorindex_vals,
                                  gsl::at(operand_tensorindex_vals, i)))));
      }
      return operand_tensor_multi_index;
    }
  };

  /// \brief Return the value of the component of the outer product tensor at a
  /// given multi-index
  ///
  /// \details
  /// This function takes the multi-index of some component of the LHS outer
  /// product to compute. The function first computes the multi-indices of the
  /// pair of components in the two RHS operand expressions, then multiplies the
  /// values at these multi-indices to obtain the value of the LHS outer product
  /// component. For example, say we are evaluating
  /// \f$L_abc = R_{b} * S_{ca}\f$. Let `lhs_multi_index == {0, 1, 2}`, which
  /// refers to the component \f$L_{012}\f$, the component we wish to compute.
  /// This function will compute the multi-indices of the operands that
  /// correspond to \f$R_{1}\f$ and \f$S_{20}\f$, retrieve their values, and
  /// return their product.
  ///
  /// \tparam LhsIndices the TensorIndexs of the outer product tensor
  /// \param lhs_multi_index the multi-index of the component of the outer
  /// product tensor to retrieve
  /// \return the value of the component at `lhs_multi_index` in the outer
  /// product tensor
  template <typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& lhs_multi_index) const {
    const std::array<size_t, num_tensor_indices_first_operand>
        first_op_tensor_multi_index =
            GetOpTensorMultiIndex<ArgsList1<Args1...>>::template apply<
                LhsIndices...>(lhs_multi_index);
    const std::array<size_t, num_tensor_indices_second_operand>
        second_op_tensor_multi_index =
            GetOpTensorMultiIndex<ArgsList2<Args2...>>::template apply<
                LhsIndices...>(lhs_multi_index);

    return t1_.template get<Args1...>(first_op_tensor_multi_index) *
           t2_.template get<Args2...>(second_op_tensor_multi_index);
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

// @{
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
// @}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a tensor
/// expression and a `double`
///
/// \note The implementation instead uses the operation, `t * (1.0 / number)`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the quotient
/// \tparam X the type of data stored in the tensor expression operand of the
/// quotient
/// \tparam ArgsList the TensorIndexs of the tensor expression operand of the
/// quotient
/// \param t the tensor expression operand of the quotient
/// \param number the `double` operand of the quotient
/// \return the tensor expression representing the quotient of a tensor
/// expression and a `double`
template <typename T, typename X, typename ArgsList>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T, X, typename T::symmetry, typename T::index_list,
                           ArgsList>& t,
    const double number) {
  return t * TensorExpressions::NumberAsExpression(1.0 / number);
}
