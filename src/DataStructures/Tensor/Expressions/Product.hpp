// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor products

#pragma once

#include "Data/Tensor/Expressions/TensorExpression.hpp"

namespace TensorExpressions {

/*!
 * \ingroup TensorExpressionsGroup
 *
 * @tparam T1
 * @tparam T2
 * @tparam ArgsList1
 * @tparam ArgsList2
 */
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2>
struct Product;

template <typename T1, typename T2, template <typename...> class ArgsList1,
          template <typename...> class ArgsList2, typename... Args1,
          typename... Args2>
struct Product<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>>
    : public TensorExpression<
          Product<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>>,
          typename T1::type, double,
          tmpl::append<typename T1::index_list, typename T2::index_list>,
          tmpl::sort<
              tmpl::append<typename T1::args_list, typename T2::args_list>>>,
      public Expression {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value,
                "Cannot product Tensors holding different data types.");
  using max_symm2 = tmpl::fold<typename T2::symmetry, tmpl::uint32_t<0>,
                               tmpl::max<tmpl::_state, tmpl::_element>>;

  using type = typename T1::type;
  using symmetry = tmpl::append<
      tmpl::transform<typename T1::symmetry, tmpl::plus<tmpl::_1, max_symm2>>,
      typename T2::symmetry>;
  using index_list =
      tmpl::append<typename T1::index_list, typename T2::index_list>;
  static constexpr auto num_tensor_indices =
      tmpl::size<index_list>::value == 0 ? 1 : tmpl::size<index_list>::value;
  using args_list =
      tmpl::sort<tmpl::append<typename T1::args_list, typename T2::args_list>>;

  Product(const T1& t1, const T2& t2) : t1_(t1), t2_(t2) {}

  // TODO: The args will need to be reduced in a careful manner, which means
  // they need to be reduced together, then split at the correct length so that
  // the indexing is correct.
  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<U, num_tensor_indices>& tensor_index) const {
    return t1_.template get<LhsIndices...>(tensor_index) *
           t2_.template get<LhsIndices...>(tensor_index);
  }

 private:
  const T1 t1_;
  const T2 t2_;
};

}  // namespace TensorExpressions

/*!
 * @ingroup TensorExpressionsGroup
 *
 * @tparam T1
 * @tparam T2
 * @tparam X
 * @tparam Symm1
 * @tparam Symm2
 * @tparam IndexList1
 * @tparam IndexList2
 * @tparam Args1
 * @tparam Args2
 * @param t1
 * @param t2
 * @return
 */
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T1, X, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, Args2>& t2) {
  // static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
  //               "Tensor addition is only possible with the same rank
  //               tensors");
  // static_assert(tmpl::equal_members<Args1, Args2>::value,
  //               "The indices when adding two tensors must be equal. This
  //               error "
  //               "occurs from expressions like A(_a, _b) + B(_c, _a)");
  return TensorExpressions::Product<
      typename std::conditional<
          std::is_base_of<Expression, T1>::value, T1,
          TensorExpression<T1, X, Symm1, IndexList1, Args1>>::type,
      typename std::conditional<
          std::is_base_of<Expression, T2>::value, T2,
          TensorExpression<T2, X, Symm2, IndexList2, Args2>>::type,
      Args1, Args2>(~t1, ~t2);
}
