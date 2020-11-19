// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Requires.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * Holds all possible TensorExpressions currently implemented
 */
namespace TensorExpressions {

namespace detail {

template <typename I1, typename I2>
using indices_contractible = std::integral_constant<
    bool, I1::dim == I2::dim and I1::ul != I2::ul and
              std::is_same_v<typename I1::Frame, typename I2::Frame> and
              I1::index_type == I2::index_type>;

template <typename T, typename X, typename SymmList, typename IndexList,
          typename TensorIndexList>
struct ContractedTypeImpl;

template <typename T, typename X, template <typename...> class SymmList,
          typename IndexList, typename TensorIndexList, typename... Symm>
struct ContractedTypeImpl<T, X, SymmList<Symm...>, IndexList, TensorIndexList> {
  using type = TensorExpression<T, X, Symmetry<Symm::value...>, IndexList,
                                TensorIndexList>;
};

template <size_t FirstContractedIndexPos, size_t SecondContractedIndexPos,
          typename T, typename X, typename Symm, typename IndexList,
          typename TensorIndexList>
struct ContractedType {
  static_assert(FirstContractedIndexPos < SecondContractedIndexPos,
                "The position of the first provided index to contract must be "
                "less than the position of the second index to contract.");
  using contracted_symmetry =
      tmpl::erase<tmpl::erase<Symm, tmpl::size_t<SecondContractedIndexPos>>,
                  tmpl::size_t<FirstContractedIndexPos>>;
  using contracted_index_list = tmpl::erase<
      tmpl::erase<IndexList, tmpl::size_t<SecondContractedIndexPos>>,
      tmpl::size_t<FirstContractedIndexPos>>;
  using contracted_tensorindex_list = tmpl::erase<
      tmpl::erase<TensorIndexList, tmpl::size_t<SecondContractedIndexPos>>,
      tmpl::size_t<FirstContractedIndexPos>>;
  using type = typename ContractedTypeImpl<T, X, contracted_symmetry,
                                           contracted_index_list,
                                           contracted_tensorindex_list>::type;
};

template <size_t I, size_t Index1, size_t Index2, typename... LhsIndices,
          typename T, typename S>
static SPECTRE_ALWAYS_INLINE decltype(auto) compute_contraction(S tensor_index,
                                                                const T& t1) {
  if constexpr (I == 0) {
    tensor_index[Index1] = 0;
    tensor_index[Index2] = 0;
    return t1.template get<LhsIndices...>(tensor_index);
  } else {
    tensor_index[Index1] = I;
    tensor_index[Index2] = I;
    return t1.template get<LhsIndices...>(tensor_index) +
           compute_contraction<I - 1, Index1, Index2, LhsIndices...>(
               tensor_index, t1);
  }
}
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <size_t FirstContractedIndexPos, size_t SecondContractedIndexPos,
          typename T, typename X, typename Symm, typename IndexList,
          typename ArgsList>
struct TensorContract
    : public TensorExpression<
          TensorContract<FirstContractedIndexPos, SecondContractedIndexPos, T,
                         X, Symm, IndexList, ArgsList>,
          X,
          typename detail::ContractedType<FirstContractedIndexPos,
                                          SecondContractedIndexPos, T, X, Symm,
                                          IndexList, ArgsList>::type::symmetry,
          typename detail::ContractedType<
              FirstContractedIndexPos, SecondContractedIndexPos, T, X, Symm,
              IndexList, ArgsList>::type::index_list,
          typename detail::ContractedType<
              FirstContractedIndexPos, SecondContractedIndexPos, T, X, Symm,
              IndexList, ArgsList>::type::args_list> {
  using CI1 = tmpl::at_c<IndexList, FirstContractedIndexPos>;
  using CI2 = tmpl::at_c<IndexList, SecondContractedIndexPos>;
  static_assert(tmpl::size<Symm>::value > 1 and
                    tmpl::size<IndexList>::value > 1,
                "Cannot contract indices on a Tensor with rank less than 2");
  static_assert(detail::indices_contractible<CI1, CI2>::value,
                "Cannot contract the requested indices.");

  using new_type =
      typename detail::ContractedType<FirstContractedIndexPos,
                                      SecondContractedIndexPos, T, X, Symm,
                                      IndexList, ArgsList>::type;

  using type = X;
  using symmetry = typename new_type::symmetry;
  using index_list = typename new_type::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = typename new_type::args_list;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}

  template <size_t I, size_t Rank>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<size_t, Rank>& tensor_index_in,
      const std::array<size_t, num_tensor_indices>& tensor_index) const {
    if constexpr (I < FirstContractedIndexPos) {
      tensor_index_in[I] = tensor_index[I];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I == FirstContractedIndexPos) {
      // 10000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] = 10000;
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > FirstContractedIndexPos and
                         I <= SecondContractedIndexPos and I < Rank - 1) {
      // tensor_index is Rank - 2 since it shouldn't be called for Rank 2 case
      // 20000 is for the slot that will be set later. Easy to debug.
      tensor_index_in[I] =
          I == SecondContractedIndexPos ? 20000 : tensor_index[I - 1];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I > SecondContractedIndexPos and I < Rank - 1) {
      // Left as Rank - 2 since it should never be called for the Rank 2 case
      tensor_index_in[I] = tensor_index[I - 2];
      fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
    } else if constexpr (I == SecondContractedIndexPos) {
      tensor_index_in[I] = 20000;
    } else {
      tensor_index_in[I] = tensor_index[I - 2];
    }
  }

  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<U, num_tensor_indices>& new_tensor_index) const {
    // new_tensor_index is the one with _fewer_ components, ie post-contraction
    std::array<size_t, tmpl::size<Symm>::value> tensor_index{};
    // Manually unrolled for loops to compute the tensor_index from the
    // new_tensor_index
    fill_contracting_tensor_index<0>(tensor_index, new_tensor_index);
    return detail::compute_contraction<CI1::dim - 1, FirstContractedIndexPos,
                                       SecondContractedIndexPos, LhsIndices...>(
        tensor_index, t_);
  }

  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    const std::array<size_t, num_tensor_indices>& new_tensor_index =
        LhsStructure::template get_canonical_tensor_index<num_tensor_indices>(
            lhs_storage_index);
    return get<LhsIndices...>(new_tensor_index);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Returns the positions of the first indices to contract in an
 * expression
 *
 * \details Given a list of values that represent an expression's generic index
 * encodings, this function looks to see if it can find a pair of values that
 * encode one generic index and the generic index with opposite valence, such as
 * `ti_A_t` and `ti_a_t`. This denotes a pair of indices that will need to be
 * contracted. If there exists more than one such pair of indices in the
 * expression, the first pair of values found will be returned.
 *
 * For example, if we have tensor \f${R^{ab}}_{ab}\f$ represented by the tensor
 * expression, `R(ti_A, ti_B, ti_a, ti_b)`, then this will return the positions
 * of the pair of values encoding `ti_A_t` and `ti_a_t`, which would be (0, 2)
 *
 * @param tensorindex_values the TensorIndex values of a tensor expression
 * @return the positions of the first pair of TensorIndex values to contract
 */
template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE static constexpr std::pair<size_t, size_t>
get_first_index_positions_to_contract(
    const std::array<size_t, NumIndices>& tensorindex_values) noexcept {
  for (size_t i = 0; i < tensorindex_values.size(); ++i) {
    const size_t current_value = gsl::at(tensorindex_values, i);
    const size_t opposite_value_to_find =
        get_tensorindex_value_with_opposite_valence(current_value);
    for (size_t j = i + 1; j < tensorindex_values.size(); ++j) {
      if (opposite_value_to_find == gsl::at(tensorindex_values, j)) {
        // We found both the lower and upper version of a generic index in the
        // list of generic indices, so we return this pair's positions
        return std::pair{i, j};
      }
    }
  }
  // We couldn't find a single pair of indices that needs to be contracted
  return std::pair{std::numeric_limits<size_t>::max(),
                   std::numeric_limits<size_t>::max()};
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Creates a contraction expression from a tensor expression if there are
 * any indices to contract
 *
 * \details If there are no indices to contract, the input TensorExpression is
 * simply returned. Otherwise, a contraction expression is created for
 * contracting one pair of upper and lower indices. If there is more than one
 * pair of indices to contract, subsequent contraction expressions are
 * recursively created, nesting one contraction expression inside another.
 *
 * For example, if we have tensor \f${R^{ab}}_{ab}\f$ represented by the tensor
 * expression, `R(ti_A, ti_B, ti_a, ti_b)`, then one contraction expression is
 * created to represent contracting \f${R^{ab}}_ab\f$ to \f${R^b}_b\f$, and a
 * second to represent contracting \f${R^b}_b\f$ to the scalar, \f${R}\f$.
 *
 * @param t the TensorExpression to potentially contract
 * @return the input tensor expression or a contraction expression of the input
 */
template <typename T, typename X, typename Symm, typename IndexList,
          typename... TensorIndices>
SPECTRE_ALWAYS_INLINE static constexpr auto contract(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<TensorIndices...>>&
        t) noexcept {
  constexpr std::array<size_t, sizeof...(TensorIndices)> tensorindex_values = {
      {TensorIndices::value...}};
  constexpr std::pair first_index_positions_to_contract =
      get_first_index_positions_to_contract(tensorindex_values);
  constexpr std::pair no_indices_to_contract_sentinel{
      std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};

  if constexpr (first_index_positions_to_contract ==
                no_indices_to_contract_sentinel) {
    // There aren't any indices to contract, so we just return the input
    return ~t;
  } else {
    // We have a pair of indices to be contract
    return contract(
        TensorContract<first_index_positions_to_contract.first,
                       first_index_positions_to_contract.second, T, X, Symm,
                       IndexList, tmpl::list<TensorIndices...>>{t});
  }
}
}  // namespace TensorExpressions
