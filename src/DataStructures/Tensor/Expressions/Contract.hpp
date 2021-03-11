// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/TMPL.hpp"

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
  // First and second \ref SpacetimeIndex "TensorIndexType"s to contract.
  // "first" and "second" here refer to the position of the indices to contract
  // in the list of indices, with "first" denoting leftmost
  //
  // e.g. `R(ti_A, ti_b, ti_a)` :
  // - `first_contracted_index` refers to the
  //   \ref SpacetimeIndex "TensorIndexType" refered to by `ti_A`
  // - `second_contracted_index` refers to the
  //   \ref SpacetimeIndex "TensorIndexType" refered to by `ti_a`
  using first_contracted_index = tmpl::at_c<IndexList, FirstContractedIndexPos>;
  using second_contracted_index =
      tmpl::at_c<IndexList, SecondContractedIndexPos>;
  static_assert(tmpl::size<Symm>::value > 1 and
                    tmpl::size<IndexList>::value > 1,
                "Cannot contract indices on a Tensor with rank less than 2");
  static_assert(detail::indices_contractible<first_contracted_index,
                                             second_contracted_index>::value,
                "Cannot contract the requested indices.");

  using new_type =
      typename detail::ContractedType<FirstContractedIndexPos,
                                      SecondContractedIndexPos, T, X, Symm,
                                      IndexList, ArgsList>::type;

  using type = X;
  using symmetry = typename new_type::symmetry;
  using index_list = typename new_type::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  static constexpr auto num_uncontracted_tensor_indices =
      tmpl::size<Symm>::value;
  using args_list = typename new_type::args_list;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}
  ~TensorContract() override = default;

  /// \brief Return the multi-index of the first uncontracted LHS component to
  /// be summed to compute a given contracted LHS component
  ///
  /// \details
  /// Returns the multi-index that results from taking the
  /// `contracted_lhs_multi_index` and inserting `0` at the two positions of the
  /// pair of indices to contract.
  ///
  /// \param contracted_lhs_multi_index the multi-index of a contracted LHS
  /// component to be computed
  /// \return the multi-index of the first uncontracted LHS component to
  /// be summed to compute the contracted LHS component at
  /// `contracted_lhs_multi_index`
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_first_uncontracted_lhs_multi_index_to_sum(
      const std::array<size_t, num_tensor_indices>&
          contracted_lhs_multi_index) noexcept {
    std::array<size_t, num_uncontracted_tensor_indices>
        first_uncontracted_lhs_multi_index{};

    for (size_t i = 0; i < FirstContractedIndexPos; i++) {
      gsl::at(first_uncontracted_lhs_multi_index, i) =
          gsl::at(contracted_lhs_multi_index, i);
    }
    first_uncontracted_lhs_multi_index[FirstContractedIndexPos] = 0;
    for (size_t i = FirstContractedIndexPos + 1; i < SecondContractedIndexPos;
         i++) {
      gsl::at(first_uncontracted_lhs_multi_index, i) =
          gsl::at(contracted_lhs_multi_index, i - 1);
    }
    first_uncontracted_lhs_multi_index[SecondContractedIndexPos] = 0;
    for (size_t i = SecondContractedIndexPos + 1;
         i < num_uncontracted_tensor_indices; i++) {
      gsl::at(first_uncontracted_lhs_multi_index, i) =
          gsl::at(contracted_lhs_multi_index, i - 2);
    }
    return first_uncontracted_lhs_multi_index;
  }

  // Inserts the first contracted TensorIndex into the list of contracted LHS
  // TensorIndexs
  template <typename... LhsIndices>
  using get_uncontracted_lhs_tensorindex_list_helper = tmpl::append<
      tmpl::front<tmpl::split_at<tmpl::list<LhsIndices...>,
                                 tmpl::size_t<FirstContractedIndexPos>>>,
      tmpl::list<tmpl::at_c<ArgsList, FirstContractedIndexPos>>,
      tmpl::back<tmpl::split_at<tmpl::list<LhsIndices...>,
                                tmpl::size_t<FirstContractedIndexPos>>>>;

  /// Constructs the uncontracted LHS's list of TensorIndexs by inserting the
  /// pair of indices being contracted into the list of contracted LHS
  /// TensorIndexs
  ///
  /// Example: Let `ti_a_t` denote the type of `ti_a`, and apply the same
  /// convention for other generic indices. If we contract RHS tensor
  /// \f$R^{a}{}_{bac}\f$ to LHS tensor \f$L_{cb}\f$, the RHS list of generic
  /// indices (`ArgsList`) is `tmpl::list<ti_A_t, ti_b_t, ti_a_t, ti_c_t>` and
  /// the LHS generic indices (`LhsIndices`) are `ti_c_t, ti_b_t`. `ti_A_t` and
  /// `ti_a_t` are inserted into `LhsIndices` at their positions from the RHS,
  /// which yields: `tmpl::list<ti_A_t, ti_c_t, ti_a_t, ti_b_t>`.
  template <typename... LhsIndices>
  using get_uncontracted_lhs_tensorindex_list = tmpl::append<
      tmpl::front<tmpl::split_at<
          get_uncontracted_lhs_tensorindex_list_helper<LhsIndices...>,
          tmpl::size_t<SecondContractedIndexPos>>>,
      tmpl::list<tmpl::at_c<ArgsList, SecondContractedIndexPos>>,
      tmpl::back<tmpl::split_at<
          get_uncontracted_lhs_tensorindex_list_helper<LhsIndices...>,
          tmpl::size_t<SecondContractedIndexPos>>>>;

  /// \brief Helper struct for computing the contraction of one pair of
  /// indices
  ///
  /// \tparam UncontractedLhsTensorIndexList the typelist of TensorIndexs of
  /// the uncontracted LHS tensor
  template <typename UncontractedLhsTensorIndexList>
  struct ComputeContraction;

  template <typename... UncontractedLhsTensorIndices>
  struct ComputeContraction<tmpl::list<UncontractedLhsTensorIndices...>> {
    /// \brief Computes the value of a component in the contracted LHS tensor
    ///
    /// \details
    /// Returns the value of the component in the contracted LHS tensor whose
    /// multi-index is that which results from removing the contracted indices
    /// from `uncontracted_lhs_multi_index_to_fill`. For example, if
    /// `uncontracted_lhs_multi_index_to_fill == {0, 1, 0, 3}` and the first and
    /// third positions are the contracted index positions, this function
    /// computes the contracted LHS component at multi-index `{1, 3}`.
    ///
    /// To compute a contracted component correctly, the external call to this
    /// function must be done with `ContractedIndexValue == 0`.
    ///
    /// \tparam ContractedIndexValue the concrete value inserted for the indices
    /// to contract
    /// \param t the expression contained within this RHS contraction expression
    /// \param uncontracted_lhs_multi_index_to_fill the multi-index of an
    /// uncontracted LHS tensor component to sum for contraction
    /// \return the value of a component of the contracted LHS tensor
    template <size_t ContractedIndexValue>
    static SPECTRE_ALWAYS_INLINE decltype(auto) apply(
        const T& t, std::array<size_t, num_uncontracted_tensor_indices>
                        uncontracted_lhs_multi_index_to_fill) noexcept {
      // Fill contracted indices in multi-index with `ContractedIndexValue`
      uncontracted_lhs_multi_index_to_fill[FirstContractedIndexPos] =
          ContractedIndexValue;
      uncontracted_lhs_multi_index_to_fill[SecondContractedIndexPos] =
          ContractedIndexValue;

      if constexpr (ContractedIndexValue < first_contracted_index::dim - 1) {
        // We have more than one component left to sum
        return t.template get<UncontractedLhsTensorIndices...>(
                   uncontracted_lhs_multi_index_to_fill) +
               apply<ContractedIndexValue + 1>(
                   t, uncontracted_lhs_multi_index_to_fill);
      } else {
        // We only have one final component to sum
        return t.template get<UncontractedLhsTensorIndices...>(
            uncontracted_lhs_multi_index_to_fill);
      }
    }
  };

  /// \brief Return the value of the component of the contracted LHS tensor at a
  /// given multi-index
  ///
  /// \details
  /// Given a RHS tensor to be contracted, the uncontracted LHS represents the
  /// uncontracted RHS tensor arranged with the LHS's generic index order. The
  /// contracted LHS represents the result of contracting this uncontracted
  /// LHS. For example, if we have RHS tensor \f$R^{a}{}_{abc}\f$ and we want to
  /// contract it to the LHS tensor \f$L_{cb}\f$, then \f$L_{cb}\f$ represents
  /// the contracted LHS, while \f$L^{a}{}_{acb}\f$ represents the uncontracted
  /// LHS. Note that the relative ordering of the LHS generic indices \f$c\f$
  /// and \f$b\f$ in the contracted LHS is preserved in the uncontracted LHS.
  ///
  /// To compute a contraction, we need to get all the uncontracted LHS
  /// components to sum. In the example above, this means that in order to
  /// compute \f$L_{cb}\f$ for some \f$c\f$ and \f$b\f$, we need to sum the
  /// components \f$L^{a}{}_{acb}\f$ for all values of \f$a\f$. This function
  /// first constructs the list of generic indices (TensorIndexs) of the
  /// uncontracted LHS, then uses helper functions to compute and return the
  /// contracted LHS component by summing the necessary uncontracted LHS
  /// components.
  ///
  /// \tparam ContractedLhsIndices the TensorIndexs of the contracted LHS tensor
  /// \param contracted_lhs_multi_index the multi-index of the contracted LHS
  /// tensor component to retrieve
  /// \return the value of the component at `contracted_lhs_multi_index` in the
  /// contracted LHS tensor
  template <typename... ContractedLhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& contracted_lhs_multi_index)
      const {
    using uncontracted_lhs_tensorindex_list =
        get_uncontracted_lhs_tensorindex_list<ContractedLhsIndices...>;
    return ComputeContraction<uncontracted_lhs_tensorindex_list>::
        template apply<0>(t_, get_first_uncontracted_lhs_multi_index_to_sum(
                                  contracted_lhs_multi_index));
  }

 private:
  T t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Returns the positions of the first indices to contract in an
 * expression
 *
 * \details Given a list of values that represent an expression's generic index
 * encodings, this function looks to see if it can find a pair of values that
 * encode one generic index and the generic index with opposite valence, such as
 * `ti_A` and `ti_a`. This denotes a pair of indices that will need to be
 * contracted. If there exists more than one such pair of indices in the
 * expression, the first pair of values found will be returned.
 *
 * For example, if we have tensor \f$R^{ab}{}_{ab}\f$ represented by the tensor
 * expression, `R(ti_A, ti_B, ti_a, ti_b)`, then this will return the positions
 * of the pair of values encoding `ti_A` and `ti_a`, which would be (0, 2)
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
 * For example, if we have tensor \f$R^{ab}{}_{ab}\f$ represented by the tensor
 * expression, `R(ti_A, ti_B, ti_a, ti_b)`, then one contraction expression is
 * created to represent contracting \f$R^{ab}{}_ab\f$ to \f$R^b{}_b\f$, and a
 * second to represent contracting \f$R^b{}_b\f$ to the scalar, \f$R\f$.
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
