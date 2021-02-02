// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Requires.hpp"
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
  using structure = Tensor_detail::Structure<symmetry, index_list>;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}
  ~TensorContract() override = default;

  /// \brief Return the tensor multi-index of one uncontracted LHS component to
  /// be summed to compute a contracted LHS component
  ///
  /// \details
  /// Example: If we have RHS tensor \f$R^{a}{}_{abc}\f$ and we want to contract
  /// it to the LHS tensor \f$L_{cb}\f$, then \f$L_{cb}\f$ represents the
  /// contracted LHS, while \f$L^{a}{}_{acb}\f$ represents the uncontracted
  /// LHS. This function takes a concrete contracted LHS multi-index as input,
  /// representing the  multi-index of a component of the contracted LHS that we
  /// wish to compute. If `lhs_contracted_multi_index == [1, 2]`, this
  /// represents \f$L_{12}\f$, the contracted LHS component we wish to compute.
  /// In this case, we will need to sum \f$L^{a}{}_{a12}\f$ for all values of
  /// \f$a\f$. `contracted_index_value` represents one such concrete value that
  /// is filled in for \f$a\f$. In continuing the example, if
  /// `contracted_index_value == 3`, then this function returns the multi-index
  /// that represents \f$L^{3}{}_{312}\f$, which is `[3, 3, 1, 2]`. In this way,
  /// what is constructed and returned is one such concrete multi-index of the
  /// uncontracted LHS tensor to be summed as part of computing a component of
  /// the contracted LHS tensor.
  ///
  /// \param lhs_contracted_multi_index the tensor multi-index of a contracted
  /// LHS component to be computed
  /// \param contracted_index_value the concrete value inserted for the indices
  /// to contract
  /// \return the tensor multi-index of one uncontracted LHS component to be
  /// summed for computing the contracted LHS component at
  /// `lhs_contracted_multi_index`
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_tensor_index_to_sum(
      const std::array<size_t, num_tensor_indices>& lhs_contracted_multi_index,
      const size_t contracted_index_value) noexcept {
    std::array<size_t, num_uncontracted_tensor_indices>
        contracting_tensor_index{};

    for (size_t i = 0; i < FirstContractedIndexPos; i++) {
      gsl::at(contracting_tensor_index, i) =
          gsl::at(lhs_contracted_multi_index, i);
    }
    contracting_tensor_index[FirstContractedIndexPos] = contracted_index_value;
    for (size_t i = FirstContractedIndexPos + 1; i < SecondContractedIndexPos;
         i++) {
      gsl::at(contracting_tensor_index, i) =
          gsl::at(lhs_contracted_multi_index, i - 1);
    }
    contracting_tensor_index[SecondContractedIndexPos] = contracted_index_value;
    for (size_t i = SecondContractedIndexPos + 1;
         i < num_uncontracted_tensor_indices; i++) {
      gsl::at(contracting_tensor_index, i) =
          gsl::at(lhs_contracted_multi_index, i - 2);
    }
    return contracting_tensor_index;
  }

  /// \brief Return the storage indices of the uncontracted LHS components to
  /// be summed to compute a contracted LHS component
  ///
  /// \details
  /// Example: If we have RHS tensor \f$R^{a}{}_{abc}\f$ and we want to contract
  /// it to the LHS tensor \f$L_{cb}\f$, then \f$L_{cb}\f$ represents the
  /// contracted LHS, while \f$L^{a}{}_{acb}\f$ represents the uncontracted
  /// LHS. `I` represents the storage index of the component \f$L_{cb}\f$ for
  /// some \f$c\f$ and \f$b\f$, an uncontracted LHS component that we wish to
  /// compute. If `c == 1` and `b == 2`, then this function computes and returns
  /// the list of storage indices of components \f$L^{a}{}_{a12}\f$ for all
  /// values of \f$a\f$, i.e. the components to sum to compute the component
  /// \f$L_{12}\f$.
  ///
  /// \tparam I the storage index of a contracted LHS component to be computed
  /// \tparam UncontractedLhsStructure the Structure of the uncontracted LHS
  /// tensor
  /// \tparam ContractedLhsStructure the Structure of the contracted LHS tensor
  /// \tparam Ints a sequence of integers from [0, dimension of contracted
  /// indices)
  /// \return the storage indices of the uncontracted LHS components to be
  /// summed to compute a contracted LHS component
  template <size_t I, typename UncontractedLhsStructure,
            typename ContractedLhsStructure, size_t... Ints>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
                                                    first_contracted_index::dim>
  get_storage_indices_to_sum(
      const std::index_sequence<Ints...>& /*dim_seq*/) noexcept {
    constexpr std::array<size_t, num_tensor_indices>
        lhs_contracted_multi_index =
            ContractedLhsStructure::get_canonical_tensor_index(I);
    return {{UncontractedLhsStructure::get_storage_index(
        get_tensor_index_to_sum(lhs_contracted_multi_index, Ints))...}};
  }

  /// \brief Computes a mapping between the storage indices of the contracted
  /// LHS components and the uncontracted LHS components to sum for a
  /// contraction
  ///
  /// \details
  /// Example: If we have RHS tensor \f$R^{a}{}_{abc}\f$ and we want to contract
  /// it to the LHS tensor \f$L_{cb}\f$, then \f$L_{cb}\f$ represents the
  /// contracted LHS, while \f$L^{a}{}_{acb}\f$ represents the uncontracted
  /// LHS. This function computes and returns a mapping between the storage
  /// indices of (1) each component of \f$L_{cb}\f$ and (2) the corresponding
  /// lists of components of \f$L^{a}{}_{acb}\f$ to sum to compute each
  /// component of \f$L_{cb}\f$.
  ///
  /// \tparam ContractedLhsNumComponents the number of components in the
  /// contracted LHS tensor
  /// \tparam UncontractedLhsStructure the Structure of the uncontracted LHS
  /// tensor
  /// \tparam ContractedLhsStructure the Structure of the contracted LHS tensor
  /// \tparam Ints a sequence of integers from [0, `ContractedLhsNumComponents`)
  /// \return a mapping between the storage indices of the contracted LHS
  /// components and the uncontracted LHS components to sum for a contraction
  template <size_t ContractedLhsNumComponents,
            typename UncontractedLhsStructure, typename ContractedLhsStructure,
            size_t... Ints>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      std::array<size_t, first_contracted_index::dim>,
      ContractedLhsNumComponents>
  get_map_of_components_to_sum(
      const std::index_sequence<Ints...>& /*index_seq*/) noexcept {
    constexpr std::make_index_sequence<first_contracted_index::dim> dim_seq{};
    return {{get_storage_indices_to_sum<Ints, UncontractedLhsStructure,
                                        ContractedLhsStructure>(dim_seq)...}};
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

  /// \brief Helper struct for computing the contraction of one pair of indices
  ///
  /// \tparam UncontractedLhsTensorIndexList the typelist of TensorIndexs of the
  /// uncontracted LHS tensor
  template <typename UncontractedLhsTensorIndexList>
  struct ComputeContraction;

  template <typename... UncontractedLhsTensorIndices>
  struct ComputeContraction<tmpl::list<UncontractedLhsTensorIndices...>> {
    /// \brief Computes the value of the component in the contracted LHS tensor
    /// at a given storage index
    ///
    /// \details
    /// This function recursively computes the value of the component in the
    /// contracted LHS tensor at a given storage index by iterating over the
    /// list of storage indices of uncontracted LHS components to sum. This list
    /// is stored at `map_of_components_to_sum[lhs_storage_index]`, and the
    /// current component being summed is at
    /// `map_of_components_to_sum[lhs_storage_index][Index]`.
    ///
    /// \tparam UncontractedLhsStructure the Structure of the uncontracted LHS
    /// tensor
    /// \tparam ContractedLhsNumComponents the number of components in the
    /// contracted LHS tensor
    /// \tparam Index for a given list of uncontracted LHS storage indices whose
    /// components are summed to compute a contracted LHS component, this is the
    /// position of one such storage index in that list
    /// \param map_of_components_to_sum a mapping between the storage indices of
    /// the contracted LHS components and the uncontracted LHS components to sum
    /// to compute the former
    /// \param t1 the expression contained within the RHS contraction expression
    /// \param lhs_storage_index the storage index of the LHS tensor component
    /// to compute
    /// \return the computed value of the component at `lhs_storage_index` in
    /// the contracted LHS tensor
    template <typename UncontractedLhsStructure,
              size_t ContractedLhsNumComponents, size_t Index, typename T1>
    static SPECTRE_ALWAYS_INLINE decltype(auto) apply(
        const std::array<std::array<size_t, first_contracted_index::dim>,
                         ContractedLhsNumComponents>& map_of_components_to_sum,
        const T1& t1, const size_t& lhs_storage_index) noexcept {
      if constexpr (Index < first_contracted_index::dim - 1) {
        // We have more than one component left to sum
        return apply<UncontractedLhsStructure, ContractedLhsNumComponents,
                     Index + 1>(map_of_components_to_sum, t1,
                                lhs_storage_index) +
               t1.template get<UncontractedLhsStructure,
                               UncontractedLhsTensorIndices...>(
                   gsl::at(gsl::at(map_of_components_to_sum, lhs_storage_index),
                           Index));
      } else {
        // We only have one final component to sum
        return t1.template get<UncontractedLhsStructure,
                               UncontractedLhsTensorIndices...>(
            gsl::at(gsl::at(map_of_components_to_sum, lhs_storage_index),
                    first_contracted_index::dim - 1));
      }
    }
  };

  /// \brief Return the value of the component of the contracted LHS tensor at a
  /// given storage index
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
  /// uncontracted LHS, then uses a series of helper functions to compute a
  /// mapping from (1) the storage indices of the components in the contracted
  /// LHS tensor to (2) their corresponding lists of storage indices of
  /// components in the uncontracted LHS tensor that need to be summed to
  /// compute each contracted LHS component. Finally, the `ComputeContraction`
  /// helper struct is used to compute the contracted component at
  /// `lhs_storage_index` by leveraging this precomputed map's lists of indices
  /// to sum for each contracted LHS component's storage index.
  ///
  /// \tparam LhsStructure the Structure of the contracted LHS tensor
  /// \tparam LhsIndices the TensorIndexs of the contracted LHS tensor
  /// \param lhs_storage_index the storage index of the contracted LHS tensor
  /// component to retrieve
  /// \return the value of the component at `lhs_storage_index` in the
  /// contracted LHS tensor
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const {
    constexpr size_t contracted_lhs_num_components = LhsStructure::size();
    using uncontracted_lhs_tensorindex_list =
        get_uncontracted_lhs_tensorindex_list<LhsIndices...>;
    using uncontracted_lhs_structure =
        typename LhsTensorSymmAndIndices<ArgsList,
                                         uncontracted_lhs_tensorindex_list,
                                         Symm, IndexList>::structure;

    constexpr std::make_index_sequence<contracted_lhs_num_components> map_seq{};

    // A map from contracted LHS storage indices to lists of uncontracted LHS
    // storage indices of components to sum for contraction
    constexpr std::array<std::array<size_t, first_contracted_index::dim>,
                         contracted_lhs_num_components>
        map_of_components_to_sum =
            get_map_of_components_to_sum<contracted_lhs_num_components,
                                         uncontracted_lhs_structure,
                                         LhsStructure>(map_seq);

    // This returns the value of the component stored at `lhs_storage_index` in
    // the contracted LHS tensor
    return ComputeContraction<uncontracted_lhs_tensorindex_list>::
        template apply<uncontracted_lhs_structure,
                       contracted_lhs_num_components, 0>(
            map_of_components_to_sum, t_, lhs_storage_index);
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
