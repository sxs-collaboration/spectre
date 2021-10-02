// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * Holds all possible TensorExpressions currently implemented
 */
namespace TensorExpressions {

namespace detail {

template <typename I1, typename I2>
using indices_contractible = std::bool_constant<
    I1::ul != I2::ul and
    std::is_same_v<typename I1::Frame, typename I2::Frame> and
    ((I1::index_type == I2::index_type and I1::dim == I2::dim) or
     // If one index is spacetime and the other is spatial, the indices can
     // be contracted if they have the same number of spatial dimensions
     (I1::index_type == IndexType::Spacetime and I1::dim == I2::dim + 1) or
     (I2::index_type == IndexType::Spacetime and I1::dim + 1 == I2::dim))>;

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

  /// \brief Return a partially-filled multi-index of the uncontracted
  /// expression where only the values of the indices that are not contracted
  /// have been filled in
  ///
  /// \details
  /// Returns the multi-index that results from taking the
  /// `contracted_multi_index` and inserting the maximum `size_t` value at the
  /// two positions of the pair of indices to contract.
  ///
  /// e.g. `R(ti_A, ti_a, ti_b, ti_c)` represents contracting some
  /// \f$R^{a}{}_{abc}\f$ to \f$R_{bc}\f$. If `contracted_multi_index` is
  /// `[5, 4]` (i.e. `b == 5`, `c == 4`), the returned
  /// `uncontracted_multi_index` is
  /// `[<max size_t value>, <max size_t value>, 5, 4]`.
  ///
  /// \param contracted_multi_index the multi-index of a component of the
  /// contracted expression
  /// \return the multi-index of the uncontracted expression where only the
  /// values of the indices that are not contracted have been filled in
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_uncontracted_multi_index_with_uncontracted_values(
      const std::array<size_t, num_tensor_indices>& contracted_multi_index) {
    std::array<size_t, num_uncontracted_tensor_indices>
        uncontracted_multi_index{};

    for (size_t i = 0; i < FirstContractedIndexPos; i++) {
      gsl::at(uncontracted_multi_index, i) = gsl::at(contracted_multi_index, i);
    }
    uncontracted_multi_index[FirstContractedIndexPos] =
        std::numeric_limits<size_t>::max();  // placeholder to later be replaced
    for (size_t i = FirstContractedIndexPos + 1; i < SecondContractedIndexPos;
         i++) {
      gsl::at(uncontracted_multi_index, i) =
          gsl::at(contracted_multi_index, i - 1);
    }
    uncontracted_multi_index[SecondContractedIndexPos] =
        std::numeric_limits<size_t>::max();  // placeholder to later be replaced
    for (size_t i = SecondContractedIndexPos + 1;
         i < num_uncontracted_tensor_indices; i++) {
      gsl::at(uncontracted_multi_index, i) =
          gsl::at(contracted_multi_index, i - 2);
    }
    return uncontracted_multi_index;
  }

  /// \brief Computes the value of a component in the resultant contracted
  /// tensor
  ///
  /// \details
  /// Returns the value of the component in the resultant contracted tensor
  /// whose multi-index is that which results from removing the contracted
  /// indices from `uncontracted_multi_index_to_fill`. For example, if
  /// `uncontracted_multi_index_to_fill == {0, 1, 0, 3}` and the first and
  /// third positions are the contracted index positions, this function
  /// computes the contracted component at multi-index `{1, 3}`.
  ///
  /// The distinction between `FirstContractedIndexValue` and
  /// `SecondContractedIndexValue` is necessary to properly compute the
  /// contraction of special cases where the "starting" index values to insert
  /// at the contracted index positions are not equal. One such case:
  /// `R(ti_J, ti_j)`, where R is a rank 2 tensor whose first index is spatial
  /// and whose second index is spacetime. The external call to this function
  /// would require `FirstContractedIndexValue == 0` and
  /// `SecondContractedIndexValue == 1` to ensure the correct "starting" index
  /// values are inserted into `uncontracted_multi_index_to_fill` at both index
  /// positions, respectively.
  ///
  /// \tparam FirstContractedIndexValue the concrete value inserted for the
  /// first index to contract
  /// \tparam SecondContractedIndexValue the concrete value inserted for the
  /// second index to contract
  /// \param t the expression contained within this contraction expression
  /// \param uncontracted_multi_index_to_fill the multi-index of the
  /// uncontracted tensor component to fill and sum for contraction
  /// \return the value of a component of the resulant contracted tensor
  template <size_t FirstContractedIndexValue, size_t SecondContractedIndexValue>
  static SPECTRE_ALWAYS_INLINE decltype(auto) compute_contraction(
      const T& t, std::array<size_t, num_uncontracted_tensor_indices>
                      uncontracted_multi_index_to_fill) {
    // Fill contracted indices in multi-index with `FirstContractedIndexValue`
    // and `SecondContractedIndexValue`
    uncontracted_multi_index_to_fill[FirstContractedIndexPos] =
        FirstContractedIndexValue;
    uncontracted_multi_index_to_fill[SecondContractedIndexPos] =
        SecondContractedIndexValue;

    if constexpr (FirstContractedIndexValue < first_contracted_index::dim - 1) {
      // We have more than one component left to sum
      return t.get(uncontracted_multi_index_to_fill) +
             compute_contraction<FirstContractedIndexValue + 1,
                                 SecondContractedIndexValue + 1>(
                 t, uncontracted_multi_index_to_fill);
    } else {
      // We only have one final component to sum
      return t.get(uncontracted_multi_index_to_fill);
    }
  }

  /// \brief Return the value of the component of the resultant contracted
  /// tensor at a given multi-index
  ///
  /// \param contracted_multi_index the multi-index of the resultant contracted
  /// tensor component to retrieve
  /// \return the value of the component at `contracted_multi_index` in the
  /// resultant contracted tensor
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& contracted_multi_index)
      const {
    constexpr size_t initial_first_contracted_index_value =
        first_contracted_index::index_type == IndexType::Spacetime and
                not tmpl::at_c<ArgsList, FirstContractedIndexPos>::is_spacetime
            ? 1
            : 0;
    constexpr size_t initial_second_contracted_index_value =
        second_contracted_index::index_type == IndexType::Spacetime and
                not tmpl::at_c<ArgsList, SecondContractedIndexPos>::is_spacetime
            ? 1
            : 0;

    return compute_contraction<initial_first_contracted_index_value,
                               initial_second_contracted_index_value>(
        t_, get_uncontracted_multi_index_with_uncontracted_values(
                contracted_multi_index));
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
    const std::array<size_t, NumIndices>& tensorindex_values) {
  for (size_t i = 0; i < tensorindex_values.size(); ++i) {
    const size_t current_value = gsl::at(tensorindex_values, i);
    // Concrete time indices are not contracted
    if (not detail::is_time_index_value(current_value)) {
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
        t) {
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
