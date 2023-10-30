// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace detail {
// Get the values that encode the generic tensor indices for the first operand,
// second operand, and result tensor so that the first NumIndicesToContract
// indices will contract and the TensorExpression written with them will be
// valid
//
// Note: Implementation assumes that the indices in the index pairs to contract
// have the same type (spatial or spacetime)
template <size_t NumIndicesToContract, size_t NumIndices1, size_t NumIndices2>
constexpr auto
get_tensor_index_values_for_tensors_to_contract_and_result_tensor(
    const std::array<bool, NumIndices1>& index_type_is_spacetime1,
    const std::array<bool, NumIndices1>& valence_is_lower1,
    const std::array<bool, NumIndices2>& index_type_is_spacetime2,
    const std::array<bool, NumIndices2>& valence_is_lower2) {
  constexpr size_t num_result_indices =
      NumIndices1 + NumIndices2 - 2 * NumIndicesToContract;

  // (first op index values, second op index values, result tensor index values)
  std::tuple<std::array<size_t, NumIndices1>, std::array<size_t, NumIndices2>,
             std::array<size_t, num_result_indices>>
      tensor_index_values{};

  // the next lower spacetime index value that we have not yet used
  size_t next_lower_spacetime_value = 0;
  // the next lower spatial index value that we have not yet used
  size_t next_lower_spatial_value = tenex::TensorIndex_detail::spatial_sentinel;

  // assign first operand's tensor index values
  for (size_t i = 0; i < NumIndices1; i++) {
    const bool index1_is_spacetime = gsl::at(index_type_is_spacetime1, i);
    const bool index1_is_lower = gsl::at(valence_is_lower1, i);
    if (index1_is_spacetime) {
      gsl::at(std::get<0>(tensor_index_values), i) =
          index1_is_lower ? next_lower_spacetime_value
                          : tenex::get_tensorindex_value_with_opposite_valence(
                                next_lower_spacetime_value);
      next_lower_spacetime_value++;
    } else {
      gsl::at(std::get<0>(tensor_index_values), i) =
          index1_is_lower ? next_lower_spatial_value
                          : tenex::get_tensorindex_value_with_opposite_valence(
                                next_lower_spatial_value);
      next_lower_spatial_value++;
    }
  }

  // assign the first NumIndicesToContract index values of the second operand so
  // that they will contract with the first NumIndicesToContract indices of the
  // first operand
  for (size_t i = 0; i < NumIndicesToContract; i++) {
    gsl::at(std::get<1>(tensor_index_values), i) =
        tenex::get_tensorindex_value_with_opposite_valence(
            gsl::at(std::get<0>(tensor_index_values), i));
  }

  // assign the remaining index values of the second operand so that they are
  // not duplicates of any previously-used indices
  for (size_t i = NumIndicesToContract; i < NumIndices2; i++) {
    const bool index2_is_spacetime = gsl::at(index_type_is_spacetime2, i);
    const bool index2_is_lower = gsl::at(valence_is_lower2, i);
    if (index2_is_spacetime) {
      gsl::at(std::get<1>(tensor_index_values), i) =
          index2_is_lower ? next_lower_spacetime_value
                          : tenex::get_tensorindex_value_with_opposite_valence(
                                next_lower_spacetime_value);
      next_lower_spacetime_value++;
    } else {
      gsl::at(std::get<1>(tensor_index_values), i) =
          index2_is_lower ? next_lower_spatial_value
                          : tenex::get_tensorindex_value_with_opposite_valence(
                                next_lower_spatial_value);
      next_lower_spatial_value++;
    }
  }

  // assign the free indices of the first operand to the result tensor indices
  for (size_t i = 0; i < NumIndices1 - NumIndicesToContract; i++) {
    gsl::at(std::get<2>(tensor_index_values), i) =
        gsl::at(std::get<0>(tensor_index_values), i + NumIndicesToContract);
  }

  // assign the free indices of the second operand to the result tensor indices
  for (size_t i = 0; i < NumIndices2 - NumIndicesToContract; i++) {
    gsl::at(std::get<2>(tensor_index_values),
            i + NumIndices1 - NumIndicesToContract) =
        gsl::at(std::get<1>(tensor_index_values), i + NumIndicesToContract);
  }

  return tensor_index_values;
}

// \brief Helper struct for contracting the first `NumIndicesToContract`
// indices of two `Tensor`s
//
// \tparam NumIndicesToContract the number of indices to contract
// \param T1 the type of the first `Tensor` of the two to contract
// \param T2 the type of the second `Tensor` of the two to contract
template <size_t NumIndicesToContract, typename T1, typename T2,
          size_t NumIndices1 = tmpl::size<typename T1::symmetry>::value,
          size_t NumIndices2 = tmpl::size<typename T2::symmetry>::value,
          size_t NumResultIndices =
              NumIndices1 + NumIndices2 - 2 * NumIndicesToContract,
          typename ContractedIndicesSeq =
              std::make_index_sequence<NumIndicesToContract>,
          typename TensorIndices1Seq = std::make_index_sequence<NumIndices1>,
          typename TensorIndices2Seq = std::make_index_sequence<NumIndices2>,
          typename ResultIndicesSeq =
              std::make_index_sequence<NumResultIndices>>
struct contract_first_n_indices_impl;

template <size_t NumIndicesToContract, typename X1, typename Symm1,
          typename... Indices1, typename X2, typename Symm2,
          typename... Indices2, size_t NumIndices1, size_t NumIndices2,
          size_t NumResultIndices, size_t... ContractedIndicesInts,
          size_t... Ints1, size_t... Ints2, size_t... ResultInts>
struct contract_first_n_indices_impl<
    NumIndicesToContract, Tensor<X1, Symm1, tmpl::list<Indices1...>>,
    Tensor<X2, Symm2, tmpl::list<Indices2...>>, NumIndices1, NumIndices2,
    NumResultIndices, std::index_sequence<ContractedIndicesInts...>,
    std::index_sequence<Ints1...>, std::index_sequence<Ints2...>,
    std::index_sequence<ResultInts...>> {
  static_assert(NumIndicesToContract <= sizeof...(Indices1) and
                    NumIndicesToContract <= sizeof...(Indices2),
                "Cannot request to contract more indices than indices in "
                "either of the two Tensors to contract.");

  static constexpr std::array<bool, NumIndices1> valence_is_lower1 = {
      {(Indices1::ul == UpLo::Lo)...}};
  static constexpr std::array<bool, NumIndices2> valence_is_lower2 = {
      {(Indices2::ul == UpLo::Lo)...}};
  static constexpr std::array<bool, NumIndices1> index_type_is_spacetime1 = {
      {(Indices1::index_type == IndexType::Spacetime)...}};
  static constexpr std::array<bool, NumIndices2> index_type_is_spacetime2 = {
      {(Indices2::index_type == IndexType::Spacetime)...}};

  // if this is removed and support for automatic contraction of a spatial index
  // with a spacetime index is wanted, the implementation of
  // get_tensor_index_values_for_tensors_to_contract_and_result_tensor() also
  // needs to be updated to support this
  static_assert(
      (... and (index_type_is_spacetime1[ContractedIndicesInts] ==
                index_type_is_spacetime2[ContractedIndicesInts])),
      "You are trying to automatically contract a spatial index with a "
      "spacetime index, but this is not supported for "
      "contract_first_n_indices().");

  // the values of the generic indices (i.e. `TensorIndex::value`s) that
  // uniquely identify different generic indices
  static constexpr auto tensor_index_values =
      get_tensor_index_values_for_tensors_to_contract_and_result_tensor<
          NumIndicesToContract>(index_type_is_spacetime1, valence_is_lower1,
                                index_type_is_spacetime2, valence_is_lower2);
  static constexpr std::array<size_t, NumIndices1> tensor_index_values1 =
      std::get<0>(tensor_index_values);
  static constexpr std::array<size_t, NumIndices2> tensor_index_values2 =
      std::get<1>(tensor_index_values);
  static constexpr std::array<size_t, NumResultIndices>
      result_tensor_index_values = std::get<2>(tensor_index_values);

  using lhs_tensorindex_list =
      tmpl::list<TensorIndex<result_tensor_index_values[ResultInts]>...>;

  // \brief Contract first N indices by evaluating a `TensorExpression`
  //
  // \param lhs_tensor the result LHS `Tensor`
  // \param tensor1 the first `Tensor` of the two to contract
  // \param tensor2 the second `Tensor` of the two to contract
  template <typename LhsTensor>
  static void apply(const gsl::not_null<LhsTensor*> lhs_tensor,
                    const Tensor<X1, Symm1, tmpl::list<Indices1...>>& tensor1,
                    const Tensor<X2, Symm2, tmpl::list<Indices2...>>& tensor2) {
    constexpr bool evaluate_subtrees =
        decltype(tensor1(TensorIndex<tensor_index_values1[Ints1]>{}...) *
                 tensor2(TensorIndex<tensor_index_values2[Ints2]>{}...))::
            primary_subtree_contains_primary_start;
    // Calls `evaluate_impl()` instead of `evaluate()` because `evaluate()`
    // takes `TensorIndex` lvalue references as template parameters, but here we
    // have `TensorIndex` types, and `evaluate()` cannot be overloaded to accept
    // both the former and the latter, so we simply circumvent the "top level"
    // `evaluate()` call that is normally used when evaluating the result of a
    // `TensorExpression`
    tenex::detail::evaluate_impl<
        evaluate_subtrees,
        TensorIndex<result_tensor_index_values[ResultInts]>...>(
        lhs_tensor, tensor1(TensorIndex<tensor_index_values1[Ints1]>{}...) *
                        tensor2(TensorIndex<tensor_index_values2[Ints2]>{}...));
  }

  // \brief Contract first N indices by evaluating a `TensorExpression`
  //
  // \param tensor1 the first `Tensor` of the two to contract
  // \param tensor2 the second `Tensor` of the two to contract
  static auto apply(const Tensor<X1, Symm1, tmpl::list<Indices1...>>& tensor1,
                    const Tensor<X2, Symm2, tmpl::list<Indices2...>>& tensor2) {
    using rhs_expression =
        decltype(tensor1(TensorIndex<tensor_index_values1[Ints1]>{}...) *
                 tensor2(TensorIndex<tensor_index_values2[Ints2]>{}...));
    using rhs_tensorindex_list = typename rhs_expression::args_list;
    using rhs_symmetry = typename rhs_expression::symmetry;
    using rhs_tensorindextype_list = typename rhs_expression::index_list;

    using lhs_tensor_symm_and_indices =
        tenex::LhsTensorSymmAndIndices<rhs_tensorindex_list,
                                       lhs_tensorindex_list, rhs_symmetry,
                                       rhs_tensorindextype_list>;

    Tensor<typename rhs_expression::type,
           typename lhs_tensor_symm_and_indices::symmetry,
           typename lhs_tensor_symm_and_indices::tensorindextype_list>
        lhs_tensor{};

    apply(make_not_null(&lhs_tensor), tensor1, tensor2);

    return lhs_tensor;
  }
};
}  // namespace detail

/// \ingroup TensorGroup
/// \brief Contract the first N indices of two `Tensor`s
///
/// \details
/// The indices of `lhs_tensor` should be the concatenation of the uncontracted
/// indices of `tensor1` and the uncontracted indices of `tensor2`, in this
/// order. For example, if `tensor1` is rank 3, `tensor2` is rank 4, and we want
/// to contract the first two indices, the indices of `lhs_tensor` need to be
/// the last index of `tensor1` followed by the 3rd index and then the 4th index
/// of `tensor2`.
///
/// The index types (spatial or spacetime) must be the same for the two indices
/// in a pair of indices being contracted. Support can be added to this function
/// to automatically contract the spatial indices of a spacetime index with a
/// spatial index.
///
/// \tparam NumIndicesToContract the number of indices to contract
/// \param lhs_tensor the result LHS `Tensor`
/// \param tensor1 the first `Tensor` of the two to contract
/// \param tensor2 the second `Tensor` of the two to contract
template <size_t NumIndicesToContract, typename LhsTensor, typename T1,
          typename T2>
void contract_first_n_indices(const gsl::not_null<LhsTensor*> lhs_tensor,
                              const T1& tensor1, const T2& tensor2) {
  detail::contract_first_n_indices_impl<NumIndicesToContract, T1, T2>::apply(
      lhs_tensor, tensor1, tensor2);
}

/// \ingroup TensorGroup
/// \brief Contract the first N indices of two `Tensor`s
///
/// \details
/// The indices of the returned `Tensor` will be the concatenation of the
/// uncontracted indices of `tensor1` and the uncontracted indices of `tensor2`,
/// in this order. For example, if `tensor1` is rank 3, `tensor2` is rank 4, and
/// we want to contract the first two indices, the indices of the returned
/// `Tensor` will be the last index of `tensor1` followed by the 3rd index and
/// then the 4th index of `tensor2`.
///
/// The index types (spatial or spacetime) must be the same for the two indices
/// in a pair of indices being contracted. Support can be added to this function
/// to automatically contract the spatial indices of a spacetime index with a
/// spatial index.
///
/// \tparam NumIndicesToContract the number of indices to contract
/// \param tensor1 the first `Tensor` of the two to contract
/// \param tensor2 the second `Tensor` of the two to contract
template <size_t NumIndicesToContract, typename T1, typename T2>
auto contract_first_n_indices(const T1& tensor1, const T2& tensor2) {
  return detail::contract_first_n_indices_impl<NumIndicesToContract, T1,
                                               T2>::apply(tensor1, tensor2);
}
