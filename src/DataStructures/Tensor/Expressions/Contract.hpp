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
using indices_contractible =
    std::integral_constant<bool,
                           I1::dim == I2::dim and I1::ul != I2::ul and
                               I1::fr == I2::fr and I1::index == I2::index>;

template <typename T, typename X, typename SymmList, typename IndexList,
          typename Args>
struct ComputeContractedTypeImpl;

template <typename T, typename X, template <typename...> class SymmList,
          typename IndexList, typename Args, typename... Symm>
struct ComputeContractedTypeImpl<T, X, SymmList<Symm...>, IndexList, Args> {
  using type =
      TensorExpression<T, X, Symmetry<Symm::value...>, IndexList, Args>;
};

template <typename Index1, typename Index2, typename T, typename X,
          typename Symm, typename IndexList, typename Args>
using ComputeContractedType = typename ComputeContractedTypeImpl<
    T, X, tmpl::erase<tmpl::erase<Symm, Index2>, Index1>,
    tmpl::erase<tmpl::erase<IndexList, Index2>, Index1>,
    tmpl::erase<tmpl::erase<Args, Index2>, Index1>>::type;

template <int I, typename Index1, typename Index2>
struct ComputeContractionImpl {
  template <typename... LhsIndices, typename T, typename S>
  static SPECTRE_ALWAYS_INLINE typename T::type apply(S tensor_index,
                                                      const T& t1) {
    tensor_index[Index1::value] = I;
    tensor_index[Index2::value] = I;
    return t1.template get<LhsIndices...>(tensor_index) +
           ComputeContractionImpl<I - 1, Index1, Index2>::template apply<
               LhsIndices...>(tensor_index, t1);
  }
};

template <typename Index1, typename Index2>
struct ComputeContractionImpl<0, Index1, Index2> {
  template <typename... LhsIndices, typename T, typename S>
  static SPECTRE_ALWAYS_INLINE typename T::type apply(S tensor_index,
                                                      const T& t1) {
    tensor_index[Index1::value] = 0;
    tensor_index[Index2::value] = 0;
    return t1.template get<LhsIndices...>(tensor_index);
  }
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename Index1, typename Index2, typename T, typename X,
          typename Symm, typename IndexList, typename ArgsList>
struct TensorContract
    : public TensorExpression<
          TensorContract<Index1, Index2, T, X, Symm, IndexList, ArgsList>, X,
          typename detail::ComputeContractedType<Index1, Index2, T, X, Symm,
                                                 IndexList, ArgsList>::symmetry,
          typename detail::ComputeContractedType<
              Index1, Index2, T, X, Symm, IndexList, ArgsList>::index_list,
          typename detail::ComputeContractedType<
              Index1, Index2, T, X, Symm, IndexList, ArgsList>::args_list>,
      public Expression {
  using CI1 = tmpl::at<IndexList, Index1>;
  using CI2 = tmpl::at<IndexList, Index2>;
  static_assert(tmpl::size<Symm>::value > 1 and
                    tmpl::size<IndexList>::value > 1,
                "Cannot contract indices on a Tensor with rank less than 2");
  static_assert(detail::indices_contractible<CI1, CI2>::value,
                "Cannot contract the requested indices.");

  using new_type = detail::ComputeContractedType<Index1, Index2, T, X, Symm,
                                                 IndexList, ArgsList>;

  using type = X;
  using symmetry = typename new_type::symmetry;
  using index_list = typename new_type::index_list;
  static constexpr auto num_tensor_indices =
      tmpl::size<index_list>::value == 0 ? 1 : tmpl::size<index_list>::value;
  using args_list = tmpl::sort<typename new_type::args_list>;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}

  template <size_t I, size_t Rank, Requires<(I <= Index1::value)> = nullptr>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<int, Rank>& tensor_index_in,
      const std::array<int, num_tensor_indices>& tensor_index) const {
    // -100 is for the slot that will be set later. Easy to debug.
    tensor_index_in[I] = I == Index1::value ? -100 : tensor_index[I];
    fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
  }

  template <size_t I, size_t Rank,
            Requires<(I > Index1::value and I <= Index2::value and
                      I < Rank - 1)> = nullptr>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<int, Rank>& tensor_index_in,
      const std::array<int, Rank - 2>& tensor_index) const {
    // tensor_index is Rank - 2 since it shouldn't be called for Rank 2 case
    // -200 is for the slot that will be set later. Easy to debug.
    tensor_index_in[I] = I == Index2::value ? -200 : tensor_index[I - 1];
    fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
  }

  template <size_t I, size_t Rank,
            Requires<(I > Index2::value and I < Rank - 1)> = nullptr>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<int, Rank>& tensor_index_in,
      const std::array<int, Rank - 2>& tensor_index) const {
    // Left as Rank - 2 since it should never be called for the Rank 2 case
    tensor_index_in[I] = tensor_index[I - 2];
    fill_contracting_tensor_index<I + 1>(tensor_index_in, tensor_index);
  }

  template <size_t I, size_t Rank, Requires<(I == Rank - 1)> = nullptr>
  SPECTRE_ALWAYS_INLINE void fill_contracting_tensor_index(
      std::array<int, Rank>& tensor_index_in,
      const std::array<int, num_tensor_indices>& tensor_index) const {
    tensor_index_in[I] = tensor_index[I - 2];
  }

  template <typename... LhsIndices, typename U>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<U, num_tensor_indices>& new_tensor_index) const {
    // new_tensor_index is the one with _fewer_ components, ie post-contraction
    std::array<int, tmpl::size<Symm>::value> tensor_index;
    // Manually unrolled for loops to compute the tensor_index from the
    // new_tensor_index
    fill_contracting_tensor_index<0>(tensor_index, new_tensor_index);
    return detail::ComputeContractionImpl<CI1::dim - 1, Index1, Index2>::
        template apply<LhsIndices...>(tensor_index, t_);
  }

 private:
  const std::conditional_t<std::is_base_of<Expression, T>::value, T,
                           TensorExpression<T, X, Symm, IndexList, ArgsList>>
      t_;
};

/*!
 * \ingroup TensorExpressionsGroup
 */
template <int Index1, int Index2, typename T, typename X, typename Symm,
          typename IndexList, typename Args>
SPECTRE_ALWAYS_INLINE auto contract(
    const TensorExpression<T, X, Symm, IndexList, Args>& t) {
  return TensorContract<tmpl::int32_t<Index1>, tmpl::int32_t<Index2>, T, X,
                        Symm, IndexList, Args>(~t);
}

namespace detail {
// Helper struct to allow contractions by using repeated indices in operator()
// calls to tensor.
template <template <typename> class TE, typename ReplacedArgList, typename I,
          typename TotalContracted>
struct fully_contract_helper {
  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t) -> decltype(
      contract<
          tmpl::index_of<ReplacedArgList, ti_contracted_t<I::value>>::value,
          tmpl::index_of<ReplacedArgList,
                         ti_contracted_t<I::value + 1>>::value>(
          fully_contract_helper<TE, ReplacedArgList, tmpl::next<I>,
                                TotalContracted>::apply(t))) {
    return contract<
        tmpl::index_of<ReplacedArgList, ti_contracted_t<I::value>>::value,
        tmpl::index_of<ReplacedArgList, ti_contracted_t<I::value + 1>>::value>(
        fully_contract_helper<TE, ReplacedArgList, tmpl::next<I>,
                              TotalContracted>::apply(t));
  }
};

template <template <typename> class TE, typename ReplacedArgList,
          typename TotalContracted>
struct fully_contract_helper<TE, ReplacedArgList,
                             tmpl::int32_t<TotalContracted::value - 1>,
                             TotalContracted> {
  using I = tmpl::int32_t<TotalContracted::value - 1>;
  template <typename T>
  SPECTRE_ALWAYS_INLINE static constexpr auto apply(const T& t) -> decltype(
      contract<
          tmpl::index_of<ReplacedArgList, ti_contracted_t<I::value>>::value,
          tmpl::index_of<ReplacedArgList,
                         ti_contracted_t<I::value + 1>>::value>(
          TE<ReplacedArgList>(t))) {
    return contract<
        tmpl::index_of<ReplacedArgList, ti_contracted_t<I::value>>::value,
        tmpl::index_of<ReplacedArgList, ti_contracted_t<I::value + 1>>::value>(
        TE<ReplacedArgList>(t));
  }
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Represents a fully contracted Tensor
 */
template <template <typename> class TE, typename ReplacedArgList, typename I,
          typename TotalContracted>
using fully_contracted =
    detail::fully_contract_helper<TE, ReplacedArgList, I, TotalContracted>;
}  // namespace TensorExpressions
