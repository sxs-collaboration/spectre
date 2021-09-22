// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for adding and subtracting tensors

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/IndexPropertyCheck.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace TensorExpressions {
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2,
          int Sign>
struct AddSub;
}  // namespace TensorExpressions
template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args, typename ReducedArgs>
struct TensorExpression;
/// \endcond

namespace TensorExpressions {
namespace detail {
/// \brief Computes the rearranged symmetry of one operand according to the
/// generic index order of the other operand
///
/// \details
/// Here is an example of what the algorithm does:
///
/// Given \f$R_{abc} + S_{cab}\f$, reorder \f$S\f$' symmetry according to
/// \f$R\f$'s index order
/// `tensorindex_transformation`:
/// \code
/// {1, 2, 0} // positions of R's indices {a, b, c} in S' indices {c, a, b}
/// \endcode
/// `input_symm`:
/// \code
/// {2, 2, 1} // S' symmetry, where c and a are symmetric indices
/// \endcode
/// returned equivalent `output_symm`:
/// \code
/// {2, 1, 2} // S' symmetry (input_symm) rearranged to R's index order (abc)
/// \endcode
///
/// One special case scenario to note is when concrete time indices are
/// involved in the transformation. Consider transforming the symmetry for
/// some tensor \f$S_{ab}\f$ to the index order of another tensor
/// \f$R_{btat}\f$. This would be necessary in an expression such as
/// \f$R_{btat} + S_{ab}\f$. The transformation of the symmetry for \f$S\f$
/// according to the index order of \f$R\f$ cannot simply be the list of
/// positions of \f$R\f$'s indices in \f$S\f$' indices, as \f$S\f$ does not
/// contain all of \f$R\f$'s indices, because it has no time indices. To handle
/// cases like this, a placeholder value for the position of any time index
/// must be substituted for an actual position, since one may not exist. In this
/// example, the proper input transformation (`tensorindex_transformation`)
/// would need to be `{1, PLACEHOLDER_VALUE, 0, PLACEHOLDER_VALUE}`, where
/// `PLACEHOLDER_VALUE` is defined by
/// `TensorIndexTransformation_detail::time_index_position_placeholder`. `1` and
/// `0` are the positions of \f$b\f$ and \f$a\f$ in \f$S\f$, and the placeholder
/// is used for the positions of time indices. In computing the output
/// transformed symmetry, the function will insert a `0` at each position where
/// this placeholder is found in the transformation. For example, if
/// `input_symm` is `{2, 1}`, the returned output multi-index will be
/// `{1, 0, 2, 0}`. Note that the symmetry returned by this function is not
/// necessarily in the canonical form defined by ::Symmetry. This example with
/// time indices is an example of this, as `0` is not a permitted ::Symmetry
/// value, and the canonical form would have increasing symmetry values from
/// right to left. In addition, even though the time indices in the rearranged
/// symmetry will have the same symmetry value (`0`), this bears no impact on
/// `get_addsub_symm`'s computation of the symmetry of the tensor resulting
/// from the addition or subtraction.
///
/// \tparam NumIndicesIn the number of indices in the operand whose symmetry is
/// being transformed
/// \tparam NumIndicesOut the number of indices in the other operand whose index
/// order is the order the input operand symmetry is being transformed to
/// \param input_symm the input operand symmetry to transform
/// \param tensorindex_transformation the positions of the other operand's
/// generic indices in the generic indices of the operand whose symmetry is
/// being transformed (see details)
/// \return the input operand symmetry rearranged according to the generic index
/// order of the other operand
template <size_t NumIndicesIn, size_t NumIndicesOut>
SPECTRE_ALWAYS_INLINE constexpr std::array<std::int32_t, NumIndicesOut>
transform_addsub_symm(const std::array<std::int32_t, NumIndicesIn>& input_symm,
                      const std::array<size_t, NumIndicesOut>&
                          tensorindex_transformation) noexcept {
  std::array<std::int32_t, NumIndicesOut> output_symm =
      make_array<NumIndicesOut, std::int32_t>(0);
  for (size_t i = 0; i < NumIndicesOut; i++) {
    gsl::at(output_symm, i) =
        (gsl::at(tensorindex_transformation, i) ==
         TensorIndexTransformation_detail::time_index_position_placeholder)
            ? 0
            : gsl::at(input_symm, gsl::at(tensorindex_transformation, i));
  }
  return output_symm;
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the canonical symmetry of the tensor resulting from
/// adding or subtracting two tensors, according to their symmetries
///
/// \details The canonical symmetry returned follows the convention defined by
/// ::Symmetry: symmetry values are in ascending order from right to left. If
/// the convention implemented by ::Symmetry changes, this function will also
/// need to be updated to match the new convention. The ::Symmetry metafunction
/// could instead be used on the result of this function, but that would
/// introduce avoidable and unnecessary extra computations, so it is not used.
///
/// This function treats the two input symmetries as aligned (i.e. each position
/// of `symm1` and `symm2` corresponds to a shared generic index at that
/// position). The resultant symmetry is determined as follows: indices that are
/// symmetric in both input symmetries are also symmetric in the resultant
/// tensor.
///
/// \param symm1 the symmetry of the first tensor being added or subtracted
/// \param symm2 the symmetry of the second tensor being added or subtracted
/// \return the canonical symmetry of the tensor resulting from adding or
/// subtracting two tensors
template <size_t NumIndices, Requires<(NumIndices >= 2)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_addsub_symm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& symm2) {
  constexpr std::int32_t max_int = std::numeric_limits<std::int32_t>::max();
  std::array<std::int32_t, NumIndices> addsub_symm =
      make_array<NumIndices>(max_int);
  size_t right_index = NumIndices - 1;
  std::int32_t symm_value_to_set = 1;

  while (right_index < NumIndices) {
    std::int32_t symm1_value_to_find = symm1[right_index];
    std::int32_t symm2_value_to_find = symm2[right_index];
    // if we haven't yet set right_index for the resultant symmetry
    if (addsub_symm[right_index] == max_int) {
      addsub_symm[right_index] = symm_value_to_set;
      for (size_t left_index = right_index - 1; left_index < NumIndices;
           left_index--) {
        // if left_index of the resultant symmetry is not yet set and we've
        // found a common symmetry between symm1 and symm2 at this index
        if (addsub_symm[left_index] == max_int and
            symm1[left_index] == symm1_value_to_find and
            symm2[left_index] == symm2_value_to_find) {
          addsub_symm[left_index] = symm_value_to_set;
        }
      }
      symm_value_to_set++;
    }
    right_index--;
  }

  return addsub_symm;
}

template <size_t NumIndices, Requires<(NumIndices == 1)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_addsub_symm(
    const std::array<std::int32_t, NumIndices>& /*symm1*/,
    const std::array<std::int32_t, NumIndices>& /*symm2*/) {
  // return {{1}} instead of symm1 in case symm1 is not in the canonical form
  return {{1}};
}

template <size_t NumIndices, Requires<(NumIndices == 0)> = nullptr>
constexpr std::array<std::int32_t, NumIndices> get_addsub_symm(
    const std::array<std::int32_t, NumIndices>& symm1,
    const std::array<std::int32_t, NumIndices>& /*symm2*/) {
  return symm1;
}
/// @}

/// \ingroup TensorExpressionsGroup
/// \brief Helper struct for computing the canonical symmetry of the tensor
/// resulting from adding or subtracting two tensors, according to their
/// symmetries and generic index orders
///
/// \details The resultant symmetry (`type`) values correspond to the index
/// order of the first tensor operand being added or subtracted:
/// `TensorIndexList1`.
///
/// \tparam SymmList1 the ::Symmetry of the first operand
/// \tparam SymmList2 the ::Symmetry of the second operand
/// \tparam TensorIndexList1 the generic indices of the first operand
/// \tparam TensorIndexList2 the generic indices of the second operand
template <typename SymmList1, typename SymmList2, typename TensorIndexList1,
          typename TensorIndexList2,
          size_t NumIndices1 = tmpl::size<SymmList1>::value,
          size_t NumIndices2 = tmpl::size<SymmList2>::value,
          typename IndexSequence1 = std::make_index_sequence<NumIndices1>>
struct AddSubSymmetry;

template <
    template <typename...> class SymmList1, typename... Symm1,
    template <typename...> class SymmList2, typename... Symm2,
    template <typename...> class TensorIndexList1, typename... TensorIndices1,
    template <typename...> class TensorIndexList2, typename... TensorIndices2,
    size_t NumIndices1, size_t NumIndices2, size_t... Ints1>
struct AddSubSymmetry<SymmList1<Symm1...>, SymmList2<Symm2...>,
                      TensorIndexList1<TensorIndices1...>,
                      TensorIndexList2<TensorIndices2...>, NumIndices1,
                      NumIndices2, std::index_sequence<Ints1...>> {
  static constexpr std::array<size_t, NumIndices1> tensorindex_values1 = {
      {TensorIndices1::value...}};
  static constexpr std::array<size_t, NumIndices2> tensorindex_values2 = {
      {TensorIndices2::value...}};
  // positions of tensorindex_values1 in tensorindex_values2
  static constexpr std::array<size_t, NumIndices1> op2_to_op1_map =
      ::TensorExpressions::compute_tensorindex_transformation(
          tensorindex_values2, tensorindex_values1);

  static constexpr std::array<std::int32_t, NumIndices1> symm1 = {
      {Symm1::value...}};
  static constexpr std::array<std::int32_t, NumIndices2> symm2 = {
      {Symm2::value...}};
  // 2nd argument is symm2 rearranged according to `TensorIndexList1` order
  // so that the two symmetry arguments to `get_addsub_symm` are aligned
  // w.r.t. their generic index orders
  static constexpr std::array<std::int32_t, NumIndices1> addsub_symm =
      get_addsub_symm(symm1, transform_addsub_symm(symm2, op2_to_op1_map));

  using type = tmpl::integral_list<std::int32_t, addsub_symm[Ints1]...>;
};

/// \ingroup TensorExpressionsGroup
/// \brief Helper struct for defining the symmetry, index list, and
/// generic index list of the tensor resulting from adding or
/// subtracting two tensor expressions
///
/// \tparam T1 the first tensor expression operand
/// \tparam T2 the second tensor expression operand
template <typename T1, typename T2>
struct AddSubType {
  static_assert(std::is_base_of_v<Expression, T1> and
                    std::is_base_of_v<Expression, T2>,
                "Parameters to AddSubType must be TensorExpressions");
  using type =
      tmpl::conditional_t<std::is_same_v<typename T1::type, DataVector> or
                              std::is_same_v<typename T2::type, DataVector>,
                          DataVector, double>;
  using symmetry =
      typename AddSubSymmetry<typename T1::symmetry, typename T2::symmetry,
                                   typename T1::args_list,
                                   typename T2::args_list>::type;
  using index_list = typename T1::index_list;
  using tensorindex_list = typename T1::args_list;
};
}  // namespace detail

template <typename T1, typename T2, typename ArgsList1, typename ArgsList2,
          int Sign>
struct AddSub;

template <typename T1, typename T2, template <typename...> class ArgsList1,
          template <typename...> class ArgsList2, typename... Args1,
          typename... Args2, int Sign>
struct AddSub<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>, Sign>
    : public TensorExpression<
          AddSub<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>, Sign>,
          typename detail::AddSubType<T1, T2>::type,
          typename detail::AddSubType<T1, T2>::symmetry,
          typename detail::AddSubType<T1, T2>::index_list,
          typename detail::AddSubType<T1, T2>::tensorindex_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value or
                    std::is_same<T1, NumberAsExpression>::value or
                    std::is_same<T2, NumberAsExpression>::value,
                "Cannot add or subtract Tensors holding different data types.");
  static_assert(
      detail::IndexPropertyCheck<typename T1::index_list,
                                 typename T2::index_list, ArgsList1<Args1...>,
                                 ArgsList2<Args2...>>::value,
      "You are attempting to add indices of different types, e.g. T^a_b + "
      "S^b_a, which doesn't make sense. The indices may also be in different "
      "frames, different types (spatial vs. spacetime) or of different "
      "dimension.");
  static_assert(Sign == 1 or Sign == -1,
                "Invalid Sign provided for addition or subtraction of Tensor "
                "elements. Sign must be 1 (addition) or -1 (subtraction).");

  using type = typename detail::AddSubType<T1, T2>::type;
  using symmetry = typename detail::AddSubType<T1, T2>::symmetry;
  using index_list = typename detail::AddSubType<T1, T2>::index_list;
  // number of indices in the tensor resulting from addition or subtraction
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  // number of indices in the second operand in the addition or subtraction
  static constexpr auto num_tensor_indices_op2 = sizeof...(Args2);
  using args_list = typename T1::args_list;
  static constexpr std::array<size_t, num_tensor_indices_op2>
      operand_index_transformation =
          compute_tensorindex_transformation<num_tensor_indices,
                                             num_tensor_indices_op2>(
              {{Args1::value...}}, {{Args2::value...}});
  // positions of indices in first operand where generic spatial indices are
  // used for spacetime indices
  static constexpr auto op1_spatial_spacetime_index_positions =
      detail::get_spatial_spacetime_index_positions<typename T1::index_list,
                                                    ArgsList1<Args1...>>();
  // positions of indices in second operand where generic spatial indices are
  // used for spacetime indices
  static constexpr auto op2_spatial_spacetime_index_positions =
      detail::get_spatial_spacetime_index_positions<typename T2::index_list,
                                                    ArgsList2<Args2...>>();

  static constexpr bool ops_have_generic_indices_at_same_positions =
      generic_indices_at_same_positions<tmpl::list<Args1...>,
                                        tmpl::list<Args2...>>::value;

  AddSub(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~AddSub() override = default;

  /// \brief Helper function for computing the sum of or difference between
  /// components at given multi-indices from both operands of the expression
  ///
  /// \details Both multi-index arguments must be ordered according to their
  /// operand's respective generic index ordering
  ///
  /// \param op1_multi_index the multi-index of the component of the first
  /// operand
  /// \param op2_multi_index the multi-index of the component of the second
  /// operand
  /// \return the sum of or difference between the two components' values
  SPECTRE_ALWAYS_INLINE decltype(auto) add_or_subtract(
      const std::array<size_t, num_tensor_indices>& op1_multi_index,
      const std::array<size_t, num_tensor_indices_op2>& op2_multi_index)
      const noexcept {
    if constexpr (Sign == 1) {
      return t1_.get(op1_multi_index) + t2_.get(op2_multi_index);
    } else {
      return t1_.get(op1_multi_index) - t2_.get(op2_multi_index);
    }
  }

  /// \brief Return the value of the component at the given multi-index of the
  /// tensor resulting from addition or subtraction
  ///
  /// \details One important detail to note about the type of the `AddSub`
  /// expression is that its two operands may have (i) different generic index
  /// orders, and/or (ii) different indices in their `index_list`s if where one
  /// operand uses a generic spatial index for a spacetime index, the other
  /// tensor may use that generic spatial index for a spatial index of the same
  /// dimension, valence, and frame. Therefore, there are four possible cases
  /// for an `AddSub` expression that are considered in the implementation:
  /// - same generic index order, spatial spacetime indices in expression
  /// - same generic index order, spatial spacetime indices not in expression
  /// - different generic index order, spatial spacetime indices in expression
  /// - different generic index order, spatial spacetime indices not in
  /// expression
  ///
  /// This means that for expressions where the generic index orders differ, a
  /// multi-index for a component of one operand is a (possible) rearrangement
  /// of the equivalent multi-index for a component in the other operand. This
  /// also means that for expressions where (at least once) a generic spatial
  /// index is used for a spacetime index, then, after accounting
  /// for potential reordering due to different generic index orders, a
  /// multi-index's values for a component of one operand are (possibly) shifted
  /// by one, compared to the multi-index's values for a component in the other
  /// operand.
  ///
  /// For example, given \f$R_{ij} + S_{ji}\f$, let \f$R\f$'s first index be
  /// a spacetime index, but \f$R\f$'s second index and both of \f$S\f$' indices
  /// be spatial indices. If \f$i = 2\f$ and \f$j = 0\f$, then when we compute
  /// \f$R_{20} + S_{02}\f$, the multi-index for \f$R_{20}\f$ is
  /// `{2 + 1, 0} = {3, 0}` (first value shifted because it is a spacetime
  /// index) and the multi-index for \f$S_{02}\f$ is `[0, 2]`. Because the first
  /// operand of an `AddSub` expresion propagates its generic index order and
  /// index list ( \ref SpacetimeIndex "TensorIndexType"s) as the `AddSub`'s own
  /// generic index order and index list, the `result_multi_index` is equivalent
  /// to the multi-index for the first operand. Thus, we need only compute the
  /// second operand's multi-index as a transformation of the first: reorder and
  /// shift the values of the first operand to compute the equivalent
  /// multi-index for the second operand.
  ///
  /// \param result_multi_index the multi-index of the component of the result
  /// tensor to retrieve
  /// \return the value of the component at `result_multi_index` in the result
  /// tensor
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    if constexpr (ops_have_generic_indices_at_same_positions) {
      if constexpr (op1_spatial_spacetime_index_positions.size() != 0 or
                    op2_spatial_spacetime_index_positions.size() != 0) {
        // Operands have the same generic index order, but at least one of them
        // has at least one spacetime index where a spatial index has been used,
        // so we need to compute the 2nd operand's (possibly) shifted
        // multi-index values
        constexpr std::array<std::int32_t, num_tensor_indices>
            spatial_spacetime_index_transformation =
                detail::spatial_spacetime_index_transformation_from_positions<
                    num_tensor_indices>(op1_spatial_spacetime_index_positions,
                                        op2_spatial_spacetime_index_positions);
        std::array<size_t, num_tensor_indices> op2_multi_index =
            result_multi_index;
        for (size_t i = 0; i < num_tensor_indices; i++) {
          gsl::at(op2_multi_index, i) = static_cast<size_t>(
              static_cast<std::int32_t>(gsl::at(op2_multi_index, i)) +
              gsl::at(spatial_spacetime_index_transformation, i));
        }
        return add_or_subtract(result_multi_index, op2_multi_index);
      } else {
        // Operands have the same generic index order and neither of them has
        // a spacetime index where a spatial index has been used, so
        // both operands have the same multi-index
        return add_or_subtract(result_multi_index, result_multi_index);
      }
    } else {
      if constexpr (op1_spatial_spacetime_index_positions.size() != 0 or
                    op2_spatial_spacetime_index_positions.size() != 0) {
        // Operands don't have the same generic index order and at least one of
        // them has at least one spacetime index where a spatial index has been
        // used, so we need to compute the 2nd operand's (possibly) shifted
        // multi-index values and reorder them with respect to the 2nd operand's
        // generic index order

        // The list of positions where generic spatial indices were used for
        // spacetime indices in the second operand, but rearranged in terms of
        // the first operand's generic index order.
        constexpr std::array<size_t,
                             op2_spatial_spacetime_index_positions.size()>
            transformed_op2_spatial_spacetime_index_positions = []() {
              std::array<size_t, op2_spatial_spacetime_index_positions.size()>
                  transformed_positions{};
              for (size_t i = 0;
                   i < op2_spatial_spacetime_index_positions.size(); i++) {
                gsl::at(transformed_positions, i) =
                    gsl::at(operand_index_transformation,
                            gsl::at(op2_spatial_spacetime_index_positions, i));
              }
              return transformed_positions;
            }();

        // According to the transformed positions above, compute the value shift
        // needed to convert from multi-indices of the first operand to
        // multi-indices of the 2nd operand (with the generic index order of the
        // first)
        constexpr std::array<std::int32_t, num_tensor_indices>
            spatial_spacetime_index_transformation =
                detail::spatial_spacetime_index_transformation_from_positions<
                    num_tensor_indices>(
                    op1_spatial_spacetime_index_positions,
                    transformed_op2_spatial_spacetime_index_positions);
        std::array<size_t, num_tensor_indices> op2_multi_index =
            result_multi_index;
        for (size_t i = 0; i < num_tensor_indices; i++) {
          gsl::at(op2_multi_index, i) = static_cast<size_t>(
              static_cast<std::int32_t>(gsl::at(op2_multi_index, i)) +
              gsl::at(spatial_spacetime_index_transformation, i));
        }
        return add_or_subtract(
            result_multi_index,
            transform_multi_index(op2_multi_index,
                                  operand_index_transformation));
      } else {
        // Operands don't have the same generic index order, but neither of them
        // has a spacetime index where a spatial index has been used, so we just
        // need to reorder the 2nd operand's multi_index according to its
        // generic index order
        return add_or_subtract(
            result_multi_index,
            transform_multi_index(result_multi_index,
                                  operand_index_transformation));
      }
    }
  }

 private:
  T1 t1_;
  T2 t2_;
};
}  // namespace TensorExpressions

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X1, typename X2, typename Symm1,
          typename Symm2, typename IndexList1, typename IndexList2,
          typename Args1, typename Args2>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T1, X1, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X2, Symm2, IndexList2, Args2>& t2) {
  using op1_generic_indices =
      typename TensorExpressions::detail::remove_time_indices<Args1>::type;
  using op2_generic_indices =
      typename TensorExpressions::detail::remove_time_indices<Args2>::type;
  static_assert(tmpl::size<op1_generic_indices>::value ==
                    tmpl::size<op2_generic_indices>::value,
                "Tensor addition is only possible when the same number of "
                "generic indices are used with both operands.");
  static_assert(
      tmpl::equal_members<op1_generic_indices, op2_generic_indices>::value,
      "The generic indices when adding two tensors must be equal. This error "
      "occurs from expressions like R(ti_a, ti_b) + S(ti_c, ti_a)");
  return TensorExpressions::AddSub<T1, T2, Args1, Args2, 1>(~t1, ~t2);
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the sum of a tensor
/// expression and a `double`
///
/// \details
/// The tensor expression operand must represent an expression that, when
/// evaluated, would be a rank 0 tensor. For example, if `R` and `S` are
/// Tensors, here is a non-exhaustive list of some of the acceptable forms that
/// the tensor expression operand could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
/// - `R(ti_t, ti_t)`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the sum
/// \tparam X the type of data stored in the tensor expression operand of the
/// sum
/// \tparam Symm the ::Symmetry of the derived TensorExpression type of the
/// tensor expression operand of the sum
/// \tparam IndexList the \ref SpacetimeIndex "TensorIndexType"s of the derived
/// TensorExpression type of the tensor expression operand of the sum
/// \tparam Args the comma-separated list of generic indices of the derived
/// TensorExpression type of the tensor expression operand of the sum
/// \param t the tensor expression operand of the sum
/// \param number the `double` operand of the sum
/// \return the tensor expression representing the sum of a tensor expression
/// and a `double`
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t,
    const double number) {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only add a number to a tensor expression that evaluates to a rank 0"
      "tensor.");
  return t + TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto operator+(
    const double number,
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t) {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only add a number to a tensor expression that evaluates to a rank 0"
      "tensor.");
  return TensorExpressions::NumberAsExpression(number) + t;
}
/// @}

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X1, typename X2, typename Symm1,
          typename Symm2, typename IndexList1, typename IndexList2,
          typename Args1, typename Args2>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T1, X1, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X2, Symm2, IndexList2, Args2>& t2) {
  using op1_generic_indices =
      typename TensorExpressions::detail::remove_time_indices<Args1>::type;
  using op2_generic_indices =
      typename TensorExpressions::detail::remove_time_indices<Args2>::type;
  static_assert(tmpl::size<op1_generic_indices>::value ==
                    tmpl::size<op2_generic_indices>::value,
                "Tensor subtraction is only possible when the same number of "
                "generic indices are used with both operands.");
  static_assert(
      tmpl::equal_members<op1_generic_indices, op2_generic_indices>::value,
      "The generic indices when subtracting two tensors must be equal. This "
      "error "
      "occurs from expressions like R(ti_a, ti_b) - S(ti_c, ti_a)");
  return TensorExpressions::AddSub<T1, T2, Args1, Args2, -1>(~t1, ~t2);
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the difference of a tensor
/// expression and a `double`
///
/// \details
/// The tensor expression operand must represent an expression that, when
/// evaluated, would be a rank 0 tensor. For example, if `R` and `S` are
/// Tensors, here is a non-exhaustive list of some of the acceptable forms that
/// the tensor expression operand could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
/// - `R(ti_t, ti_t)`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the difference
/// \tparam X the type of data stored in the tensor expression operand of the
/// difference
/// \tparam Symm the ::Symmetry of the derived TensorExpression type of the
/// tensor expression operand of the difference
/// \tparam IndexList the \ref SpacetimeIndex "TensorIndexType"s of the derived
/// TensorExpression type of the tensor expression operand of the difference
/// \tparam Args the comma-separated list of generic indices of the derived
/// TensorExpression type of the tensor expression operand of the difference
/// \param t the tensor expression operand of the difference
/// \param number the `double` operand of the difference
/// \return the tensor expression representing the difference of a tensor
/// expression and a `double`
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t,
    const double number) {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only subtract a number from a tensor expression that evaluates to a "
      "rank 0 tensor.");
  return t - TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto operator-(
    const double number,
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t) {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only subtract a number from a tensor expression that evaluates to a "
      "rank 0 tensor.");
  return TensorExpressions::NumberAsExpression(number) - t;
}
/// @}
