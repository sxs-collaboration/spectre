// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Structure.hpp"
#include "ErrorHandling/Assert.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"  // IWYU pragma: keep

// The below values are used to separate upper indices from lower indices and
// spatial indices from spacetime indices.
//
// Tensor expressions perform as many calculations as possible in a constexpr
// context, which means working with fundamental types, specifically integer
// types, is easiest. By using sentinel values defined in one location we can
// easily control the encoding without having magic values floating around in
// many places. Furthermore, encoding all the information in the `size_t` means
// that when a failure occurs in one of the constexpr calculations it is
// reasonably easy to debug because, while encoded, the full type information is
// present. This approach can effectively be thought of as using specific bits
// in the `size_t` to mark information, using the size_t more as a bitfield than
// anything else. For human readability, we use base-10 numbers instead of
// base-2 values that would truly set individual bits.
//
// Spacetime indices are represented by values [0, `spatial_sentinel`) and
// spatial indices are represented by values
// [`spatial_sentinel`, `max_sentinel`). Lower spacetime indices are represented
// by values [0, `upper_sentinel`), and upper spacetime indices are represented
// by values [`upper_sentinel`, `spatial_sentinel`). Lower spatial indices are
// represented by values
// [`spatial_sentinel`, `spatial_sentinel` + `upper_sentinel`), and upper
// spatial indices are represented by values
// [`spatial_sentinel` + `upper_sentinel`, `max_sentinel`). Values equal to or
// above `max_sentinel` are considered invalid for representing an index.
static constexpr size_t spatial_sentinel = 1000;
static constexpr size_t upper_sentinel = 500;
static constexpr size_t upper_spatial_sentinel =
    spatial_sentinel + upper_sentinel;
static constexpr size_t max_sentinel = 2000;

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Represents the indices in a TensorExpression
 *
 * \details
 * Used to denote a tensor index in a tensor slot. This allows the following
 * type of expressions to work:
 * \code{.cpp}
 * auto T = evaluate<ti_a, ti_b>(F(ti_a, ti_b) + S(ti_b, ti_a));
 * \endcode
 * where `decltype(ti_a) == TensorIndex<0>` and
 * `decltype(ti_b) == TensorIndex<1>`. That is, `ti_a` and `ti_b` are
 * placeholders for objects of type `TensorIndex<0>` and `TensorIndex<1>`,
 * respectively.
 */
template <std::size_t I, Requires<(I < max_sentinel)> = nullptr>
struct TensorIndex {
  using value_type = std::size_t;
  using type = TensorIndex<I>;
  static constexpr value_type value = I;
  static constexpr UpLo valence =
      ((I < upper_sentinel) or
       (I >= spatial_sentinel and I < upper_spatial_sentinel))
          ? UpLo::Lo
          : UpLo::Up;
  static constexpr bool is_spacetime = I < spatial_sentinel;
};

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Returns the TensorIndex value of with opposite valence.
 *
 * \details The input value represents a TensorIndex value, which encodes
 * both the valence of the index and whether the index is spacetime or
 * spatial. This function returns the value that corresponds to the encoding of
 * the TensorIndex with the same index type, but opposite valence.
 *
 * For example, 0 is the TensorIndex value for `ti_a`. If `i == 0`, then 500
 * will be returned, which is the TensorIndex value for `ti_A`. If `i == 500`
 * (representing `ti_A`), then 0 (representing `ti_a`) is returned.
 *
 * @param i a TensorIndex value that represents a generic index
 * @return the TensorIndex value that encodes the generic index with the
 * opposite valence
 */
SPECTRE_ALWAYS_INLINE static constexpr size_t
get_tensorindex_value_with_opposite_valence(const size_t i) noexcept {
  assert(i < max_sentinel);  // NOLINT
  if ((i >= upper_sentinel and i < spatial_sentinel) or
      (i >= upper_spatial_sentinel)) {
    // `i` represents an upper index, so return the lower index's encoding
    return i - upper_sentinel;
  } else {
    // `i` represents a lower index, so return the upper index's encoding
    return i + upper_sentinel;
  }
}

// @{
/*!
 * \ingroup TensorExpressionsGroup
 * \brief The available TensorIndex's to use in a TensorExpression
 *
 * Available tensor indices to use in a Tensor Expression.
 * \snippet Test_AddSubtract.cpp use_tensor_index
 */
static constexpr TensorIndex<0> ti_a{};
static constexpr TensorIndex<upper_sentinel> ti_A{};
static constexpr TensorIndex<1> ti_b{};
static constexpr TensorIndex<upper_sentinel + 1> ti_B{};
static constexpr TensorIndex<2> ti_c{};
static constexpr TensorIndex<upper_sentinel + 2> ti_C{};
static constexpr TensorIndex<3> ti_d{};
static constexpr TensorIndex<upper_sentinel + 3> ti_D{};
static constexpr TensorIndex<4> ti_e{};
static constexpr TensorIndex<upper_sentinel + 4> ti_E{};
static constexpr TensorIndex<5> ti_f{};
static constexpr TensorIndex<upper_sentinel + 5> ti_F{};
static constexpr TensorIndex<6> ti_g{};
static constexpr TensorIndex<upper_sentinel + 6> ti_G{};
static constexpr TensorIndex<7> ti_h{};
static constexpr TensorIndex<upper_sentinel + 7> ti_H{};
static constexpr TensorIndex<spatial_sentinel> ti_i{};
static constexpr TensorIndex<upper_spatial_sentinel> ti_I{};
static constexpr TensorIndex<spatial_sentinel + 1> ti_j{};
static constexpr TensorIndex<upper_spatial_sentinel + 1> ti_J{};
static constexpr TensorIndex<spatial_sentinel + 2> ti_k{};
static constexpr TensorIndex<upper_spatial_sentinel + 2> ti_K{};
static constexpr TensorIndex<spatial_sentinel + 3> ti_l{};
static constexpr TensorIndex<upper_spatial_sentinel + 3> ti_L{};
// @}

namespace tt {
/*!
 * \ingroup TypeTraitsGroup TensorExpressionsGroup
 * \brief Check if a type `T` is a TensorIndex used in TensorExpressions
 */
template <typename T>
struct is_tensor_index : std::false_type {};
template <size_t I>
struct is_tensor_index<TensorIndex<I>> : std::true_type {};
}  // namespace tt

namespace detail {
template <typename State, typename Element, typename LHS>
struct rhs_elements_in_lhs_helper {
  using type = std::conditional_t<not std::is_same<tmpl::index_of<LHS, Element>,
                                                   tmpl::no_such_type_>::value,
                                  tmpl::push_back<State, Element>, State>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// Returns a list of all the elements in the typelist Rhs that are also in the
/// typelist Lhs.
///
/// \details
/// Given two typelists `Lhs` and `Rhs`, returns a typelist of all the elements
/// in `Rhs` that are also in `Lhs` in the same order that they are in the
/// `Rhs`.
///
/// ### Usage
/// For typelists `List1` and `List2`,
/// \code{.cpp}
/// using result = rhs_elements_in_lhs<List1, List2>;
/// \endcode
/// \metareturns
/// typelist
///
/// \semantics
/// If `Lhs = tmpl::list<A, B, C, D>` and `Rhs = tmpl::list<B, E, A>`, then
/// \code{.cpp}
/// result = tmpl::list<B, A>;
/// \endcode
template <typename Lhs, typename Rhs>
using rhs_elements_in_lhs =
    tmpl::fold<Rhs, tmpl::list<>,
               detail::rhs_elements_in_lhs_helper<tmpl::_state, tmpl::_element,
                                                  tmpl::pin<Lhs>>>;

namespace detail {
template <typename Element, typename Iteration, typename Lhs, typename Rhs,
          typename RhsWithOnlyLhs, typename IndexInLhs>
struct generate_transformation_helper {
  using tensor_index_to_find = tmpl::at<RhsWithOnlyLhs, IndexInLhs>;
  using index_to_replace_with = tmpl::index_of<Rhs, tensor_index_to_find>;
  using type = tmpl::size_t<index_to_replace_with::value>;
};

template <typename Element, typename Iteration, typename Lhs, typename Rhs,
          typename RhsWithOnlyLhs>
struct generate_transformation_helper<Element, Iteration, Lhs, Rhs,
                                      RhsWithOnlyLhs, tmpl::no_such_type_> {
  using type = tmpl::size_t<Iteration::value>;
};

template <typename State, typename Element, typename Iteration, typename Lhs,
          typename Rhs, typename RhsWithOnlyLhs>
struct generate_transformation_impl {
  using index_in_lhs = tmpl::index_of<Lhs, Element>;
  using type = tmpl::push_back<State, typename generate_transformation_helper<
                                          Element, Iteration, Lhs, Rhs,
                                          RhsWithOnlyLhs, index_in_lhs>::type>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// \brief Generate transformation to account for index order difference in RHS
/// and LHS.
///
/// \details
/// Generates the transformation \f$\mathcal{T}\f$ that rearranges the Tensor
/// index array to account for index order differences between the LHS and RHS
/// of the tensor expression.
///
/// ### Usage
/// For typelists `Rhs`, `Lhs` and `RhsOnyWithLhs`, where `RhsOnlyWithLhs` is
/// the result of the metafunction rhs_elements_in_lhs,
/// \code{.cpp}
/// using result = generate_transformation<Rhs, Lhs, RhsOnlyWithLhs>;
/// \endcode
/// \metareturns
/// typelist
template <typename Rhs, typename Lhs, typename RhsOnyWithLhs>
using generate_transformation = tmpl::enumerated_fold<
    Rhs, tmpl::list<>,
    detail::generate_transformation_impl<tmpl::_state, tmpl::_element, tmpl::_3,
                                         tmpl::pin<Lhs>, tmpl::pin<Rhs>,
                                         tmpl::pin<RhsOnyWithLhs>>>;

/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a TensorExpression
///
/// \details
/// The empty base class that all TensorExpression`s must inherit from.
/// \derivedrequires
/// 1) The args_list will be the sorted args_list received as input
///
/// 2) The tensor indices will be swapped to conform with mathematical notation
struct Expression {};

/// \cond
template <typename DataType, typename Symm, typename IndexList>
class Tensor;
/// \endcond

// @{
/// \ingroup TensorExpressionsGroup
/// \brief The base class all tensor expression implementations derive from
///
/// \tparam Derived the derived class needed for
/// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
/// \tparam DataType the type of the data being stored in the Tensor's
/// \tparam Symm the ::Symmetry of the Derived class
/// \tparam IndexList the list of \ref SpacetimeIndex "TensorIndex"'s
/// \tparam Args the tensor indices, e.g. `_a` and `_b` in `F(_a, _b)`
/// \cond HIDDEN_SYMBOLS
template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args = tmpl::list<>,
          typename ReducedArgs = tmpl::list<>>
struct TensorExpression;
/// \endcond

template <typename Derived, typename DataType, typename Symm,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorExpression<Derived, DataType, Symm, tmpl::list<Indices...>,
                        ArgsList<Args...>> : public Expression {
  static_assert(sizeof...(Args) == 0 or sizeof...(Args) == sizeof...(Indices),
                "the number of Tensor indices must match the number of "
                "components specified in an expression.");
  using type = DataType;
  using symmetry = Symm;
  using index_list = tmpl::list<Indices...>;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  /// Typelist of the tensor indices, e.g. `_a_t` and `_b_t` in `F(_a, _b)`
  using args_list = ArgsList<Args...>;
  using structure = Tensor_detail::Structure<symmetry, Indices...>;

  // @{
  /// If Derived is a TensorExpression, it is casted down to the derived
  /// class. This is enabled by the
  /// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
  ///
  /// Otherwise, it is a Tensor. Since Tensor is not derived from
  /// TensorExpression (because of complications arising from the indices being
  /// part of the expression, specifically Tensor may need to derive off of
  /// hundreds or thousands of base classes, which is not feasible), return a
  /// reference to a TensorExpression, which has a sufficient interface to
  /// evaluate the expression.
  ///
  /// \returns const TensorExpression<Derived, DataType, Symm, IndexList,
  /// ArgsList<Args...>>&
  SPECTRE_ALWAYS_INLINE const auto& operator~() const noexcept {
    if constexpr (tt::is_a_v<Tensor, Derived>) {
      return *this;
    } else {
      return static_cast<const Derived&>(*this);
    }
  }

  // @}

  // @{
  /// \cond HIDDEN_SYMBOLS
  /// \ingroup TensorExpressionsGroup
  /// Helper struct to compute the correct tensor index array from a
  /// typelist of std::integral_constant's indicating the ordering. This is
  /// needed for dealing with expressions such as \f$T_{ab} = F_{ba}\f$ and gets
  /// the ordering on the RHS to be correct compared with where the indices are
  /// on the LHS.
  template <typename U>
  struct ComputeCorrectTensorIndex;

  template <template <typename...> class RedArgsList, typename... RedArgs>
  struct ComputeCorrectTensorIndex<RedArgsList<RedArgs...>> {
    template <typename U, std::size_t Size>
    SPECTRE_ALWAYS_INLINE static constexpr std::array<U, Size> apply(
        const std::array<U, Size>& tensor_index) {
      return std::array<U, Size>{{tensor_index[RedArgs::value]...}};
    }
  };
  /// \endcond
  // @}

  /// \brief return the value of type DataType with tensor index `tensor_index`
  ///
  /// \details
  /// If Derived is a TensorExpression, `tensor_index` is forwarded onto the
  /// concrete derived TensorExpression.
  ///
  /// Otherwise, it is a Tensor, where one big challenge with TensorExpression
  /// implementation is the reordering of the Indices on the RHS and LHS of the
  /// expression. This algorithm implemented in ::rhs_elements_in_lhs and
  /// ::generate_transformation handles the index sorting.
  ///
  /// Here are some examples of what the algorithm does:
  ///
  /// LhsIndices is the desired ordering.
  ///
  /// LHS:
  /// \code
  /// <0, 1>
  /// \endcode
  /// RHS:
  /// \code
  /// <1, 2, 3, 0> -Transform> <3, 1, 2, 0>
  /// \endcode
  ///
  /// LHS:
  /// \code
  /// <0, 1, 2> <a, b, c>
  /// \endcode
  /// RHS:
  /// \code
  /// <2, 0, 1> -Transform> <2 , 1, 0>
  /// \endcode
  ///
  /// Below is pseudo-code of the algorithm written in a non-functional way
  /// \verbatim
  /// for Element in RHS:
  ///   if (Element in LHS):
  ///     index_in_LHS = index_of<LHS, Element>
  ///     tensor_index_to_find = at<RHS_with_only_LHS, index_in_LHS>
  ///     index_to_replace_with = index_of<RHS, tensor_index_to_find>
  ///     T_RHS = push_back<T_RHS, index_to_replace_with>
  ///   else:
  ///     T_RHS = push_back<T_RHS, iteration>
  ///   endif
  /// end for
  /// \endverbatim
  ///
  /// \tparam LhsIndices the tensor indices on the LHS on the expression
  /// \param tensor_index the tensor component to retrieve
  /// \return the value of the DataType of component `tensor_index`
  template <typename... LhsIndices, typename ArrayValueType>
  SPECTRE_ALWAYS_INLINE decltype(auto)
  get(const std::array<ArrayValueType, num_tensor_indices>& tensor_index)
      const noexcept {
    if constexpr (tt::is_a_v<Tensor, Derived>) {
      ASSERT(t_ != nullptr,
             "A TensorExpression that should be holding a pointer to a Tensor "
             "is holding a nullptr.");
      using rhs = args_list;
      // To deal with Tensor products we need the ordering of only the subset of
      // tensor indices present in this term
      using lhs = rhs_elements_in_lhs<rhs, tmpl::list<LhsIndices...>>;
      using rhs_only_with_lhs = rhs_elements_in_lhs<lhs, rhs>;
      using transformation =
          generate_transformation<rhs, lhs, rhs_only_with_lhs>;
      return t_->get(
          ComputeCorrectTensorIndex<transformation>::apply(tensor_index));
    } else {
      ASSERT(t_ == nullptr,
             "A TensorExpression that shouldn't be holding a pointer to a "
             "Tensor is holding one.");
      return (~*this).template get<LhsIndices...>(tensor_index);
    }
  }

  /// \brief Computes the right hand side tensor multi-index that corresponds to
  /// the left hand side tensor multi-index, according to their generic indices
  ///
  /// \details
  /// Given the order of the generic indices for the left hand side (LHS) and
  /// right hand side (RHS) and a specific LHS tensor multi-index, the
  /// computation of the equivalent multi-index for the RHS tensor accounts for
  /// differences in the ordering of the generic indices on the LHS and RHS.
  ///
  /// Here, the elements of `lhs_index_order` and `rhs_index_order` refer to
  /// TensorIndex::values that correspond to generic tensor indices,
  /// `lhs_tensor_multi_index` is a multi-index for the LHS tensor, and the
  /// equivalent RHS tensor multi-index is returned. If we have LHS tensor
  /// \f$L_{ab}\f$, RHS tensor \f$R_{ba}\f$, and the LHS component \f$L_{31}\f$,
  /// the corresponding RHS component is \f$R_{13}\f$.
  ///
  /// Here is an example of what the algorithm does:
  ///
  /// `lhs_index_order`:
  /// \code
  /// [0, 1, 2] // i.e. abc
  /// \endcode
  /// `rhs_index_order`:
  /// \code
  /// [1, 2, 0] // i.e. bca
  /// \endcode
  /// `lhs_tensor_multi_index`:
  /// \code
  /// [4, 0, 3] // i.e. a = 4, b = 0, c = 3
  /// \endcode
  /// returned RHS tensor multi-index:
  /// \code
  /// [0, 3, 4] // i.e. b = 0, c = 3, a = 4
  /// \endcode
  ///
  /// \param lhs_index_order the generic index order of the LHS tensor
  /// \param rhs_index_order the generic index order of the RHS tensor
  /// \param lhs_tensor_multi_index the specific LHS tensor multi-index
  /// \return the RHS tensor multi-index that corresponds to
  /// `lhs_tensor_multi_index`, according to the index orders in
  /// `lhs_index_order` and `rhs_index_order`
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, sizeof...(Indices)>
  compute_rhs_tensor_index(
      const std::array<size_t, sizeof...(Indices)>& lhs_index_order,
      const std::array<size_t, sizeof...(Indices)>& rhs_index_order,
      const std::array<size_t, sizeof...(Indices)>&
          lhs_tensor_multi_index) noexcept {
    std::array<size_t, sizeof...(Indices)> rhs_tensor_multi_index{};
    for (size_t i = 0; i < sizeof...(Indices); ++i) {
      gsl::at(rhs_tensor_multi_index,
              static_cast<unsigned long>(std::distance(
                  rhs_index_order.begin(),
                  alg::find(rhs_index_order, gsl::at(lhs_index_order, i))))) =
          gsl::at(lhs_tensor_multi_index, i);
    }
    return rhs_tensor_multi_index;
  }

  /// \brief Computes a mapping from the storage indices of the left hand side
  /// tensor to the right hand side tensor
  ///
  /// \tparam LhsStructure the Structure of the Tensor on the left hand side of
  /// the TensorExpression
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the left hand side
  /// \return the mapping from the left hand side to the right hand side storage
  /// indices
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
                                                    LhsStructure::size()>
  compute_lhs_to_rhs_map() noexcept {
    constexpr size_t num_components = LhsStructure::size();
    std::array<size_t, num_components> lhs_to_rhs_map{};
    const auto lhs_storage_to_tensor_indices =
        LhsStructure::storage_to_tensor_index();
    for (size_t lhs_storage_index = 0; lhs_storage_index < num_components;
         ++lhs_storage_index) {
      // `compute_rhs_tensor_index` will return the RHS tensor multi-index that
      // corresponds to the LHS tensor multi-index, according to the order of
      // the generic indices for the LHS and RHS. structure::get_storage_index
      // will then get the RHS storage index that corresponds to this RHS
      // tensor multi-index.
      gsl::at(lhs_to_rhs_map, lhs_storage_index) =
          structure::get_storage_index(compute_rhs_tensor_index(
              {{LhsIndices::value...}}, {{Args::value...}},
              lhs_storage_to_tensor_indices[lhs_storage_index]));
    }
    return lhs_to_rhs_map;
  }

  /// \brief return the value at a left hand side tensor's storage index
  ///
  /// \details
  /// If Derived is a TensorExpression, `storage_index` is forwarded onto the
  /// concrete derived TensorExpression.
  ///
  /// Otherwise, it is a Tensor, where one big challenge with TensorExpression
  /// implementation is the reordering of the indices on the left hand side
  /// (LHS) and right hand side (RHS) of the expression. The algorithms
  /// implemented in `compute_lhs_to_rhs_map` and `compute_rhs_tensor_index`
  /// handle the index sorting by mapping between the generic index orders of
  /// the LHS and RHS.
  ///
  /// \tparam LhsStructure the Structure of the Tensor on the LHS of the
  /// TensorExpression
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the LHS of the tensor
  /// expression
  /// \param lhs_storage_index the storage index of the LHS tensor component to
  /// retrieve
  /// \return the value of the DataType of the component at `lhs_storage_index`
  /// in the LHS tensor
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto)
  get(const size_t lhs_storage_index) const noexcept {
    if constexpr (not tt::is_a_v<Tensor, Derived>) {
      return static_cast<const Derived&>(*this)
          .template get<LhsStructure, LhsIndices...>(lhs_storage_index);
    } else if constexpr (std::is_same_v<LhsStructure, structure> and
                         std::is_same_v<tmpl::list<LhsIndices...>,
                                        tmpl::list<Args...>>) {
      // the LHS and RHS tensors have the same structure and generic index
      // order, so the RHS storage index is equivalent to the LHS storage index
      return (*t_)[lhs_storage_index];
    } else {
      // the LHS and RHS tensors do not have the same structure or generic index
      // order, so we must map the LHS storage index to its corresponding RHS
      // storage index
      constexpr std::array<size_t, LhsStructure::size()> lhs_to_rhs_map =
          compute_lhs_to_rhs_map<LhsStructure, LhsIndices...>();
      return (*t_)[gsl::at(lhs_to_rhs_map, lhs_storage_index)];
    }
  }

  /// Retrieve the i'th entry of the Tensor being held
  template <typename V = Derived,
            Requires<tt::is_a<Tensor, V>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE type operator[](const size_t i) const {
    return t_->operator[](i);
  }

  /// \brief Construct a TensorExpression from another TensorExpression.
  ///
  /// In this case we do not need to store a pointer to the TensorExpression
  /// since we can cast back to the derived class using operator~.
  template <typename V = Derived,
            Requires<not tt::is_a<Tensor, V>::value> = nullptr>
  TensorExpression() {}  // NOLINT

  /// \brief Construct a TensorExpression from a Tensor.
  ///
  /// We need to store a pointer to the Tensor in a member variable in order
  /// to be able to access the data when later evaluating the tensor expression.
  explicit TensorExpression(const Tensor<DataType, Symm, index_list>& t)
      : t_(&t) {}

 private:
  /// Holds a pointer to a Tensor if the TensorExpression represents one.
  ///
  /// The pointer is needed so that the Tensor class need not derive from
  /// TensorExpression. The reason deriving off of TensorExpression is
  /// problematic for Tensor is that the index structure is part of the type
  /// of the TensorExpression, so every possible permutation and combination of
  /// indices must be derived from. For a rank-3 tensor this is already over 500
  /// base classes, which the Intel compiler takes too long to compile.
  ///
  /// Benchmarking shows that GCC 6 and Clang 3.9.0 can derive off of 672 base
  /// classes with compilation time of about 5 seconds, while the Intel compiler
  /// v16.3 takes around 8 minutes. These tests were done on a Haswell Core i5.
  const Derived* t_ = nullptr;
};
// @}
