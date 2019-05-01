// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class

#pragma once

#include <array>
#include <cstddef>

#include "ErrorHandling/Assert.hpp"  // IWYU pragma: keep
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Represents the indices in a TensorExpression
 *
 * \details
 * Used to denote a tensor index in a tensor slot. This allows the following
 * type of expressions to work:
 * \code{.cpp}
 * auto T = evaluate<ti_a_t, ti_b_t>(F(ti_a, ti_b) + S(ti_b, ti_a));
 * \endcode
 * where `using ti_a_t = TensorIndex<0>;` and `TensorIndex<0> ti_a;`, that is,
 * `ti_a` and `ti_b` are place holders for objects of type `TensorIndex<0>` and
 * `TensorIndex<1>` respectively.
 */
template <std::size_t I>
struct TensorIndex {
  using value_type = std::size_t;
  using type = TensorIndex<I>;
  static constexpr value_type value = I;
};

/// \ingroup TensorExpressionsGroup
/// Given an `TensorIndex<size_t>` return `TensorIndex<size_t + 1>`
/// \tparam T the args to increment
template <typename T>
using next_tensor_index = TensorIndex<T::value + 1>;

/// \ingroup TensorExpressionsGroup
/// Metafunction to return the sum of two TensorIndex's
template <typename A, typename B>
using plus_tensor_index = TensorIndex<A::value + B::value>;

// @{
/*!
 * \ingroup TensorExpressionsGroup
 * \brief The available TensorIndex's to use in a TensorExpression
 *
 * Available tensor indices to use in a Tensor Expression.
 * \snippet Test_TensorExpressions.cpp use_tensor_index
 */
static TensorIndex<0> ti_a{};
static TensorIndex<0> ti_A{};
static TensorIndex<1> ti_b{};
static TensorIndex<1> ti_B{};
static TensorIndex<2> ti_c{};
static TensorIndex<2> ti_C{};
static TensorIndex<3> ti_d{};
static TensorIndex<3> ti_D{};
static TensorIndex<4> ti_e{};
static TensorIndex<4> ti_E{};
static TensorIndex<5> ti_f{};
static TensorIndex<5> ti_F{};
static TensorIndex<6> ti_g{};
static TensorIndex<6> ti_G{};
static TensorIndex<7> ti_h{};
static TensorIndex<7> ti_H{};
static TensorIndex<8> ti_i{};
static TensorIndex<8> ti_I{};
static TensorIndex<9> ti_j{};
static TensorIndex<9> ti_J{};

using ti_a_t = decltype(ti_a);
using ti_A_t = decltype(ti_A);
using ti_b_t = decltype(ti_b);
using ti_B_t = decltype(ti_B);
using ti_c_t = decltype(ti_c);
using ti_C_t = decltype(ti_C);
using ti_d_t = decltype(ti_d);
using ti_D_t = decltype(ti_D);
using ti_e_t = decltype(ti_e);
using ti_E_t = decltype(ti_E);
using ti_f_t = decltype(ti_f);
using ti_F_t = decltype(ti_F);
using ti_g_t = decltype(ti_g);
using ti_G_t = decltype(ti_G);
using ti_h_t = decltype(ti_h);
using ti_H_t = decltype(ti_H);
using ti_i_t = decltype(ti_i);
using ti_I_t = decltype(ti_I);
using ti_j_t = decltype(ti_j);
using ti_J_t = decltype(ti_J);
// @}

/// \cond HIDDEN_SYMBOLS
/// \ingroup TensorExpressionsGroup
/// Type alias used when Tensor Expressions manipulate indices. These are used
/// to denote contracted as opposed to free indices.
template <int I>
using ti_contracted_t = TensorIndex<static_cast<size_t>(I + 1000)>;

/// \ingroup TensorExpressionsGroup
template <int I>
TensorIndex<static_cast<size_t>(I + 1000)> ti_contracted();
/// \endcond

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
  using type = TensorIndex<index_to_replace_with::value>;
};

template <typename Element, typename Iteration, typename Lhs, typename Rhs,
          typename RhsWithOnlyLhs>
struct generate_transformation_helper<Element, Iteration, Lhs, Rhs,
                                      RhsWithOnlyLhs, tmpl::no_such_type_> {
  using type = TensorIndex<Iteration::value>;
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

namespace detail {
template <typename Seq, typename S, typename E>
struct repeated_helper {
  using type = typename std::conditional<
      std::is_same<tmpl::count_if<Seq, std::is_same<E, tmpl::_1>>,
                   tmpl::size_t<2>>::value and
          std::is_same<tmpl::index_of<S, E>, tmpl::no_such_type_>::value,
      tmpl::push_back<S, E>, S>::type;
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * Returns a list of all the types that occurred more than once in List.
 */
template <typename List>
using repeated = tmpl::fold<
    List, tmpl::list<>,
    detail::repeated_helper<tmpl::pin<List>, tmpl::_state, tmpl::_element>>;

namespace detail {
template <typename List, typename Element, typename R>
using index_replace = tmpl::replace_at<
    tmpl::replace_at<List, tmpl::index_of<List, Element>, R>,
    tmpl::index_of<tmpl::replace_at<List, tmpl::index_of<List, Element>, R>,
                   Element>,
    next_tensor_index<R>>;

/// \cond HIDDEN_SYMBOLS
template <typename List, typename ReplaceList, int I>
struct replace_indices_impl
    : replace_indices_impl<
          index_replace<List, tmpl::front<ReplaceList>, ti_contracted_t<2 * I>>,
          tmpl::pop_front<ReplaceList>, I + 1> {};
/// \endcond

template <typename List, int I>
struct replace_indices_impl<List, tmpl::list<>, I> {
  using type = List;
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename List, typename ReplaceList>
using replace_indices =
    typename detail::replace_indices_impl<List, ReplaceList, 0>::type;

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
          typename IndexList, template <typename...> class ArgsList,
          typename... Args>
struct TensorExpression<Derived, DataType, Symm, IndexList, ArgsList<Args...>> {
  static_assert(sizeof...(Args) == 0 or
                    sizeof...(Args) == tmpl::size<IndexList>::value,
                "the number of Tensor indices must match the number of "
                "components specified in an expression.");
  using type = DataType;
  using symmetry = Symm;
  using index_list = IndexList;
  static constexpr auto num_tensor_indices =
      tmpl::size<index_list>::value == 0 ? 1 : tmpl::size<index_list>::value;
  /// Typelist of the tensor indices, e.g. `_a_t` and `_b_t` in `F(_a, _b)`
  using args_list = ArgsList<Args...>;

  // @{
  /// Cast down to the derived class. This is enabled by the
  /// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
  template <typename V = Derived,
            Requires<not tt::is_a<Tensor, V>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE const Derived& operator~() const {
    return static_cast<const Derived&>(*this);
  }
  // @}
  // @{
  /// If the Derived class is a Tensor return a const reference to a
  /// TensorExpression.
  ///
  /// Since Tensor is not derived from TensorExpression (because of
  /// complications arising from the indices being part of the expression,
  /// specifically Tensor may need to derive off of hundreds or thousands of
  /// base classes, which is not feasible), return a reference to a
  /// TensorExpression, which has a sufficient interface to evaluate the
  /// expression.
  /// \returns const TensorExpression<Derived, DataType, Symm, IndexList,
  /// ArgsList<Args...>>&
  template <typename V = Derived,
            Requires<tt::is_a<Tensor, V>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE auto operator~() const {
    return static_cast<const TensorExpression<Derived, DataType, Symm,
                                              IndexList, ArgsList<Args...>>&>(
        *this);
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
  /// _Note:_ This version is selected if Derived is a Tensor
  ///
  /// One big challenge with a general Tensor Expression implementation is the
  /// ordering of the Indices on the RHS and LHS of the expression. This
  /// algorithm implemented in ::rhs_elements_in_lhs and
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
  /// \tparam V used for SFINAE
  /// \param tensor_index the tensor component to retrieve
  /// \return the value of the DataType of component `tensor_index`
  template <typename... LhsIndices, typename ArrayValueType,
            typename V = Derived,
            Requires<tt::is_a<Tensor, V>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<ArrayValueType, num_tensor_indices>& tensor_index)
      const {
    ASSERT(t_ != nullptr,
           "A TensorExpression that should be holding a pointer to a Tensor "
           "is holding a nullptr.");
    using rhs = args_list;
    // To deal with Tensor products we need the ordering of only the subset of
    // tensor indices present in this term
    using lhs = rhs_elements_in_lhs<rhs, tmpl::list<LhsIndices...>>;
    using rhs_only_with_lhs = rhs_elements_in_lhs<lhs, rhs>;
    using transformation = generate_transformation<rhs, lhs, rhs_only_with_lhs>;
    return t_->get(
        ComputeCorrectTensorIndex<transformation>::apply(tensor_index));
  }

  /// \brief return the value of type DataType with tensor index `tensor_index`
  ///
  /// \details
  /// _Note:_ This version is selected if Derived is a TensorExpression
  ///
  /// Forward the tensor_index onwards to the next TensorExpression
  ///
  /// \tparam LhsIndices the tensor indices on the LHS on the expression
  /// \tparam V used for SFINAE
  /// \param tensor_index the tensor component to retrieve
  /// \return the value of the DataType of component `tensor_index`
  template <typename... LhsIndices, typename ArrayValueType,
            typename V = Derived,
            Requires<not tt::is_a<Tensor, V>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<ArrayValueType, num_tensor_indices>& tensor_index)
      const {
    ASSERT(t_ == nullptr,
           "A TensorExpression that shouldn't be holding a pointer to a "
           "Tensor is holding one.");
    return static_cast<const Derived&>(*this).template get<LhsIndices...>(
        tensor_index);
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
  explicit TensorExpression(const Tensor<DataType, Symm, IndexList>& t)
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
