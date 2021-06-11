// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base class for all tensor expressions and the generic tensor indices
/// that they use

#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

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

/// @{
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
/// @}

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

/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a TensorExpression
///
/// \details
/// The empty base class provides a simple means for checking if a type is a
/// TensorExpression.
struct Expression {};

/// @{
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

  virtual ~TensorExpression() = 0;

  /// @{
  /// Derived is casted down to the derived class. This is enabled by the
  /// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
  ///
  /// \returns const TensorExpression<Derived, DataType, Symm, IndexList,
  /// ArgsList<Args...>>&
  SPECTRE_ALWAYS_INLINE const auto& operator~() const noexcept {
      return static_cast<const Derived&>(*this);
  }
  /// @}
};

template <typename Derived, typename DataType, typename Symm,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
TensorExpression<Derived, DataType, Symm, tmpl::list<Indices...>,
                 ArgsList<Args...>>::~TensorExpression() = default;
/// @}
