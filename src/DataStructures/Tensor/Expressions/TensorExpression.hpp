// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base class for all tensor expressions

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

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
  SPECTRE_ALWAYS_INLINE const auto& operator~() const {
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
