// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \file
/// Defines functions and metafunctions used for helping evaluate
/// TensorExpression equations where concrete time indices are used for
/// spacetime indices

namespace tt {
/// \ingroup TypeTraitsGroup TensorExpressionsGroup
/// \brief Check if a type `T` is a TensorIndex representing a concrete time
/// index
template <typename T>
struct is_time_index : std::false_type {};
template <>
struct is_time_index<std::decay_t<decltype(ti_t)>> : std::true_type {};
template <>
struct is_time_index<std::decay_t<decltype(ti_T)>> : std::true_type {};
}  // namespace tt

namespace TensorExpressions {
namespace detail {
/// \brief Returns whether or not the provided value is the TensorIndex value
/// that encodes the upper or lower concrete time index (`ti_T` or `ti_t`)
///
/// \param value the value to check
/// \return whether or not the value encodes the upper or lower concrete time
/// index
constexpr bool is_time_index_value(const size_t value) {
  return value == ti_t.value or value == ti_T.value;
}

template <typename State, typename Element>
struct remove_time_indices_impl {
  using type =
      typename std::conditional_t<not tt::is_time_index<Element>::value,
                                  tmpl::push_back<State, Element>, State>;
};

/// \brief Given a TensorIndex list, returns the TensorIndex list with time
/// indices removed
///
/// \tparam TensorIndexList the generic index list
template <typename TensorIndexList>
struct remove_time_indices {
  using type =
      tmpl::fold<TensorIndexList, tmpl::list<>,
                 remove_time_indices_impl<tmpl::_state, tmpl::_element>>;
};

template <typename State, typename Element, typename Iteration>
struct time_index_positions_impl {
  using type =
      typename std::conditional_t<tt::is_time_index<Element>::value,
                                  tmpl::push_back<State, Iteration>, State>;
};

/// \brief Given a TensorIndex list, returns the list of positions of concrete
/// time indices
///
/// \tparam TensorIndexList the TensorIndex list
template <typename TensorIndexList>
using time_index_positions = tmpl::enumerated_fold<
    TensorIndexList, tmpl::list<>,
    time_index_positions_impl<tmpl::_state, tmpl::_element, tmpl::_3>,
    tmpl::size_t<0>>;

/// \brief Given a TensorIndex list, returns the list of positions of concrete
/// time indices
///
/// \tparam TensorIndexList the TensorIndex list
/// \return the list of positions of concrete time indices
template <typename TensorIndexList>
constexpr auto get_time_index_positions() {
  using time_index_positions_ = time_index_positions<TensorIndexList>;
  using make_list_type =
      std::conditional_t<tmpl::size<time_index_positions_>::value == 0, size_t,
                         time_index_positions_>;
  return make_array_from_list<make_list_type>();
}
}  // namespace detail
}  // namespace TensorExpressions
