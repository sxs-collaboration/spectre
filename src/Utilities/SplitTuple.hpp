// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace split_tuple_detail {
template <size_t Offset, size_t... Indices, typename... TupleTypes>
constexpr std::tuple<
    std::tuple_element_t<Offset + Indices, std::tuple<TupleTypes...>>...>
extract_subset(
    std::index_sequence<Indices...> /*meta*/,
    [[maybe_unused]] const gsl::not_null<std::tuple<TupleTypes...>*> tuple) {
  return {std::forward<
      std::tuple_element_t<Offset + Indices, std::tuple<TupleTypes...>>>(
      std::get<Offset + Indices>(*tuple))...};
}

template <typename... Offsets, typename... Sizes, typename... TupleTypes>
constexpr auto impl(tmpl::list<Offsets...> /*meta*/,
                    tmpl::list<Sizes...> /*meta*/,
                    [[maybe_unused]] std::tuple<TupleTypes...> tuple) {
  static_assert((0 + ... + Sizes::value) == sizeof...(TupleTypes),
                "Tuple size does not match output sizes.");
  return std::make_tuple(extract_subset<Offsets::value>(
      std::make_index_sequence<Sizes::value>{}, make_not_null(&tuple))...);
}
}  // namespace split_tuple_detail

/// \ingroup UtilitiesGroup
/// Split a `std::tuple` into multiple tuples
///
/// \note There are two functions with this name, but doxygen can't
/// figure that out.  They have signatures
/// ```
/// template <typename SizeList, typename... TupleTypes>
/// constexpr auto split_tuple(std::tuple<TupleTypes...> tuple);
/// template <size_t... Sizes, typename... TupleTypes>
/// constexpr auto split_tuple(std::tuple<TupleTypes...> tuple);
/// ```
///
/// Given a list of sizes, either directly as template parameters or
/// as a typelist of integral constant types, split the passed tuple
/// into pieces containing the specified number of entries.  The
/// passed sizes must sum to the size of the tuple.
///
/// \returns a `std::tuple` of `std::tuple`s
///
/// \see std::tuple_cat for the inverse operation.
///
/// \snippet Utilities/Test_SplitTuple.cpp split_tuple
/// @{
template <typename SizeList, typename... TupleTypes>
constexpr auto split_tuple(std::tuple<TupleTypes...> tuple) {
  using offsets = tmpl::pop_back<
      tmpl::fold<SizeList, tmpl::list<tmpl::size_t<0>>,
                 tmpl::bind<tmpl::push_back, tmpl::_state,
                            tmpl::plus<tmpl::bind<tmpl::back, tmpl::_state>,
                                       tmpl::_element>>>>;
  return split_tuple_detail::impl(offsets{}, SizeList{}, std::move(tuple));
}

template <size_t... Sizes, typename... TupleTypes>
constexpr auto split_tuple(std::tuple<TupleTypes...> tuple) {
  return split_tuple<tmpl::list<tmpl::size_t<Sizes>...>>(std::move(tuple));
}
/// @}
