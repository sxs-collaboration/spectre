// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for working with boost.

#pragma once

#include <array>
#include <boost/none.hpp>
#include <boost/variant.hpp>
#include <cstddef>
#include <initializer_list>
#include <pup.h>
#include <string>
#include <utility>  // IWYU pragma: keep

#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace boost {
template <typename T>
class optional;
}  // namespace boost
/// \endcond

namespace detail {
template <typename Sequence>
struct make_boost_variant_over_impl;

template <template <typename...> class Sequence, typename... Ts>
struct make_boost_variant_over_impl<Sequence<Ts...>> {
  static_assert(not cpp17::disjunction<std::is_same<
                    std::decay_t<std::remove_pointer_t<Ts>>, void>...>::value,
                "Cannot create a boost::variant with a 'void' type.");
  using type = boost::variant<Ts...>;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a boost::variant with all all the types inside the typelist
 * Sequence
 *
 * \metareturns boost::variant of all types inside `Sequence`
 */
template <typename Sequence>
using make_boost_variant_over = typename detail::make_boost_variant_over_impl<
    tmpl::remove_duplicates<Sequence>>::type;

namespace BoostVariant_detail {
// clang-tidy: do not use non-const references
template <class T, class... Ts>
char pup_helper(int& index, PUP::er& p, boost::variant<Ts...>& var,  // NOLINT
                const int send_index) {
  if (index == send_index) {
    if (p.isUnpacking()) {
      T t{};
      p | t;
      var = std::move(t);
    } else {
      p | boost::get<T>(var);
    }
  }
  index++;
  return '0';
}
}  // namespace BoostVariant_detail

namespace PUP {
template <class... Ts>
void pup(er& p, boost::variant<Ts...>& var) noexcept {  // NOLINT
  int index = 0;
  int send_index = var.which();
  p | send_index;
  (void)std::initializer_list<char>{
      ::BoostVariant_detail::pup_helper<Ts>(index, p, var, send_index)...};
}

template <typename... Ts>
inline void operator|(er& p, boost::variant<Ts...>& d) noexcept {  // NOLINT
  pup(p, d);
}
}  // namespace PUP

/*!
 * \ingroup UtilitiesGroup
 * \brief Get the type name of the current state of the boost::variant
 */
template <typename... Ts>
std::string type_of_current_state(
    const boost::variant<Ts...>& variant) noexcept {
  // clang-format off
  // clang-tidy: use gsl::at (we know it'll be in bounds and want fewer
  // includes) clang-format moves the comment to the wrong line
  return std::array<std::string, sizeof...(Ts)>{  // NOLINT
      {pretty_type::get_name<Ts>()...}}[static_cast<size_t>(variant.which())];
  // clang-format on
}

namespace PUP {
template <class T>
void pup(er& p, boost::optional<T>& var) noexcept {  // NOLINT
  bool has_data = var != boost::none;
  p | has_data;
  if (has_data) {
    if (p.isUnpacking()) {
      var.emplace();
    }
    p | *var;
  } else {
    var = boost::none;
  }
}

template <typename T>
inline void operator|(er& p, boost::optional<T>& var) noexcept {  // NOLINT
  pup(p, var);
}
}  // namespace PUP
