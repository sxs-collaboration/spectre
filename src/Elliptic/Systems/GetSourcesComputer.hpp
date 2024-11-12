// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace detail {
template <typename System, typename = std::void_t<>>
struct sources_computer_linearized {
  using type = typename System::sources_computer;
};
template <typename System>
struct sources_computer_linearized<
    System, std::void_t<typename System::sources_computer_linearized>> {
  using type = typename System::sources_computer_linearized;
};
struct NoSourcesComputer {
  using argument_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
};
}  // namespace detail

/// The `System::sources_computer` or the `System::sources_computer_linearized`,
/// depending on the `Linearized` parameter. If the system has no
/// `sources_computer_linearized` alias it is assumed to be linear, so the
/// `System::sources_computer` is returned either way.
template <typename System, bool Linearized>
using get_sources_computer = tmpl::conditional_t<
    Linearized, typename detail::sources_computer_linearized<System>::type,
    typename System::sources_computer>;

/// The `argument_tags` of either the `System::sources_computer` or the
/// `System::sources_computer_linearized`, depending on the `Linearized`
/// parameter, or an empty list if the sources computer is `void`.
template <typename System, bool Linearized>
using get_sources_argument_tags = typename tmpl::conditional_t<
    std::is_same_v<get_sources_computer<System, Linearized>, void>,
    detail::NoSourcesComputer,
    get_sources_computer<System, Linearized>>::argument_tags;

/// The `const_global_cache_tags` of either the `System::sources_computer` or
/// the `System::sources_computer_linearized`, depending on the `Linearized`
/// parameter, or an empty list if the sources computer is `void`.
template <typename System, bool Linearized>
using get_sources_const_global_cache_tags = typename tmpl::conditional_t<
    std::is_same_v<get_sources_computer<System, Linearized>, void>,
    detail::NoSourcesComputer,
    get_sources_computer<System, Linearized>>::const_global_cache_tags;
}  // namespace elliptic
