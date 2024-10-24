// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace detail {
template <typename System, typename = std::void_t<>>
struct fluxes_computer_linearized {
  using type = typename System::fluxes_computer;
};
template <typename System>
struct fluxes_computer_linearized<
    System, std::void_t<typename System::fluxes_computer_linearized>> {
  using type = typename System::fluxes_computer_linearized;
};
}  // namespace detail

/// The `System::fluxes_computer` or the `System::fluxes_computer_linearized`,
/// depending on the `Linearized` parameter. If the system has no
/// `fluxes_computer_linearized` alias it is assumed that the linear flux is
/// functionally identical to the non-linear flux, so the
/// `System::fluxes_computer` is returned either way.
template <typename System, bool Linearized>
using get_fluxes_computer = tmpl::conditional_t<
    Linearized, typename detail::fluxes_computer_linearized<System>::type,
    typename System::fluxes_computer>;

/// The `argument_tags` of either the `System::fluxes_computer` or the
/// `System::fluxes_computer_linearized`, depending on the `Linearized`
/// parameter.
template <typename System, bool Linearized>
using get_fluxes_argument_tags =
    typename get_fluxes_computer<System, Linearized>::argument_tags;

/// The `volume_tags` of either the `System::fluxes_computer` or the
/// `System::fluxes_computer_linearized`, depending on the `Linearized`
/// parameter.
template <typename System, bool Linearized>
using get_fluxes_volume_tags =
    typename get_fluxes_computer<System, Linearized>::volume_tags;

/// The `const_global_cache_tags` of either the `System::fluxes_computer` or the
/// `System::fluxes_computer_linearized`, depending on the `Linearized`
/// parameter.
template <typename System, bool Linearized>
using get_fluxes_const_global_cache_tags =
    typename get_fluxes_computer<System, Linearized>::const_global_cache_tags;
}  // namespace elliptic
