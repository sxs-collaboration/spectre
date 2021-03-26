// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace evolution::dg {
/*!
 * \brief If `Metavars` has a `SubcellOptions` member struct and
 * `SubcellOptions::subcell_enabled` is `true` then inherits from
 * `std::true_type`, otherwise inherits from `std::false_type`.
 *
 * \note This check is intentionally not inside the `DgSubcell` library so that
 * executables that do not use subcell do not need to link against it.
 */
template <typename Metavars, typename = std::void_t<>>
struct using_subcell : std::false_type {};

/// \cond
template <typename Metavars>
struct using_subcell<Metavars, std::void_t<typename Metavars::SubcellOptions>>
    : std::bool_constant<Metavars::SubcellOptions::subcell_enabled> {};
/// \endcond

/*!
 * \brief If `Metavars` has a `SubcellOptions` member struct and
 * `SubcellOptions::subcell_enabled` is `true` then is `true`, otherwise
 * `false`.
 *
 * \note This check is intentionally not inside the `DgSubcell` library so that
 * executables that do not use subcell do not need to link against it.
 */
template <typename Metavars>
constexpr bool using_subcell_v = using_subcell<Metavars>::value;
}  // namespace evolution::dg
