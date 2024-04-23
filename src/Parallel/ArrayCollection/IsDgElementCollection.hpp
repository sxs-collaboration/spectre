// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

/// \cond
namespace Parallel {
template <size_t Dim, typename Metavariables, typename PhaseDepActionList>
class DgElementCollection;
}  // namespace Parallel
/// \endcond

namespace Parallel {
/// \brief Is `std::true_type` if `T` is a `Parallel::DgElementCollection`, else
/// is `std::false_type`.
template <typename T>
struct is_dg_element_collection : std::false_type {};

/// \cond
template <size_t Dim, typename Metavariables, typename PhaseDepActionList>
struct is_dg_element_collection<
    DgElementCollection<Dim, Metavariables, PhaseDepActionList>>
    : std::true_type {};
/// \endcond

/// \brief Is `true` if `T` is a `Parallel::DgElementCollection`, else is
/// `false`.
template <class T>
constexpr bool is_dg_element_collection_v = is_dg_element_collection<T>::value;
}  // namespace Parallel
