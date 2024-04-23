// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

/// \cond
namespace Parallel {
template <size_t Dim, typename Metavariables, typename PhaseDepActionList,
          typename SimpleTagsFromOptions>
class DgElementArrayMember;
template <size_t Dim>
class DgElementArrayMemberBase;
}  // namespace Parallel
/// \endcond

namespace Parallel {
/// \brief Is `std::true_type` if `T` is a `Parallel::DgElementArrayMember` or
/// `Parallel::DgElementArrayMemberBase`, else is `std::false_type`.
template <typename T>
struct is_dg_element_array_member : std::false_type {};

/// \cond
template <size_t Dim, typename Metavariables, typename PhaseDepActionList,
          typename SimpleTagsFromOptions>
struct is_dg_element_array_member<DgElementArrayMember<
    Dim, Metavariables, PhaseDepActionList, SimpleTagsFromOptions>>
    : std::true_type {};

template <size_t Dim>
struct is_dg_element_array_member<DgElementArrayMemberBase<Dim>>
    : std::true_type {};
/// \endcond

/// \brief Is `true` if `T` is a `Parallel::DgElementArrayMember` or
/// `Parallel::DgElementArrayMemberBase`, else is `false`.
template <class T>
constexpr bool is_dg_element_array_member_v =
    is_dg_element_array_member<T>::value;
}  // namespace Parallel
