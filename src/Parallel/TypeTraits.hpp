// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines type traits related to Charm++ types

#pragma once

#include <charm++.h>
#include <pup.h>

#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Parallel::Algorithms {
struct Array;
struct Singleton;
struct Group;
struct Nodegroup;
}  // namespace Parallel::Algorithms
/// \endcond

namespace Parallel {

/// \ingroup ParallelGroup
/// Check if `T` is a Charm++ proxy for an array chare (the entire array)
template <typename T>
struct is_array_proxy : std::is_base_of<CProxy_ArrayElement, T>::type {};

/// \ingroup ParallelGroup
/// Check if `T` is a Charm++ proxy for an array chare element (from indexing
/// into the array chare)
template <typename T>
struct is_array_element_proxy
    : std::is_base_of<CProxyElement_ArrayElement, T>::type {};

/// \ingroup ParallelGroup
/// Check if `T` is a Charm++ proxy for a group chare
template <typename T>
struct is_group_proxy : std::is_base_of<CProxy_IrrGroup, T>::type {};

/// \ingroup ParallelGroup
/// Check if `T` is a Charm++ proxy for a node group chare
template <typename T>
struct is_node_group_proxy : std::is_base_of<CProxy_NodeGroup, T>::type {};

/// \ingroup ParallelGroup
/// Check if `T` is a ParallelComponent for a Charm++ bound array
template <typename T, typename = std::void_t<>>
struct is_bound_array : std::false_type {};

template <typename T>
struct is_bound_array<T, std::void_t<typename T::bind_to>> : std::true_type {
  static_assert(Parallel::is_array_proxy<typename T::type>::value,
                "Can only bind an array chare");
  static_assert(Parallel::is_array_proxy<typename T::bind_to::type>::value,
                "Can only bind to an array chare");
};

/// \ingroup ParallelGroup
/// @{
/// Check if `T` is a SpECTRE Array
template <typename T>
struct is_array : std::is_same<Parallel::Algorithms::Array,
                               typename T::chare_type::component_type> {};
template <typename T>
constexpr bool is_array_v = is_array<T>::value;
/// @}

/// \ingroup ParallelGroup
/// @{
/// Check if `T` is a SpECTRE Singleton
template <typename T>
struct is_singleton : std::is_same<Parallel::Algorithms::Singleton,
                                   typename T::chare_type::component_type> {};
template <typename T>
constexpr bool is_singleton_v = is_singleton<T>::value;
/// @}

/// \ingroup ParallelGroup
/// @{
/// Check if `T` is a SpECTRE Group
template <typename T>
struct is_group : std::is_same<Parallel::Algorithms::Group,
                               typename T::chare_type::component_type> {};
template <typename T>
constexpr bool is_group_v = is_group<T>::value;
/// @}

/// \ingroup ParallelGroup
/// @{
/// Check if `T` is a SpECTRE Nodegroup
template <typename T>
struct is_nodegroup : std::is_same<Parallel::Algorithms::Nodegroup,
                                   typename T::chare_type::component_type> {};
template <typename T>
constexpr bool is_nodegroup_v = is_nodegroup<T>::value;
/// @}

/// @{
/// \ingroup ParallelGroup
/// Retrieve a parallel component from its proxy.
template <typename Proxy>
struct get_parallel_component_from_proxy;

template <typename ParallelComponent, template <typename...> class Proxy,
          typename... Ts>
struct get_parallel_component_from_proxy<Proxy<ParallelComponent, Ts...>> {
  using type = ParallelComponent;
};
/// @}

/// @{
/// \ingroup ParallelGroup
/// \brief Check if `T` has a `pup` member function
///
/// \details
/// Inherits from std::true_type if the type `T` has a `pup(PUP::er&)` member
/// function, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_pup_member<T>;
/// \endcode
///
/// \metareturns
/// std::bool_constant
///
/// \semantics
/// If the type `T` has a `pup(PUP::er&)` member function, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Parallel/Test_TypeTraits.cpp has_pup_member_example
/// \see is_pupable
/// \tparam T the type to check
template <typename T, typename = std::void_t<>>
struct has_pup_member : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_pup_member<
    T, std::void_t<decltype(std::declval<T>().pup(std::declval<PUP::er&>()))>>
    : std::true_type {};
/// \endcond
/// \see has_pup_member
template <typename T>
constexpr bool has_pup_member_v = has_pup_member<T>::value;

/// \see has_pup_member
template <typename T>
using has_pup_member_t = typename has_pup_member<T>::type;
/// @}

/// @{
/// \ingroup ParallelGroup
/// \brief Check if type `T` has operator| defined for Charm++ serialization
///
/// \details
/// Inherits from std::true_type if the type `T` has operator| defined,
/// otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::is_pupable<T>;
/// \endcode
///
/// \metareturns
/// std::bool_constant
///
/// \semantics
/// If the type `T` has operator| defined, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Parallel/Test_TypeTraits.cpp is_pupable_example
/// \see has_pup_member
/// \tparam T the type to check
template <typename T, typename U = void>
struct is_pupable : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_pupable<
    T, std::void_t<decltype(std::declval<PUP::er&>() | std::declval<T&>())>>
    : std::true_type {};
/// \endcond
/// \see is_pupable
template <typename T>
constexpr bool is_pupable_v = is_pupable<T>::value;

/// \see is_pupable
template <typename T>
using is_pupable_t = typename is_pupable<T>::type;
/// @}

}  // namespace Parallel
