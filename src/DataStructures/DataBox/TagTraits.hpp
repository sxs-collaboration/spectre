// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace db {
struct SimpleTag;
struct BaseTag;
struct ComputeTag;
}  // namespace db
/// \endcond

namespace db {
// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` derives off of db::ComputeTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_compute_tag : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct is_compute_tag<Tag, Requires<std::is_base_of_v<db::ComputeTag, Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_compute_tag_v = is_compute_tag<Tag>::value;
// @}

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a simple tag.
 */
template <typename Tag, typename = std::nullptr_t>
struct is_simple_tag : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct is_simple_tag<Tag, Requires<std::is_base_of_v<db::SimpleTag, Tag> and
                                   not is_compute_tag_v<Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_simple_tag_v = is_simple_tag<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a non-base DataBox tag. I.e. a SimpleTag or a
 * ComputeTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_non_base_tag : std::false_type {};
/// \cond
template <typename Tag>
struct is_non_base_tag<Tag, Requires<std::is_base_of_v<db::ComputeTag, Tag> or
                                     std::is_base_of_v<db::SimpleTag, Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_non_base_tag_v = is_non_base_tag<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a BaseTag, SimpleTag, or ComputeTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_tag : std::false_type {};
/// \cond
template <typename Tag>
struct is_tag<Tag, Requires<std::is_base_of_v<db::ComputeTag, Tag> or
                            std::is_base_of_v<db::SimpleTag, Tag> or
                            std::is_base_of_v<db::BaseTag, Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_tag_v = is_tag<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a base DataBox tag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_base_tag : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct is_base_tag<Tag, Requires<std::is_base_of_v<db::BaseTag, Tag> and
                                 not std::is_base_of_v<db::SimpleTag, Tag> and
                                 not is_compute_tag_v<Tag>>> : std::true_type {
};
/// \endcond

template <typename Tag>
constexpr bool is_base_tag_v = is_base_tag<Tag>::value;
// @}

}  // namespace db
