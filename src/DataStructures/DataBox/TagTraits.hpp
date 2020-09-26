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
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` derives off of db::ComputeTag.
 *
 * \see is_compute_tag_v ComputeTag
 */
template <typename Tag>
struct is_compute_tag : std::is_base_of<db::ComputeTag, Tag> {};

/// \ingroup DataBoxGroup
/// \brief True if `Tag` derives from db::ComputeTag.
template <typename Tag>
constexpr bool is_compute_tag_v = is_compute_tag<Tag>::value;

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a simple tag.
 *
 * \details This is done by deriving from std::true_type if `Tag` is derived
 * from db::SimpleTag, but not from db::ComputeTag.
 *
 * \see is_simple_tag_v SimpleTag
 */
template <typename Tag>
struct is_simple_tag
    : std::bool_constant<std::is_base_of_v<db::SimpleTag, Tag> and
                         not is_compute_tag_v<Tag>> {};

/// \ingroup DataBoxGroup
/// \brief True if `Tag` is a simple tag.
template <typename Tag>
constexpr bool is_simple_tag_v = is_simple_tag<Tag>::value;

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is not a base tag.
 *
 * \see is_non_base_tag_v BaseTag
 */
template <typename Tag>
struct is_non_base_tag
    : std::bool_constant<std::is_base_of_v<db::ComputeTag, Tag> or
                         std::is_base_of_v<db::SimpleTag, Tag>> {};

/// \ingroup DataBoxGroup
/// \brief True if `Tag` is not a base tag.
template <typename Tag>
constexpr bool is_non_base_tag_v = is_non_base_tag<Tag>::value;

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a DataBox tag, i.e. a BaseTag, SimpleTag, or
 * ComputeTag
 *
 * \see is_tag_v
 */
template <typename Tag>
struct is_tag : std::bool_constant<std::is_base_of_v<db::ComputeTag, Tag> or
                                   std::is_base_of_v<db::SimpleTag, Tag> or
                                   std::is_base_of_v<db::BaseTag, Tag>> {};

/// \ingroup DataBoxGroup
/// \brief True if `Tag` is a DataBox tag.
template <typename Tag>
constexpr bool is_tag_v = is_tag<Tag>::value;

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a base DataBox tag.
 *
 * \see is_base_tag_v BaseTag
 */
template <typename Tag>
struct is_base_tag
    : std::bool_constant<std::is_base_of_v<db::BaseTag, Tag> and
                         not std::is_base_of_v<db::SimpleTag, Tag> and
                         not is_compute_tag_v<Tag>> {};

/// \ingroup DataBoxGroup
/// \brief True if `Tag` is a base tag.
template <typename Tag>
constexpr bool is_base_tag_v = is_base_tag<Tag>::value;

}  // namespace db
