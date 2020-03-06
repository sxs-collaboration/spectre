// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace db {
struct PrefixTag;
}  // namespace db
/// \endcond

namespace db {

namespace DataBox_detail {
template <typename Tag, typename = cpp17::void_t<>>
struct tag_name_impl;

template <typename Tag, typename = std::nullptr_t, typename = cpp17::void_t<>>
struct tag_name_impl2 {
  static_assert(not is_compute_item_v<Tag>,
                "Compute tags must have a name function or a base alias.");
  static std::string name() noexcept { return pretty_type::short_name<Tag>(); }
};

template <typename Tag>
struct tag_name_impl2<Tag, Requires<is_compute_item_v<Tag>>,
                      cpp17::void_t<typename Tag::base>>
    : tag_name_impl<typename Tag::base> {};

template <typename Tag>
struct tag_name_impl2<Tag, Requires<cpp17::is_base_of_v<db::PrefixTag, Tag> and
                                    not is_compute_item_v<Tag>>> {
  static std::string name() noexcept {
    return pretty_type::short_name<Tag>() + "(" +
           tag_name_impl<typename Tag::tag>::name() + ")";
  }
};

template <typename Tag, typename>
struct tag_name_impl : tag_name_impl2<Tag> {};

template <typename Tag>
struct tag_name_impl<Tag, cpp17::void_t<decltype(Tag::name())>> : public Tag {};
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get the name of a DataBoxTag, including prefixes
 *
 * \details
 * Given a DataBoxTag returns the name of the DataBoxTag as a std::string. If
 * the DataBoxTag is also a PrefixTag then the prefix is added.
 *
 * \tparam Tag the DataBoxTag whose name to get
 * \return string holding the DataBoxTag's name
 */
template <typename Tag>
std::string tag_name() noexcept {
  return DataBox_detail::tag_name_impl<Tag>::name();
}

}  // namespace db
