// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <type_traits>

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

/// \cond
namespace db {
struct PrefixTag;
}  // namespace db
/// \endcond

namespace db {

namespace detail {
CREATE_HAS_TYPE_ALIAS(base)
CREATE_HAS_TYPE_ALIAS_V(base)
CREATE_IS_CALLABLE(name)
CREATE_IS_CALLABLE_V(name)
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get the name of a DataBox tag, including prefixes
 *
 * \details
 * Given a DataBox tag returns the name of the DataBox tag as a std::string. If
 * the DataBox tag is also a PrefixTag then the prefix is added.
 *
 * \tparam Tag the DataBox tag whose name to get
 * \return string holding the DataBox tag's name
 */
template <typename Tag>
std::string tag_name() {
  if constexpr (detail::is_name_callable_v<Tag>) {
    return Tag::name();
  } else if constexpr (detail::has_base_v<Tag>) {
    return tag_name<typename Tag::base>();
  } else if constexpr (std::is_base_of_v<db::PrefixTag, Tag>) {
    return pretty_type::short_name<Tag>() + "(" +
           tag_name<typename Tag::tag>() + ")";
  } else {
    return pretty_type::short_name<Tag>();
  }
}
}  // namespace db
