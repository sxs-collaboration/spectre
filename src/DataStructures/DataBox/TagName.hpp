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
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

/// \cond
namespace db {
struct PrefixTag;
}  // namespace db
/// \endcond

namespace db {

namespace detail {
CREATE_IS_CALLABLE(name)
CREATE_IS_CALLABLE_V(name)
CREATE_HAS_TYPE_ALIAS(base)
CREATE_HAS_TYPE_ALIAS_V(base)
}  // namespace detail

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
  if constexpr (detail::is_name_callable_v<Tag>) {
    return Tag::name();
  } else if constexpr (db::is_compute_tag_v<Tag>) {
    static_assert(detail::has_base_v<Tag>,
                  "Compute tags must have a name function or a base alias");
    return tag_name<typename Tag::base>();
  } else if constexpr (std::is_base_of_v<db::PrefixTag, Tag>) {
    return pretty_type::short_name<Tag>() + "(" +
           tag_name<typename Tag::tag>() + ")";
  } else {
    return pretty_type::short_name<Tag>();
  }
}
}  // namespace db
