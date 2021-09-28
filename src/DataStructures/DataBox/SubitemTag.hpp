// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {

/// \cond
// Declaration of a tag for a subitem
// Unless specialized, it will be the reference tag below,
template <typename Tag, typename ParentTag, typename = std::nullptr_t>
struct Subitem;
/// \endcond

/// \brief a reference tag that refers to a particular Tag that is a subitem of
/// an item tagged with ParentTag
template <typename Tag, typename ParentTag, typename>
struct Subitem : Tag, db::ReferenceTag {
  using base = Tag;
  using parent_tag = ParentTag;
  static const auto& get(const typename parent_tag::type& parent_value) {
    return ::db::Subitems<parent_tag>::template create_compute_item<base>(
        parent_value);
  }
  using argument_tags = tmpl::list<parent_tag>;
};
}  // namespace Tags
