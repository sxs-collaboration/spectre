// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <typename TagsList>
class Variables;
/// \endcond

namespace Tags {
template <typename TagsList>
struct Variables : db::SimpleTag {
  static_assert(tt::is_a<tmpl::list, TagsList>::value,
                "The TagsList passed to Tags::Variables is not a typelist");
  using tags_list = TagsList;
  using type = ::Variables<TagsList>;
  static std::string name() {
    std::string tag_name{"Variables("};
    size_t iter = 0;
    tmpl::for_each<TagsList>([&tag_name, &iter](auto tag) {
      tag_name += db::tag_name<tmpl::type_from<decltype(tag)>>();
      if (iter + 1 != tmpl::size<TagsList>::value) {
        tag_name += ",";
      }
      iter++;
    });
    return tag_name + ")";
  }
};
}  // namespace Tags
