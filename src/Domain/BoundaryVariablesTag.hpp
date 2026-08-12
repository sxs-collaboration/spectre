// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/BoundaryVariables.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Tags {
template <size_t Dim, typename TagsList>
struct BoundaryVariables : db::SimpleTag {
  static_assert(
      tt::is_a_v<tmpl::list, TagsList>,
      "The TagsList passed to Tags::BoundaryVariables is not a typelist");
  using tags_list = TagsList;
  using type = ::BoundaryVariables<Dim, TagsList>;
  static std::string name() {
    std::string tag_name{"BoundaryVariables("};
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
