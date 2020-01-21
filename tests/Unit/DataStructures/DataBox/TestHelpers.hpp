// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Utilities/TypeTraits.hpp"

namespace TestHelpers {

namespace db {

namespace detail {
CREATE_IS_CALLABLE(name)
}  // namespace detail

template <typename Tag>
void test_simple_tag(const std::string& expected_name) {
  static_assert(cpp17::is_base_of_v<::db::SimpleTag, Tag> and
                    not cpp17::is_base_of_v<::db::ComputeTag, Tag>,
                "A simple tag must be derived from "
                "db::SimpleTag, but not db::ComputeTag");
  static_assert(not cpp17::is_same_v<Tag, typename Tag::type>,
                "A type cannot be its own tag.");
  CHECK(::db::tag_name<Tag>() == expected_name);
  if (detail::is_name_callable_v<Tag>) {
    INFO("Do not define name for Tag '" << ::db::tag_name<Tag>() << "',");
    INFO("as it will automatically be generated with that name.");
    CHECK(::db::tag_name<Tag>() != pretty_type::short_name<Tag>());
  }
}

}  // namespace db

}  // namespace TestHelpers
