// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace TestHelpers {

namespace db {

namespace detail {
CREATE_HAS_TYPE_ALIAS(argument_tags)
CREATE_HAS_TYPE_ALIAS_V(argument_tags)
CREATE_HAS_TYPE_ALIAS(base)
CREATE_HAS_TYPE_ALIAS_V(base)
CREATE_HAS_TYPE_ALIAS(return_type)
CREATE_HAS_TYPE_ALIAS_V(return_type)

CREATE_IS_CALLABLE(name)
CREATE_IS_CALLABLE_V(name)

template <typename Tag>
void check_tag_name(const std::string& expected_name) {
  CHECK(::db::tag_name<Tag>() == expected_name);
  if (is_name_callable_v<Tag> and not ::db::is_compute_tag_v<Tag>) {
    INFO("Do not define name for Tag '" << ::db::tag_name<Tag>() << "',");
    INFO("as it will automatically be generated with that name.");
    CHECK(::db::tag_name<Tag>() != pretty_type::short_name<Tag>());
  }
}
}  // namespace detail

template <typename Tag>
void test_base_tag(const std::string& expected_name) {
  static_assert(::db::is_base_tag_v<Tag>,
                "A base tag must be derived from db::BaseTag, but "
                "not from db::SimpleTag nor db::ComputeTag");
  detail::check_tag_name<Tag>(expected_name);
}

template <typename Tag>
void test_compute_tag(const std::string& expected_name) {
  static_assert(::db::is_compute_tag_v<Tag>,
                "A compute tag must be derived from db::ComputeTag");
  static_assert(detail::has_return_type_v<Tag>);
  static_assert(detail::has_argument_tags_v<Tag>);
  static_assert(detail::has_base_v<Tag>);
  static_assert(std::is_same_v<typename Tag::return_type, typename Tag::type>);
  detail::check_tag_name<Tag>(expected_name);
}

template <typename Tag>
void test_prefix_tag(const std::string& expected_name) {
  static_assert(std::is_base_of_v<::db::PrefixTag, Tag>,
                "A prefix tag must be derived from db::PrefixTag");
  detail::check_tag_name<Tag>(expected_name);
}

template <typename Tag>
void test_simple_tag(const std::string& expected_name) {
  static_assert(std::is_base_of_v<::db::SimpleTag, Tag> and
                    not std::is_base_of_v<::db::ComputeTag, Tag>,
                "A simple tag must be derived from "
                "db::SimpleTag, but not db::ComputeTag");
  static_assert(not std::is_same_v<Tag, typename Tag::type>,
                "A type cannot be its own tag.");
  detail::check_tag_name<Tag>(expected_name);
}

}  // namespace db

}  // namespace TestHelpers
