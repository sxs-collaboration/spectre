// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {

struct SomeType;

void do_something(gsl::not_null<SomeType*> /* return_value */);

namespace db {
namespace Tags {

struct Bad {};

struct Simple : ::db::SimpleTag {
  using type = SomeType;
};

struct Base : ::db::BaseTag {};

struct SimpleWithBase : Base, ::db::SimpleTag {
  using type = SomeType;
};

struct SimpleCompute : Simple, ::db::ComputeTag {
  using base = Simple;
  using return_type = SomeType;
  using argument_tags = tmpl::list<>;
  static constexpr auto function = do_something;
};

struct SimpleWithBaseCompute : SimpleWithBase, ::db::ComputeTag {
  using base = SimpleWithBase;
  using return_type = SomeType;
  using argument_tags = tmpl::list<>;
  static constexpr auto function = do_something;
};

struct ParentTag : ::db::SimpleTag {
  using type = SomeType;
};

struct SimpleReference : Simple, ::db::ReferenceTag {
  using base = Simple;
  using parent_tag = ParentTag;
  static const auto& get(const typename parent_tag::type& /* parent_value */);
  using argument_tags = tmpl::list<parent_tag>;
};

struct SimpleWithBaseReference : SimpleWithBase, ::db::ReferenceTag {
  using base = SimpleWithBase;
  using parent_tag = ParentTag;
  static const auto& get(const typename parent_tag::type& /* parent_value */);
  using argument_tags = tmpl::list<parent_tag>;
};

template <typename Tag>
struct Label : ::db::PrefixTag, ::db::SimpleTag {
  using tag = Tag;
  using type = SomeType;
};


}  // namespace Tags
}  // namespace db
}  // namespace TestHelpers
