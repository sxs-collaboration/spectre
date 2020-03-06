// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {

struct SomeType;

SomeType do_something();

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

struct Compute : ::db::ComputeTag {
  using argument_list = tmpl::list<>;
  static constexpr auto function = do_something;
  static std::string name() noexcept { return "Compute"; }
};

struct SimpleCompute : Simple, ::db::ComputeTag {
  using base = Simple;
  using argument_list = tmpl::list<>;
  static constexpr auto function = do_something;
};

struct BaseCompute : Base, ::db::ComputeTag {
  using base = Base;
  using argument_list = tmpl::list<>;
  static constexpr auto function = do_something;
};

struct SimpleWithBaseCompute : SimpleWithBase, ::db::ComputeTag {
  using base = SimpleWithBase;
  using argument_list = tmpl::list<>;
  static constexpr auto function = do_something;
};

template <typename Tag>
struct Label : ::db::PrefixTag, ::db::SimpleTag {
  using tag = Tag;
  using type = SomeType;
};

template <typename Tag>
struct Operator : ::db::PrefixTag, ::db::ComputeTag {
  using tag = Tag;
  using argument_list = tmpl::list<>;

  static std::string name() noexcept {
    return "Operator(" + ::db::tag_name<Tag>() + ")";
  }
  static constexpr auto function = do_something;
};

template <typename Tag>
struct LabelCompute : Label<Tag>, ::db::ComputeTag {
  using base = Label<Tag>;
  using tag = Tag;
  using argument_list = tmpl::list<>;
  static constexpr auto function = do_something;
};

}  // namespace Tags
}  // namespace db
}  // namespace TestHelpers
