// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/VariablesTag.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
template <typename TagsList>
class Variables;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Mark a tag as the being on the subcell grid
template <typename Tag>
struct OnSubcells : db::PrefixTag, db::SimpleTag {
  static_assert(tt::is_a_v<Tensor, typename Tag::type>,
                "A subcell tag must be either a Tensor or a Variables "
                "currently, though this can be generalized if needed.");
  using type = typename Tag::type;
  using tag = Tag;
};

/// \cond
template <typename TagList>
struct OnSubcells<::Tags::Variables<TagList>> : db::PrefixTag,
                                                   db::SimpleTag {
 private:
  using wrapped_tags_list = db::wrap_tags_in<OnSubcells, TagList>;

 public:
  using tag = ::Tags::Variables<TagList>;
  using type = Variables<wrapped_tags_list>;
};
/// \endcond
}  // namespace evolution::dg::subcell::Tags
