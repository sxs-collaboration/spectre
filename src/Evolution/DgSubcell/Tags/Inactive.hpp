// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"

/// \cond
template <typename TagsList>
class Variables;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Mark a tag as holding data for the inactive grid.
///
/// As an example, if the evolution in the element is currently being done on
/// the DG grid then the subcell evolved variables would be in
/// `Inactive<evolved_vars>`.
template <typename Tag>
struct Inactive : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = typename tag::type;
};

/// \copydoc Inactive
template <typename TagList>
struct Inactive<::Tags::Variables<TagList>> : db::PrefixTag, db::SimpleTag {
  using tag = ::Tags::Variables<TagList>;
  using type = Variables<db::wrap_tags_in<Inactive, TagList>>;
};
}  // namespace evolution::dg::subcell::Tags
