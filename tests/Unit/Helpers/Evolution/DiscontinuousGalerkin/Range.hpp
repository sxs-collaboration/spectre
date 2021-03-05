// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

namespace TestHelpers::evolution::dg::Tags {

/// Tag for a `TaggedTuple` that holds the range of validity for the variable
/// associated with `Tag`.
template <typename Tag>
struct Range {
  using tag = Tag;
  using type = std::array<double, 2>;
};

}  // namespace TestHelpers::evolution::dg::Tags
