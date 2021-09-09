// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace gr {
namespace Tags {
/*!
 * \brief The quantity `Tag` scaled by a conformal factor to the given `Power`
 */
template <typename Tag, int Power>
struct Conformal : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
  static constexpr int conformal_factor_power = Power;
};
}  // namespace Tags
}  // namespace gr
