// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the slope
 * limiter from the input file
 */
template <typename SlopeLimiterType>
struct SlopeLimiter {
  static constexpr OptionString help = "Options for the slope limiter";
  using type = SlopeLimiterType;
};
}  // namespace OptionTags
