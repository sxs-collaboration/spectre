// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Holds the `OptionTags::SlopeLimiter` option in the input file
 */
struct SlopeLimiterGroup {
  static std::string name() noexcept { return "SlopeLimiter"; }
  static constexpr OptionString help = "Options for limiting troubled cells";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the slope
 * limiter from the input file
 */
template <typename SlopeLimiterType>
struct SlopeLimiter {
  static std::string name() noexcept { return option_name<SlopeLimiterType>(); }
  static constexpr OptionString help = "Options for the slope limiter";
  using type = SlopeLimiterType;
  using group = SlopeLimiterGroup;
};
}  // namespace OptionTags
