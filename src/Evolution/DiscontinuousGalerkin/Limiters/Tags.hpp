// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Holds the `OptionTags::Limiter` option in the input file
 */
struct LimiterGroup {
  static std::string name() noexcept { return "Limiter"; }
  static constexpr OptionString help = "Options for limiting troubled cells";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the limiter
 * from the input file
 */
template <typename LimiterType>
struct Limiter {
  static std::string name() noexcept { return option_name<LimiterType>(); }
  static constexpr OptionString help = "Options for the limiter";
  using type = LimiterType;
  using group = LimiterGroup;
};
}  // namespace OptionTags
