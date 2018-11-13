// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"
#include "Utilities/PrettyType.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the variable
 * fixer from the input file.
 */
template <typename VariableFixerType>
struct VariableFixerParams {
  static constexpr OptionString help = "The options for the variable fixer";
  using type = VariableFixerType;
  static std::string name() noexcept {
    return pretty_type::short_name<VariableFixerType>();
  }
};
}  // namespace OptionTags
