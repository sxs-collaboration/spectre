// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"
#include "Utilities/PrettyType.hpp"

namespace Tags {
/*!
 * \brief The variable fixer in the DataBox
 *
 * \see OptionTags::VariableFixer
 */
template <typename VariableFixerType>
struct VariableFixer : db::SimpleTag {
  using type = VariableFixerType;
  static std::string name() noexcept {
    return "VariableFixer(" + pretty_type::short_name<VariableFixerType>() +
           ")";
  }
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the variable fixer configurations in the input file.
 */
struct VariableFixingGroup {
  static std::string name() noexcept { return "VariableFixing"; }
  static constexpr OptionString help = "Options for variable fixing";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the variable
 * fixer from the input file.
 */
template <typename VariableFixerType>
struct VariableFixer {
  static constexpr OptionString help = "Options for the variable fixer";
  using type = VariableFixerType;
  static std::string name() noexcept {
    return pretty_type::short_name<VariableFixerType>();
  }
  using group = VariableFixingGroup;
  using container_tag = Tags::VariableFixer<VariableFixerType>;
};
}  // namespace OptionTags
