// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the variable fixer configurations in the input file.
 */
struct VariableFixingGroup {
  static std::string name() noexcept { return "VariableFixing"; }
  static constexpr Options::String help = "Options for variable fixing";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the variable
 * fixer from the input file.
 */
template <typename VariableFixerType>
struct VariableFixer {
  static constexpr Options::String help = "Options for the variable fixer";
  using type = VariableFixerType;
  static std::string name() noexcept {
    return Options::name<VariableFixerType>();
  }
  using group = VariableFixingGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The global cache tag for the variable fixer
 */
template <typename VariableFixerType>
struct VariableFixer : db::SimpleTag {
  static std::string name() noexcept { return "VariableFixer"; }
  using type = VariableFixerType;
  using option_tags =
      tmpl::list<::OptionTags::VariableFixer<VariableFixerType>>;

  static constexpr bool pass_metavariables = false;
  static VariableFixerType create_from_options(
      const VariableFixerType& variable_fixer) noexcept {
    return variable_fixer;
  }
};
}  // namespace Tags
