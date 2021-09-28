// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"

/// \cond
enum class Verbosity;
/// \endcond

/// \ingroup LoggingGroup
/// Items related to logging
namespace logging {
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup LoggingGroup
template <typename OptionsGroup>
struct Verbosity {
  using type = ::Verbosity;
  static constexpr Options::String help{"Verbosity"};
  using group = OptionsGroup;
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup LoggingGroup
/// \brief Tag for putting `::Verbosity` in a DataBox.
template <typename OptionsGroup>
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;
  static std::string name() {
    return "Verbosity(" + Options::name<OptionsGroup>() + ")";
  }

  using option_tags = tmpl::list<OptionTags::Verbosity<OptionsGroup>>;
  static constexpr bool pass_metavariables = false;
  static ::Verbosity create_from_options(const ::Verbosity& verbosity) {
    return verbosity;
  }
};
}  // namespace Tags
}  // namespace logging
