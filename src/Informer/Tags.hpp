// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"

/// \cond
enum class Verbosity;
/// \endcond

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup LoggingGroup
struct Verbosity {
  using type = ::Verbosity;
  static constexpr OptionString help{"Verbosity"};
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup LoggingGroup
/// \brief Tag for putting `::Verbosity` in a DataBox.
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;
  using option_tags = tmpl::list<OptionTags::Verbosity>;

  template <typename Metavariables>
  static ::Verbosity create_from_options(
      const ::Verbosity& verbosity) noexcept {
    return verbosity;
  }
};
}  // namespace Tags
