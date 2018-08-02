// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"

/// \cond
enum class Verbosity;
/// \endcond

namespace Tags {
/// \ingroup LoggingGroup
/// \brief Tag for putting `::Verbosity` in a DataBox.
struct Verbosity : db::SimpleTag {
  static std::string name() noexcept { return "Verbosity"; }
  using type = ::Verbosity;
};
}  // namespace Tags

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup LoggingGroup
struct Verbosity {
  using type = ::Verbosity;
  static constexpr OptionString help{"Verbosity"};
};
}  // namespace OptionTags
