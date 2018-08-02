// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"

/// \cond
enum class Verbosity;
/// \endcond

namespace Tags {
/// \ingroup LoggingGroup
/// \brief Tag for putting 'Verbosity' in a DataBox.
struct Verbosity : db::SimpleTag {
  static std::string name() noexcept { return "Verbosity"; }
  using type = ::Verbosity;
};
}  // namespace Tags
