// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::dg::subcell {
/// \ingroup DgSubcellGroup
/// The grid that is currently being used for the DG-subcell evolution.
enum class ActiveGrid { Dg, Subcell };

std::ostream& operator<<(std::ostream& os, ActiveGrid active_grid);
}  // namespace evolution::dg::subcell

template <>
struct Options::create_from_yaml<evolution::dg::subcell::ActiveGrid> {
  template <typename Metavariables>
  static evolution::dg::subcell::ActiveGrid create(
      const Options::Option& options) {
    const auto active_grid = options.parse_as<std::string>();
    if (active_grid == "Dg") {
      return evolution::dg::subcell::ActiveGrid::Dg;
    } else if (active_grid == "Subcell") {
      return evolution::dg::subcell::ActiveGrid::Subcell;
    }
    PARSE_ERROR(options.context(), "ActiveGrid must be 'Dg' or 'Subcell'.");
  }
};
