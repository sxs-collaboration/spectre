// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/ActiveGrid.hpp"

#include <ostream>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::dg::subcell {
std::ostream& operator<<(std::ostream& os, ActiveGrid active_grid) {
  switch (active_grid) {
    case ActiveGrid::Dg:
      return os << "Dg";
    case ActiveGrid::Subcell:
      return os << "Subcell";
    default:
      ERROR("ActiveGrid must be either 'Dg' or 'Subcell' but has value "
            << static_cast<int>(active_grid));
  }
}
}  // namespace evolution::dg::subcell

template <>
evolution::dg::subcell::ActiveGrid
Options::create_from_yaml<evolution::dg::subcell::ActiveGrid>::create<void>(
    const Options::Option& options) {
  const auto active_grid = options.parse_as<std::string>();
  if (active_grid == "Dg") {
    return evolution::dg::subcell::ActiveGrid::Dg;
  } else if (active_grid == "Subcell") {
    return evolution::dg::subcell::ActiveGrid::Subcell;
  }
  PARSE_ERROR(options.context(), "ActiveGrid must be 'Dg' or 'Subcell'.");
}
