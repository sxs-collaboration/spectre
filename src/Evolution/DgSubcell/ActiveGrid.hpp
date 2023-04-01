// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

#include "Utilities/ErrorHandling/Error.hpp"

/// \cond
namespace Options {
struct Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

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
    return create<void>(options);
  }
};

template <>
evolution::dg::subcell::ActiveGrid
Options::create_from_yaml<evolution::dg::subcell::ActiveGrid>::create<void>(
    const Options::Option& options);
