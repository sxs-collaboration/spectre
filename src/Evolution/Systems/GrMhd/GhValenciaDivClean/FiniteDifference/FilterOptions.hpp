// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief Filtering/dissipation options
 */
struct FilterOptions {
  /// Kreiss-Oliger dissipation applied to metric variables
  ///
  /// Must be positive and less than 1.
  struct SpacetimeDissipation {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help{
        "The amount of Kreiss-Oliger filter dissipation to apply. Must be "
        "positive and less than 1. If 'None' then no dissipation is applied."};
  };
  using options = tmpl::list<SpacetimeDissipation>;

  static constexpr Options::String help{
      "Parameters for controlling filter on the FD grid."};

  FilterOptions() = default;
  explicit FilterOptions(std::optional<double> in_spacetime_dissipation,
                         const Options::Context& context = {});
  void pup(PUP::er& p);

  std::optional<double> spacetime_dissipation = std::nullopt;
};
}  // namespace grmhd::GhValenciaDivClean::fd
