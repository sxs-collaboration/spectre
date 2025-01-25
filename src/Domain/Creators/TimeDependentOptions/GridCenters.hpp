// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Class that holds map options from the grid centers location.
 *
 * This is needed in BNS simulations where the stars are not at the centers of
 * the cubes.
 *
 * \details This class can also be used as an option tag with the \p type type
 * alias, `name()` function, and \p help string.
 */
struct GridCentersOptions {
  using type = Options::Auto<GridCentersOptions, Options::AutoLabel::None>;
  static std::string name() { return "GridCenters"; }

  struct SpecEvolutionParametersPerlFile {
    using type = std::string;
    static constexpr Options::String help = {
        "Path to the EvolutionParameters.perl file that SpEC outputs to parse "
        "for the initial positions and velocities."};
  };

  struct ScaleInspiralRateBy {
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "The inspiral rate from the initial data is not exact because of gauge "
        "changes and in general that initial data is not perfect. The control "
        "system can have an easier time if the inspiral rate is scaled in some "
        "cases. You typically shouldn't need to scale by more than a factor of "
        "2 larger or smaller."};
  };

  using options =
      tmpl::list<SpecEvolutionParametersPerlFile, ScaleInspiralRateBy>;
  static constexpr Options::String help = {
      "Sets the initial value of the GridCenters of the objects."};

  GridCentersOptions();
  explicit GridCentersOptions(
      const std::string& spec_evolution_parameters_perl_file,
      std::optional<double> in_scale_inspiral_rate_by,
      const Options::Context& context = {});

  std::array<DataVector, 3> initial_values{};
  std::optional<double> scale_inspiral_rate_by{std::nullopt};
};

/*!
 * \brief Helper function that creates the FunctionOfTime from the center
 * options, initial time, and expiration time.
 */
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_grid_centers(
    const GridCentersOptions& grid_centers_options, double initial_time,
    double expiration_time);
}  // namespace domain::creators::time_dependent_options
