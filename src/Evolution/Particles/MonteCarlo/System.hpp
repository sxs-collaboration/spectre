// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

namespace Particles::MonteCarlo {

struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 3;
  // The EoS is used within the MC code itself, even if we provide
  // the fluid variables as a background.
  static constexpr size_t thermodynamic_dim = 3;

  using mc_variables_tag = ::Tags::Variables<
      tmpl::list<Particles::MonteCarlo::Tags::PacketsOnElement>>;
  using variables_tag = ::Tags::Variables<tmpl::list<>>;  // mc_variables_tag;
  using flux_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  // GR tags needed for background metric
  using spacetime_variables_tag =
      ::Tags::Variables<gr::tags_for_hydro<volume_dim, DataVector>>;
  using flux_spacetime_variables_tag = ::Tags::Variables<tmpl::list<>>;
  // Hydro tags needed for background fluid
  using hydro_variables_tag = ::Tags::Variables<hydro::grmhd_tags<DataVector>>;
  using primitive_variables_tag = hydro_variables_tag;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, 3>;
};
}  // namespace Particles::MonteCarlo
