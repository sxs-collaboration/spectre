// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
struct System {
  using flux_variables = tmpl::list<>;
  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 3;
  static constexpr bool is_in_flux_conservative_form = false;
  /* shifting_shift and f should be created from options in the future */
  // whether to add the advective terms in the Gamma-driver condition
  static constexpr bool shifting_shift = false;

  // The free parameter f in the Gamma-driver condition.
  static constexpr double f = 0.75;

  // the order of the following evolved variables is important
  // as it is assumed in the filter
  using variables_tag = ::Tags::Variables<tmpl::list<
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      // gauge variables
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>;

  using variables_tag_list = typename variables_tag::tags_list;

  using gradients_tags = variables_tag_list;
};
}  // namespace Ccz4::fd
