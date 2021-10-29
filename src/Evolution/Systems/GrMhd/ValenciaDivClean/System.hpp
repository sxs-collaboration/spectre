// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to general relativistic magnetohydrodynamics (GRMHD)
namespace grmhd {
/// The Valencia formulation of ideal GRMHD with divergence cleaning.
///
/// References:
/// - [Numerical 3+1 General Relativistic Magnetohydrodynamics: A Local
/// Characteristic Approach](http://iopscience.iop.org/article/10.1086/498238)
/// - [GRHydro: a new open-source general-relativistic magnetohydrodynamics code
/// for the Einstein toolkit]
/// (http://iopscience.iop.org/article/10.1088/0264-9381/31/1/015005)
namespace ValenciaDivClean {

struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = 3;

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<>,
                                   Tags::TildeB<>, Tags::TildePhi>>;
  using flux_variables =
      tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<>, Tags::TildeB<>,
                 Tags::TildePhi>;
  using non_conservative_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  using primitive_variables_tag =
      ::Tags::Variables<hydro::grmhd_tags<DataVector>>;
  using spacetime_variables_tag =
      ::Tags::Variables<gr::tags_for_hydro<volume_dim, DataVector>>;
  using flux_spacetime_variables_tag = ::Tags::Variables<tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<3>, gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms;

  using conservative_from_primitive = ConservativeFromPrimitive;
  template <typename OrderedListOfPrimitiveRecoverySchemes>
  using primitive_from_conservative =
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataVector>;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
