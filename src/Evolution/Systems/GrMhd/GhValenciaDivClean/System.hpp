// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd {

/// Namespace associated with utilities for the combined Generalized Harmonic
/// and Valencia formulation of ideal GRMHD with divergence cleaning systems.
namespace GhValenciaDivClean {
struct System {
  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = 3;
  using grmhd_system = grmhd::ValenciaDivClean::System;
  using gh_system = GeneralizedHarmonic::System<3_st>;

  using variables_tag = ::Tags::Variables<
      tmpl::append<typename gh_system::variables_tag::tags_list,
                   typename grmhd_system::variables_tag::tags_list>>;
  using non_conservative_variables =
      typename gh_system::variables_tag::tags_list;
  using flux_variables = tmpl::append<typename gh_system::flux_variables,
                                      typename grmhd_system::flux_variables>;
  using gradient_variables =
      tmpl::append<typename gh_system::gradient_variables,
                   typename grmhd_system::gradient_variables>;
  using gradients_tags = gradient_variables;
  using sourced_variables = typename grmhd_system::sourced_variables;
  static constexpr bool is_in_flux_conservative_form = false;

  using primitive_variables_tag =
      typename grmhd_system::primitive_variables_tag;
  using spacetime_variables_tag = ::Tags::Variables<tmpl::list<
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms;
  using volume_fluxes = typename grmhd_system::volume_fluxes;
  using volume_sources = typename grmhd_system::volume_sources;

  using conservative_from_primitive =
      typename grmhd_system::conservative_from_primitive;
  template <typename OrderedListOfPrimitiveRecoverySchemes>
  using primitive_from_conservative =
      typename grmhd_system::template primitive_from_conservative<
          OrderedListOfPrimitiveRecoverySchemes>;

  using compute_largest_characteristic_speed =
      gh_system::compute_largest_characteristic_speed;

  using inverse_spatial_metric_tag =
      typename gh_system::inverse_spatial_metric_tag;
};
}  // namespace GhValenciaDivClean
}  // namespace grmhd
