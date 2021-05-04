// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Characteristics.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Sources.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to general relativistic radiation transport
namespace RadiationTransport {
/// The M1 scheme for radiation transport
///
/// References:
/// - Post-merger evolution of a neutron star-black hole binary with
///   neutrino transport, \cite Foucart2015vpa
namespace M1Grey {

template <typename NeutrinoSpeciesList>
struct System;

template <typename... NeutrinoSpecies>
struct System<tmpl::list<NeutrinoSpecies...>> {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 3;
  // If coupling to hydro, we'll want 3D equations of state
  // i.e. P(rho,T,Ye)... but this is not implemented yet.
  // For early tests of M1, we'll ignore coupling to the fluid
  // and provide analytical expressions for its 4-velocity / LorentzFactor
  //static constexpr size_t thermodynamic_dim = 3;

  using boundary_conditions_base =
      BoundaryConditions::BoundaryCondition<tmpl::list<NeutrinoSpecies...>>;
  using boundary_correction_base =
      BoundaryCorrections::BoundaryCorrection<tmpl::list<NeutrinoSpecies...>>;

  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...>>;
  using flux_variables =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...>;
  using gradient_variables = tmpl::list<>;
  using sourced_variables =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...>;
  using primitive_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::ClosureFactor<NeutrinoSpecies>...,
      Tags::TildeP<Frame::Inertial, NeutrinoSpecies>...,
      Tags::TildeJ<NeutrinoSpecies>..., Tags::TildeHNormal<NeutrinoSpecies>...,
      Tags::TildeHSpatial<Frame::Inertial, NeutrinoSpecies>...,
      Tags::M1HydroCouplingNormal<NeutrinoSpecies>...,
      Tags::M1HydroCouplingSpatial<Frame::Inertial, NeutrinoSpecies>...>>;
  // gr::tags_for_hydro contains all these tags plus SqrtDetSpatialMetric,
  // so it can be used when adding M1 coupling to hydro
  using spacetime_variables_tag = ::Tags::Variables<tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>>;
  using hydro_variables_tag = ::Tags::Variables<
      tmpl::list<hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 Tags::GreyEmissivity<NeutrinoSpecies>...,
                 Tags::GreyAbsorptionOpacity<NeutrinoSpecies>...,
                 Tags::GreyScatteringOpacity<NeutrinoSpecies>...>>;

  using compute_volume_time_derivative_terms =
      TimeDerivativeTerms<NeutrinoSpecies...>;
  using volume_fluxes = ComputeFluxes<NeutrinoSpecies...>;
  using volume_sources = ComputeSources<NeutrinoSpecies...>;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>;
};
}  // namespace M1Grey
}  // namespace RadiationTransport
