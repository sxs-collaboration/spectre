// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Conservative/ConservativeDuDt.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
template <class>
class Variables;
}  // namespace Tags

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
  static constexpr bool is_conservative = true;
  static constexpr size_t volume_dim = 3;

  using primitive_variables_tag = ::Tags::Variables<tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
      hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
      hydro::Tags::DivergenceCleaningField<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>, hydro::Tags::Pressure<DataVector>,
      hydro::Tags::SpecificEnthalpy<DataVector>>>;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<>,
                                   Tags::TildeB<>, Tags::TildePhi>>;

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

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>;

  using char_speeds_tag = ComputeCharacteristicSpeeds<2>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed;

  using conservative_from_primitive = ConservativeFromPrimitive;
  using primitive_from_conservative =
      PrimitiveFromConservative<PrimitiveRecoverySchemes::NewmanHamlin, 2>;

  using volume_fluxes = ComputeFluxes;

  using volume_sources = ComputeSources;

  using du_dt = ConservativeDuDt<System>;

  // skip TildeD as its source is zero.
  using sourced_variables = tmpl::list<Tags::TildeTau, Tags::TildeS<>,
                                       Tags::TildeB<>, Tags::TildePhi>;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
