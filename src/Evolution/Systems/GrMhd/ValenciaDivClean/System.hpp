// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
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

  using primitive_variables_tag = ::Tags::Variables<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>>;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<>,
                                   Tags::TildeB<>, Tags::TildePhi>>;

  using spacetime_variables_tag = ::Tags::Variables<
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                 gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>>;

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>;

  using char_speeds_tag = Tags::CharacteristicSpeeds;

  using conservative_from_primitive = ConservativeFromPrimitive;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
