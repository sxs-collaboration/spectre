// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
auto PrimitiveGhostVariables::apply(
    const Variables<hydro::grmhd_tags<DataVector>>& prims,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi)
    -> Variables<tags_for_reconstruction> {
  Variables<tags_for_reconstruction> vars_to_reconstruct(
      prims.number_of_grid_points());
  get<hydro::Tags::RestMassDensity<DataVector>>(vars_to_reconstruct) =
      get<hydro::Tags::RestMassDensity<DataVector>>(prims);
  get<hydro::Tags::ElectronFraction<DataVector>>(vars_to_reconstruct) =
      get<hydro::Tags::ElectronFraction<DataVector>>(prims);
  get<hydro::Tags::Pressure<DataVector>>(vars_to_reconstruct) =
      get<hydro::Tags::Pressure<DataVector>>(prims);
  get<hydro::Tags::MagneticField<DataVector, 3>>(vars_to_reconstruct) =
      get<hydro::Tags::MagneticField<DataVector, 3>>(prims);
  get<hydro::Tags::DivergenceCleaningField<DataVector>>(vars_to_reconstruct) =
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(prims);

  auto& lorentz_factor_time_spatial_velocity =
      get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
          vars_to_reconstruct) =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prims);
  for (size_t i = 0; i < 3; ++i) {
    lorentz_factor_time_spatial_velocity.get(i) *=
        get(get<hydro::Tags::LorentzFactor<DataVector>>(prims));
  }
  get<gr::Tags::SpacetimeMetric<3>>(vars_to_reconstruct) = spacetime_metric;
  get<GeneralizedHarmonic::Tags::Phi<3>>(vars_to_reconstruct) = phi;
  get<GeneralizedHarmonic::Tags::Pi<3>>(vars_to_reconstruct) = pi;
  return vars_to_reconstruct;
}
}  // namespace grmhd::GhValenciaDivClean::subcell
