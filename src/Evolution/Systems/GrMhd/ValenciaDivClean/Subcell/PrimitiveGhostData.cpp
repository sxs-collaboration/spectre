// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
DataVector PrimitiveGhostVariables::apply(
    const Variables<hydro::grmhd_tags<DataVector>>& prims,
    const size_t rdmp_size) {
  DataVector buffer{
      prims.number_of_grid_points() *
          Variables<
              prims_to_reconstruct_tags>::number_of_independent_components +
      rdmp_size};
  Variables<prims_to_reconstruct_tags> vars_to_reconstruct(
      buffer.data(), buffer.size() - rdmp_size);
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
  return buffer;
}
}  // namespace grmhd::ValenciaDivClean::subcell
