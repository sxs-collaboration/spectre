// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
auto PrimitiveGhostDataOnSubcells::apply(
    const Variables<hydro::grmhd_tags<DataVector>>& prims) noexcept
    -> Variables<prims_to_reconstruct_tags> {
  Variables<prims_to_reconstruct_tags> vars_to_reconstruct(
      prims.number_of_grid_points());
  get<hydro::Tags::RestMassDensity<DataVector>>(vars_to_reconstruct) =
      get<hydro::Tags::RestMassDensity<DataVector>>(prims);
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
  return vars_to_reconstruct;
}

auto PrimitiveGhostDataToSlice::apply(
    const Variables<hydro::grmhd_tags<DataVector>>& prims,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh) noexcept
    -> Variables<prims_to_reconstruct_tags> {
  return evolution::dg::subcell::fd::project(
      PrimitiveGhostDataOnSubcells::apply(prims), dg_mesh,
      subcell_mesh.extents());
}
}  // namespace grmhd::ValenciaDivClean::subcell
