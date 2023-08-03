// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/SetInitialRdmpData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace grmhd::ValenciaDivClean::subcell {
void SetInitialRdmpData::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    const Scalar<DataVector> tilde_b_magnitude = magnitude(tilde_b);

    rdmp_tci_data->max_variables_values =
        DataVector{max(get(tilde_d)), max(get(tilde_ye)), max(get(tilde_tau)),
                   max(get(tilde_b_magnitude))};
    rdmp_tci_data->min_variables_values =
        DataVector{min(get(tilde_d)), min(get(tilde_ye)), min(get(tilde_tau)),
                   min(get(tilde_b_magnitude))};
  } else {
    const Scalar<DataVector> tilde_b_magnitude = magnitude(tilde_b);
    const auto subcell_tilde_b_mag = evolution::dg::subcell::fd::project(
        get(tilde_b_magnitude), dg_mesh, subcell_mesh.extents());
    const auto subcell_tilde_d = evolution::dg::subcell::fd::project(
        get(tilde_d), dg_mesh, subcell_mesh.extents());
    const auto subcell_tilde_ye = evolution::dg::subcell::fd::project(
        get(tilde_ye), dg_mesh, subcell_mesh.extents());
    const auto subcell_tilde_tau = evolution::dg::subcell::fd::project(
        get(tilde_tau), dg_mesh, subcell_mesh.extents());

    using std::max;
    using std::min;
    rdmp_tci_data->max_variables_values =
        DataVector{max(max(subcell_tilde_d), max(get(tilde_d))),
                   max(max(subcell_tilde_ye), max(get(tilde_ye))),
                   max(max(subcell_tilde_tau), max(get(tilde_tau))),
                   max(max(subcell_tilde_b_mag), max(get(tilde_b_magnitude)))};
    rdmp_tci_data->min_variables_values =
        DataVector{min(min(subcell_tilde_d), min(get(tilde_d))),
                   min(min(subcell_tilde_ye), min(get(tilde_ye))),
                   min(min(subcell_tilde_tau), min(get(tilde_tau))),
                   min(min(subcell_tilde_b_mag), min(get(tilde_b_magnitude)))};
  }
}
}  // namespace grmhd::ValenciaDivClean::subcell
