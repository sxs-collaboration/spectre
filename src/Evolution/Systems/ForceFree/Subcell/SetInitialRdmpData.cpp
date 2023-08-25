// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/SetInitialRdmpData.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"

namespace ForceFree::subcell {

void SetInitialRdmpData::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_q,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    const Scalar<DataVector> tilde_e_magnitude = magnitude(tilde_e);
    const Scalar<DataVector> tilde_b_magnitude = magnitude(tilde_b);

    rdmp_tci_data->max_variables_values =
        DataVector{max(get(tilde_e_magnitude)), max(get(tilde_b_magnitude)),
                   max(get(tilde_q))};
    rdmp_tci_data->min_variables_values =
        DataVector{min(get(tilde_e_magnitude)), min(get(tilde_b_magnitude)),
                   min(get(tilde_q))};
  } else {
    using std::max;
    using std::min;
    const Scalar<DataVector> tilde_e_magnitude = magnitude(tilde_e);
    const Scalar<DataVector> tilde_b_magnitude = magnitude(tilde_b);
    const auto subcell_tilde_e_mag = evolution::dg::subcell::fd::project(
        get(tilde_e_magnitude), dg_mesh, subcell_mesh.extents());
    const auto subcell_tilde_b_mag = evolution::dg::subcell::fd::project(
        get(tilde_b_magnitude), dg_mesh, subcell_mesh.extents());
    const auto subcell_tilde_q = evolution::dg::subcell::fd::project(
        get(tilde_q), dg_mesh, subcell_mesh.extents());

    rdmp_tci_data->max_variables_values =
        DataVector{max(max(get(tilde_e_magnitude)), max(subcell_tilde_e_mag)),
                   max(max(get(tilde_b_magnitude)), max(subcell_tilde_b_mag)),
                   max(max(get(tilde_q)), max(subcell_tilde_q))};
    rdmp_tci_data->min_variables_values =
        DataVector{min(min(get(tilde_e_magnitude)), min(subcell_tilde_e_mag)),
                   min(min(get(tilde_b_magnitude)), min(subcell_tilde_b_mag)),
                   min(min(get(tilde_q)), min(subcell_tilde_q))};
  }
}

}  // namespace ForceFree::subcell
