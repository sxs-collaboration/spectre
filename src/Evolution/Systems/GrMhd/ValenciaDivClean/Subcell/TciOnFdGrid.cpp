// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnFdGrid.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace grmhd::ValenciaDivClean::subcell {
bool TciOnFdGrid::apply(const Scalar<DataVector>& tilde_d,
                        const Scalar<DataVector>& tilde_tau,
                        const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
                        const bool vars_needed_fixing, const Mesh<3>& dg_mesh,
                        const TciOptions& tci_options,
                        const double persson_exponent) {
  bool cell_is_troubled =
      vars_needed_fixing or
      min(get(tilde_d)) <
          tci_options.minimum_rest_mass_density_times_lorentz_factor or
      min(get(tilde_tau)) < tci_options.minimum_tilde_tau or
      evolution::dg::subcell::persson_tci(tilde_d, dg_mesh, persson_exponent) or
      evolution::dg::subcell::persson_tci(tilde_tau, dg_mesh, persson_exponent);
  if (tci_options.magnetic_field_cutoff.has_value() and not cell_is_troubled) {
    const Scalar<DataVector> tilde_b_magnitude = magnitude(tilde_b);
    cell_is_troubled = (max(get(tilde_b_magnitude)) >
                            tci_options.magnetic_field_cutoff.value() and
                        evolution::dg::subcell::persson_tci(
                            tilde_b_magnitude, dg_mesh, persson_exponent));
  }
  return cell_is_troubled;
}
}  // namespace grmhd::ValenciaDivClean::subcell
