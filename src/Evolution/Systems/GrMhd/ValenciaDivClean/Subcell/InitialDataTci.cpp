// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace grmhd::ValenciaDivClean::subcell {
bool DgInitialDataTci::apply(
    const Variables<tmpl::list<
        ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeTau,
        ValenciaDivClean::Tags::TildeS<>, ValenciaDivClean::Tags::TildeB<>,
        ValenciaDivClean::Tags::TildePhi>>& dg_vars,
    const Variables<tmpl::list<Inactive<ValenciaDivClean::Tags::TildeD>,
                               Inactive<ValenciaDivClean::Tags::TildeTau>,
                               Inactive<ValenciaDivClean::Tags::TildeS<>>,
                               Inactive<ValenciaDivClean::Tags::TildeB<>>,
                               Inactive<ValenciaDivClean::Tags::TildePhi>>>&
        subcell_vars,
    double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
    const Mesh<3>& dg_mesh, const TciOptions& tci_options) noexcept {
  const Scalar<DataVector> tilde_b_magnitude =
      tci_options.magnetic_field_cutoff.has_value()
          ? magnitude(get<ValenciaDivClean::Tags::TildeB<>>(dg_vars))
          : Scalar<DataVector>{};

  return min(get(get<ValenciaDivClean::Tags::TildeD>(dg_vars))) <
             tci_options.minimum_rest_mass_density_times_lorentz_factor or
         min(get(get<Inactive<ValenciaDivClean::Tags::TildeD>>(subcell_vars))) <
             tci_options.minimum_rest_mass_density_times_lorentz_factor or
         min(get(get<ValenciaDivClean::Tags::TildeTau>(dg_vars))) <
             tci_options.minimum_tilde_tau or
         min(get(get<Inactive<ValenciaDivClean::Tags::TildeTau>>(
             subcell_vars))) < tci_options.minimum_tilde_tau or
         evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon) or
         evolution::dg::subcell::persson_tci(
             get<ValenciaDivClean::Tags::TildeD>(dg_vars), dg_mesh,
             persson_exponent, 1.0e-18) or
         evolution::dg::subcell::persson_tci(
             get<ValenciaDivClean::Tags::TildeTau>(dg_vars), dg_mesh,
             persson_exponent, 1.0e-18) or
         (tci_options.magnetic_field_cutoff.has_value() and
          max(get(tilde_b_magnitude)) >
              tci_options.magnetic_field_cutoff.value() and
          evolution::dg::subcell::persson_tci(tilde_b_magnitude, dg_mesh,
                                              persson_exponent, 1.0e-18));
}
}  // namespace grmhd::ValenciaDivClean::subcell
