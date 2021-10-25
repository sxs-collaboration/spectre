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
namespace detail {
bool initial_data_tci_work(
    const Scalar<DataVector>& dg_tilde_d,
    const Scalar<DataVector>& dg_tilde_tau,
    const Scalar<DataVector>& subcell_tilde_d,
    const Scalar<DataVector>& subcell_tilde_tau,
    const tnsr::I<DataVector, 3, Frame::Inertial>& dg_tilde_b,
    const double persson_exponent, const Mesh<3>& dg_mesh,
    const TciOptions& tci_options) {
  const Scalar<DataVector> tilde_b_magnitude =
      tci_options.magnetic_field_cutoff.has_value() ? magnitude(dg_tilde_b)
                                                    : Scalar<DataVector>{};

  return min(get(dg_tilde_d)) <
             tci_options.minimum_rest_mass_density_times_lorentz_factor or
         min(get(subcell_tilde_d)) <
             tci_options.minimum_rest_mass_density_times_lorentz_factor or
         min(get(dg_tilde_tau)) < tci_options.minimum_tilde_tau or
         min(get(subcell_tilde_tau)) < tci_options.minimum_tilde_tau or
         evolution::dg::subcell::persson_tci(dg_tilde_d, dg_mesh,
                                             persson_exponent) or
         evolution::dg::subcell::persson_tci(dg_tilde_tau, dg_mesh,
                                             persson_exponent) or
         (tci_options.magnetic_field_cutoff.has_value() and
          max(get(tilde_b_magnitude)) >
              tci_options.magnetic_field_cutoff.value() and
          evolution::dg::subcell::persson_tci(tilde_b_magnitude, dg_mesh,
                                              persson_exponent));
}
}  // namespace detail

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
    const double rdmp_delta0, const double rdmp_epsilon,
    const double persson_exponent, const Mesh<3>& dg_mesh,
    const TciOptions& tci_options) {
  return detail::initial_data_tci_work(
             get<ValenciaDivClean::Tags::TildeD>(dg_vars),
             get<ValenciaDivClean::Tags::TildeTau>(dg_vars),
             get<Inactive<ValenciaDivClean::Tags::TildeD>>(subcell_vars),
             get<Inactive<ValenciaDivClean::Tags::TildeTau>>(subcell_vars),
             get<ValenciaDivClean::Tags::TildeB<>>(dg_vars), persson_exponent,
             dg_mesh, tci_options) or
         evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon);
}
}  // namespace grmhd::ValenciaDivClean::subcell
