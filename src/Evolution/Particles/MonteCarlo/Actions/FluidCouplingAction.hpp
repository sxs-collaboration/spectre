// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"

namespace Particles::MonteCarlo {

/// Mutator adding the Monte-Carlo contribution
/// to the evolution of the fluid.
struct FluidCouplingMutator {
  static const size_t Dim = 3;

  // We modify the fluid evolved variables, and reset the coupling
  // terms to zero.
  using return_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeYe,
                 grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
                 Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, Dim>>;
  using argument_tags = tmpl::list<
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>;

  static void apply(
      const gsl::not_null<Scalar<DataVector>*> tilde_tau,
      const gsl::not_null<Scalar<DataVector>*> tilde_ye,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> tilde_s,
      const gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
      const gsl::not_null<Scalar<DataVector>*> coupling_tilde_rho_ye,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> coupling_tilde_s,
      const Mesh<Dim>& mesh,
      const evolution::dg::subcell::ActiveGrid& active_grid,
      const Scalar<DataVector>& det_inverse_jacobian_logical_to_inertial) {
    // Currently, MC skips all non-communication actions when not using
    // Subcell.
    if (active_grid != evolution::dg::subcell::ActiveGrid::Subcell) {
      return;
    }

    // Coupling terms are on the mesh with ghost zones, while the evolved
    // variables are not. We also need to normalize the coupling terms with the
    // cell volume to get the evolution of energy/momentum density
    const Index<3>& extents = mesh.extents();
    const size_t num_ghost_zones = 1;
    const Index<3> extents_with_ghost{extents[0] + 2 * num_ghost_zones,
                                      extents[1] + 2 * num_ghost_zones,
                                      extents[2] + 2 * num_ghost_zones};

    Scalar<DataVector> det_jacobian_logical_to_inertial(*tilde_tau);
    get(det_jacobian_logical_to_inertial) =
        1.0 / get(det_inverse_jacobian_logical_to_inertial);
    Scalar<DataVector> cell_inertial_three_volume =
        make_with_value<Scalar<DataVector>>(*tilde_tau, 0.0);
    cell_inertial_coordinate_three_volume_finite_difference(
        &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);

    for (size_t i = 0; i < extents[0]; i++) {
      for (size_t j = 0; j < extents[1]; j++) {
        for (size_t k = 0; k < extents[2]; k++) {
          // The coupling terms are computed on a grid with ghost zone points,
          // the fluid variables are on the grid without GZ points
          // Index without GZ
          const size_t local_idx = collapsed_index(Index<3>{i, j, k}, extents);
          // Index with GZ
          const size_t extended_idx =
              collapsed_index(Index<3>{i + num_ghost_zones, j + num_ghost_zones,
                                       k + num_ghost_zones},
                              extents_with_ghost);
          // The MC coupling calculates the change in energy, momentum, and
          // lepton number. We need to divide by the cell 3-volume to get
          // the change in the evolved variables.
          const double& volume = get(cell_inertial_three_volume)[local_idx];
          get(*tilde_tau)[local_idx] +=
              get(*coupling_tilde_tau)[extended_idx] / volume;
          get(*tilde_ye)[local_idx] +=
              get(*coupling_tilde_rho_ye)[extended_idx] / volume;
          for (size_t d = 0; d < 3; d++) {
            tilde_s->get(d)[local_idx] +=
                coupling_tilde_s->get(d)[extended_idx] / volume;
          }
        }
      }
    }
    // Reset coupling terms to 0 after use
    get(*coupling_tilde_tau) = 0.0;
    get(*coupling_tilde_rho_ye) = 0.0;
    for (size_t d = 0; d < 3; d++) {
      coupling_tilde_s->get(d) = 0.0;
    }
  }
};

}  // namespace Particles::MonteCarlo
