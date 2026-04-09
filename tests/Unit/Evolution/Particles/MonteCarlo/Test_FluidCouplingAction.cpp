// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Particles/MonteCarlo/Actions/FluidCouplingAction.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  using flux_variables = tmpl::list<Var1>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using initial_tags = tmpl::list<
      domain::Tags::Mesh<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      ::Tags::Variables<tmpl::list<Var1>>,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeYe,
      grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
      Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>,
      Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>,
      Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, Dim>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags>,
          Actions::MutateApply<Particles::MonteCarlo::FluidCouplingMutator>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;

  using const_global_cache_tags =
      tmpl::list<hydro::Tags::GrmhdEquationOfState,
                 Particles::MonteCarlo::Tags::InteractionRatesTable<4, 3>>;
};

void test_fluid_coupling() {
  const size_t Dim = 3;

  using metavars = Metavariables<Dim>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;
  const size_t n_pts = subcell_mesh.number_of_grid_points();

  const DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  const ElementId<Dim> self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
  const Element<Dim> element{self_id, neighbors};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{n_pts};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));

  const DataVector zero_dv(n_pts, 0.0);

  // Fluid variables
  Scalar<DataVector> tilde_tau{DataVector(n_pts, 0.1)};
  Scalar<DataVector> tilde_ye{DataVector(n_pts, 0.1)};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_tau, 0.0);

  // Coupling data
  const auto& subcell_extents = subcell_mesh.extents();
  const size_t mesh_size_1d = subcell_extents[0];
  const size_t num_ghost_zones = 1;
  const size_t mesh_size_with_ghost_1d = mesh_size_1d + 2 * num_ghost_zones;
  size_t mesh_size_with_ghost = 1;
  for (size_t d = 0; d < Dim; d++) {
    mesh_size_with_ghost *= mesh_size_with_ghost_1d;
  }
  const DataVector zero_dv_with_ghost(mesh_size_with_ghost, 0.0);
  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv_with_ghost, 0.0);
  Scalar<DataVector> coupling_tilde_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv_with_ghost, 0.0);
  tnsr::i<DataVector, Dim> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, Dim>>(zero_dv_with_ghost, 0.0);
  alg::iota(get(coupling_tilde_tau), 1.0);
  alg::iota(get(coupling_tilde_rho_ye), 2.0);
  alg::iota(coupling_tilde_s.get(0), 3.0);
  alg::iota(coupling_tilde_s.get(1), 4.0);
  alg::iota(coupling_tilde_s.get(2), 5.0);

  // Jacobian for volume element
  Scalar<DataVector> det_inverse_jacobian_logical_to_inertial(n_pts, 2.0);
  const double cell_volume =
      8.0 / static_cast<double>(subcell_mesh.number_of_grid_points()) / 2.0;

  // Expected post-coupling values of fluid variables
  Scalar<DataVector> expected_tilde_tau{DataVector(n_pts, 0.0)};
  Scalar<DataVector> expected_tilde_ye{DataVector(n_pts, 0.0)};
  tnsr::i<DataVector, 3, Frame::Inertial> expected_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_tau, 0.0);
  for (size_t i = 0; i < mesh_size_1d; i++) {
    for (size_t j = 0; j < mesh_size_1d; j++) {
      for (size_t k = 0; k < mesh_size_1d; k++) {
        const size_t local_idx = i + mesh_size_1d * (j + k * mesh_size_1d);
        const size_t extended_idx =
            num_ghost_zones + i +
            mesh_size_with_ghost_1d *
                (num_ghost_zones + j +
                 (num_ghost_zones + k) * mesh_size_with_ghost_1d);
        get(expected_tilde_tau)[local_idx] =
            get(tilde_tau)[local_idx] +
            get(coupling_tilde_tau)[extended_idx] / cell_volume;
        get(expected_tilde_ye)[local_idx] =
            get(tilde_ye)[local_idx] +
            get(coupling_tilde_rho_ye)[extended_idx] / cell_volume;
        for (size_t d = 0; d < Dim; d++) {
          expected_tilde_s.get(d)[local_idx] =
              tilde_s.get(d)[local_idx] +
              coupling_tilde_s.get(d)[extended_idx] / cell_volume;
        }
      }
    }
  }

  MockRuntimeSystem runner{{}};

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {dg_mesh, subcell_mesh, active_grid, element, evolved_vars, tilde_tau,
       tilde_ye, tilde_s, coupling_tilde_tau, coupling_tilde_rho_ye,
       coupling_tilde_s, det_inverse_jacobian_logical_to_inertial});

  // Run singe time step
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  const auto& tilde_tau_from_box =
      ActionTesting::get_databox_tag<comp,
                                     grmhd::ValenciaDivClean::Tags::TildeTau>(
          runner, self_id);
  const auto& tilde_ye_from_box =
      ActionTesting::get_databox_tag<comp,
                                     grmhd::ValenciaDivClean::Tags::TildeYe>(
          runner, self_id);
  const auto& tilde_s_from_box = ActionTesting::get_databox_tag<
      comp, grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(runner,
                                                                    self_id);
  const auto& coupling_tilde_tau_from_box = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>>(runner,
                                                                       self_id);
  const auto& coupling_tilde_ye_from_box = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>>(
      runner, self_id);
  const auto& coupling_tilde_s_from_box = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, Dim>>(
      runner, self_id);
  CHECK_ITERABLE_APPROX(tilde_tau_from_box, expected_tilde_tau);
  CHECK_ITERABLE_APPROX(tilde_ye_from_box, expected_tilde_ye);
  CHECK_ITERABLE_APPROX(tilde_s_from_box, expected_tilde_s);
  CHECK_ITERABLE_APPROX(get(coupling_tilde_tau_from_box), zero_dv_with_ghost);
  CHECK_ITERABLE_APPROX(get(coupling_tilde_ye_from_box), zero_dv_with_ghost);
  for (size_t d = 0; d < Dim; d++) {
    CHECK_ITERABLE_APPROX(coupling_tilde_s_from_box.get(d), zero_dv_with_ghost);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloFluidCouplingAction",
                  "[Unit][Evolution]") {
  test_fluid_coupling();
}
