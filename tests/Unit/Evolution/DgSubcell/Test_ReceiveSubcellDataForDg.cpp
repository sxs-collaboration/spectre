// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/ReceiveSubcellDataForDg.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/MeshForGhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
using GhostDataMap = DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>;

template <size_t Dim>
using BoundaryData = evolution::dg::BoundaryData<Dim>;

template <size_t Dim>
using BoundaryDataMap = DirectionalIdMap<Dim, BoundaryData<Dim>>;

template <size_t Dim>
void test() {
  CAPTURE(Dim);

  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  rdmp_tci_data.max_variables_values = DataVector{1.0, 2.0};
  rdmp_tci_data.min_variables_values = DataVector{-2.0, 0.1};
  auto expected_rdmp_tci_data = rdmp_tci_data;
  GhostDataMap<Dim> neighbor_data_map{};  // NOLINT(misc-const-correctness)
  const Mesh<Dim> dg_volume_mesh{2 + 2 * Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  auto box = db::create<
      tmpl::list<domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::MeshForGhostData<Dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
                 evolution::dg::subcell::Tags::DataForRdmpTci>>(
      dg_volume_mesh, DirectionalIdMap<Dim, Mesh<Dim>>{},
      std::move(neighbor_data_map), std::move(rdmp_tci_data));

  const Mesh<Dim> fd_volume_mesh{2 + 2 * Dim + 1,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered};
  const Mesh<Dim - 1> mortar_mesh{2 + 2 * Dim + 1, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};

  const DirectionalId<Dim> dg_id{Direction<Dim>::upper_xi(), ElementId<Dim>{0}};
  const DirectionalId<Dim> fd_id{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};

  DataVector fd_recons_and_rdmp_data(2 * Dim + 1 + 4, 4.0);
  DataVector dg_recons_and_rdmp_data(2 * Dim + 1 + 4, 7.0);
  for (size_t i = 0; i < 4; ++i) {
    dg_recons_and_rdmp_data[2 * Dim + 1 + i] =
        (i > 1 ? -1.0 : 1.0) * 7.0 * (static_cast<double>(i) + 5.0);
    fd_recons_and_rdmp_data[2 * Dim + 1 + i] =
        (i > 1 ? -1.0 : 1.0) * 7.0 * (static_cast<double>(i) + 50.0);
  }
  expected_rdmp_tci_data.max_variables_values =
      max(expected_rdmp_tci_data.max_variables_values,
          DataVector(&fd_recons_and_rdmp_data[2 * Dim + 1], 2));
  expected_rdmp_tci_data.min_variables_values =
      min(expected_rdmp_tci_data.min_variables_values,
          DataVector(&fd_recons_and_rdmp_data[2 * Dim + 3], 2));
  DataVector dg_flux_data(2 * Dim + 1);

  evolution::dg::subcell::receive_subcell_data_for_dg<Dim>(
      make_not_null(&box), dg_id,
      BoundaryData<Dim>{dg_volume_mesh,
                        dg_volume_mesh,
                        mortar_mesh,
                        dg_recons_and_rdmp_data,
                        dg_flux_data,
                        {},
                        1});
  evolution::dg::subcell::receive_subcell_data_for_dg<Dim>(
      make_not_null(&box), fd_id,
      BoundaryData<Dim>{dg_volume_mesh,
                        fd_volume_mesh,
                        std::nullopt,
                        fd_recons_and_rdmp_data,
                        std::nullopt,
                        {},
                        2});

  CHECK(db::get<evolution::dg::subcell::Tags::DataForRdmpTci>(box) ==
        expected_rdmp_tci_data);

  const auto& ghost_meshes =
      db::get<evolution::dg::subcell::Tags::MeshForGhostData<Dim>>(box);
  const auto& reconstruction_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
          box);

  REQUIRE(reconstruction_data.contains(dg_id));
  CHECK(
      reconstruction_data.at(dg_id).neighbor_ghost_data_for_reconstruction() ==
      (DataVector{dg_recons_and_rdmp_data.data(),
                  dg_recons_and_rdmp_data.size() - 4}));

  REQUIRE(reconstruction_data.contains(fd_id));
  CHECK(
      reconstruction_data.at(fd_id).neighbor_ghost_data_for_reconstruction() ==
      (DataVector{fd_recons_and_rdmp_data.data(),
                  fd_recons_and_rdmp_data.size() - 4}));

  REQUIRE(ghost_meshes.contains(dg_id));
  CHECK(ghost_meshes.at(dg_id) == dg_volume_mesh);
  REQUIRE(ghost_meshes.contains(fd_id));
  CHECK(ghost_meshes.at(fd_id) == fd_volume_mesh);
}

// Elements whose mesh does not support subcell (e.g. spherical shells) must
// be skipped entirely: they may have more neighbors than the DirectionalIdMap
// capacity and they never need ghost data.
template <size_t Dim>
void test_nonhypercube_mesh() {
  CAPTURE(Dim);

  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  rdmp_tci_data.max_variables_values = DataVector{1.0, 2.0};
  rdmp_tci_data.min_variables_values = DataVector{-2.0, 0.1};
  const auto initial_rdmp = rdmp_tci_data;

  const Mesh<Dim> sph_mesh{4, Spectral::Basis::SphericalHarmonic,
                           Spectral::Quadrature::Gauss};

  auto box = db::create<
      tmpl::list<domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::MeshForGhostData<Dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
                 evolution::dg::subcell::Tags::DataForRdmpTci>>(
      sph_mesh, DirectionalIdMap<Dim, Mesh<Dim>>{}, GhostDataMap<Dim>{},
      std::move(rdmp_tci_data));

  const DirectionalId<Dim> did{Direction<Dim>::upper_xi(), ElementId<Dim>{0}};
  evolution::dg::subcell::receive_subcell_data_for_dg<Dim>(
      make_not_null(&box), did, evolution::dg::BoundaryData<Dim>{});

  // Nothing should have been stored; RDMP data is also unchanged.
  CHECK(db::get<evolution::dg::subcell::Tags::MeshForGhostData<Dim>>(box)
            .empty());
  CHECK(db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
            box)
            .empty());
  CHECK(db::get<evolution::dg::subcell::Tags::DataForRdmpTci>(box) ==
        initial_rdmp);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.ReceiveSubcellDataForDg",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
  test_nonhypercube_mesh<1>();
  test_nonhypercube_mesh<2>();
  test_nonhypercube_mesh<3>();
}
