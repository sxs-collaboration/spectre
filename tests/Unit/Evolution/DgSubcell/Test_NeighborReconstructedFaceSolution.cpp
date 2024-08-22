// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct VolumeDouble : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
using GhostDataMap = DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>;
template <size_t Dim>
using NeighborReconstructionMap = DirectionalIdMap<Dim, DataVector>;

template <size_t Dim>
using BoundaryData = evolution::dg::BoundaryData<Dim>;

template <size_t Dim>
using BoundaryDataMap = DirectionalIdMap<Dim, BoundaryData<Dim>>;

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  struct SubcellOptions {
    struct DgComputeSubcellNeighborPackagedData {
      static NeighborReconstructionMap<Dim> apply(
          const db::Access& box, const std::vector<DirectionalId<volume_dim>>&
                                     mortars_to_reconstruct_to) {
        const GhostDataMap<Dim>& ghost_data = db::get<
            evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(box);

        // We just simply copy over the data sent since it doesn't actually
        // matter what we fill the packaged data with in the test, just that
        // this function is called and that we can retrieve the correct data
        // from the stored NeighborData.
        NeighborReconstructionMap<Dim> neighbor_package_data{};
        for (const auto& mortar_id : mortars_to_reconstruct_to) {
          neighbor_package_data[mortar_id] =
              ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
        }
        return neighbor_package_data;
      }
    };
  };
};

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  using metavars = Metavariables<Dim>;

  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  rdmp_tci_data.max_variables_values = DataVector{1.0, 2.0};
  rdmp_tci_data.min_variables_values = DataVector{-2.0, 0.1};
  GhostDataMap<Dim> neighbor_data_map{};
  auto box = db::create<
      tmpl::list<evolution::dg::subcell::Tags::MeshForGhostData<Dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
                 evolution::dg::subcell::Tags::DataForRdmpTci, VolumeDouble>>(
      DirectionalIdMap<Dim, Mesh<Dim>>{}, std::move(neighbor_data_map),
      std::move(rdmp_tci_data), 2.5);

  std::pair<TimeStepId, BoundaryDataMap<Dim>> mortar_data_from_neighbors{};
  const Mesh<Dim> dg_volume_mesh{2 + 2 * Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> fd_volume_mesh{2 + 2 * Dim + 1,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered};
  const Mesh<Dim - 1> dg_face_mesh{2 + 2 * Dim, Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> fd_face_mesh{2 + 2 * Dim + 1,
                                   Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered};
  for (size_t d = 0; d < Dim; ++d) {
    DataVector fd_recons_and_rdmp_data(2 * Dim + 1 + 4, 4.0);
    DataVector dg_recons_and_rdmp_data(2 * Dim + 1 + 4, 7.0);
    for (size_t i = 0; i < 4; ++i) {
      dg_recons_and_rdmp_data[2 * Dim + 1 + i] =
          (i > 1 ? -1.0 : 1.0) * (d + 1.0) * 7.0 * (i + 5.0);
      fd_recons_and_rdmp_data[2 * Dim + 1 + i] =
          (i > 1 ? -1.0 : 1.0) * (d + 1.0) * 7.0 * (i + 50.0);
    }
    DataVector dg_flux_data(2 * Dim + 1);
    if (d % 2 == 0) {
      mortar_data_from_neighbors.second[DirectionalId<Dim>{
          Direction<Dim>{d, Side::Upper}, ElementId<Dim>{2 * d}}] =
          BoundaryData<Dim>{dg_volume_mesh,
                            dg_volume_mesh,
                            dg_face_mesh,
                            dg_recons_and_rdmp_data,
                            dg_flux_data,
                            {},
                            1};
      mortar_data_from_neighbors.second[DirectionalId<Dim>{
          Direction<Dim>{d, Side::Lower}, ElementId<Dim>{2 * d + 1}}] =
          BoundaryData<Dim>{fd_volume_mesh,
                            fd_volume_mesh,
                            fd_face_mesh,
                            fd_recons_and_rdmp_data,
                            std::nullopt,
                            {},
                            2};
    } else {
      mortar_data_from_neighbors.second[DirectionalId<Dim>{
          Direction<Dim>{d, Side::Lower}, ElementId<Dim>{2 * d}}] =
          BoundaryData<Dim>{dg_volume_mesh,
                            dg_volume_mesh,
                            dg_face_mesh,
                            dg_recons_and_rdmp_data,
                            dg_flux_data,
                            {},
                            3};
      mortar_data_from_neighbors.second[DirectionalId<Dim>{
          Direction<Dim>{d, Side::Upper}, ElementId<Dim>{2 * d + 1}}] =
          BoundaryData<Dim>{fd_volume_mesh,
                            fd_volume_mesh,
                            fd_face_mesh,
                            fd_recons_and_rdmp_data,
                            std::nullopt,
                            {},
                            4};
    }
  }
  evolution::dg::subcell::neighbor_reconstructed_face_solution<
      Dim,
      typename metavars::SubcellOptions::DgComputeSubcellNeighborPackagedData>(
      make_not_null(&box), make_not_null(&mortar_data_from_neighbors));
  const auto& ghost_meshes =
      db::get<evolution::dg::subcell::Tags::MeshForGhostData<Dim>>(box);
  for (size_t d = 0; d < Dim; ++d) {
    CAPTURE(d);
    const bool d_is_odd = (d % 2 != 0);
    const DirectionalId<Dim> id{
        Direction<Dim>{d, d_is_odd ? Side::Upper : Side::Lower},
        ElementId<Dim>{2 * d + 1}};
    CAPTURE(id);
    REQUIRE(mortar_data_from_neighbors.second.contains(id));
    REQUIRE(
        mortar_data_from_neighbors.second.at(id).ghost_cell_data.has_value());
    REQUIRE(mortar_data_from_neighbors.second.at(id)
                .boundary_correction_data.has_value());
    CHECK(mortar_data_from_neighbors.second.at(id)
              .boundary_correction_data.value() ==
          (DataVector{
              mortar_data_from_neighbors.second.at(id).ghost_cell_data->data(),
              mortar_data_from_neighbors.second.at(id).ghost_cell_data->size() -
                  4}));
    REQUIRE(ghost_meshes.contains(id));
    CHECK(ghost_meshes.at(id) == fd_volume_mesh);
    if (d_is_odd) {
      CHECK(mortar_data_from_neighbors.second.at(id).tci_status == 4);
    } else {
      CHECK(mortar_data_from_neighbors.second.at(id).tci_status == 2);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborReconstructedFaceSolution",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
