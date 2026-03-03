// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
using GhostDataMap = DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>;
template <size_t Dim>
using NeighborReconstructionMap = DirectionalIdMap<Dim, DataVector>;

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

  GhostDataMap<Dim> ghost_data{};
  const Mesh<Dim> dg_volume_mesh{2 + 2 * Dim, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> reconstructed_mesh = dg_volume_mesh.slice_away(0);

  const Mesh<Dim - 1> mortar_mesh{2 + 2 * Dim + 1, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
  DirectionalIdMap<Dim, evolution::dg::MortarDataHolder<Dim>> mortar_data_in{};
  DirectionalIdMap<Dim, evolution::dg::MortarInfo<Dim>> mortar_info_in{};
  for (size_t d = 0; d < Dim; ++d) {
    const bool d_is_odd = (d % 2 != 0);
    const DirectionalId<Dim> dg_id{
        Direction<Dim>{d, d_is_odd ? Side::Lower : Side::Upper},
        ElementId<Dim>{2 * d}};
    const DirectionalId<Dim> fd_id{
        Direction<Dim>{d, d_is_odd ? Side::Upper : Side::Lower},
        ElementId<Dim>{2 * d + 1}};
    mortar_data_in[dg_id].neighbor().mortar_mesh = mortar_mesh;
    mortar_data_in[dg_id].neighbor().mortar_data =
        DataVector(2 * Dim + 1, static_cast<double>(d) + 7.0);
    mortar_data_in[fd_id];
    ghost_data[fd_id] = evolution::dg::subcell::GhostData{1};
    ghost_data[fd_id].neighbor_ghost_data_for_reconstruction() =
        DataVector(2 * Dim + 1, static_cast<double>(d) + 4.0);
    mortar_info_in[dg_id].time_stepping_policy() =
        d == 2 ? evolution::dg::TimeSteppingPolicy::Conservative
               : evolution::dg::TimeSteppingPolicy::EqualRate;
    mortar_info_in[fd_id].time_stepping_policy() =
        d == 2 ? evolution::dg::TimeSteppingPolicy::Conservative
               : evolution::dg::TimeSteppingPolicy::EqualRate;
  }

  auto box = db::create<
      tmpl::list<domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
                 evolution::dg::Tags::MortarData<Dim>,
                 evolution::dg::Tags::MortarInfo<Dim>>>(
      dg_volume_mesh, std::move(ghost_data), std::move(mortar_data_in),
      std::move(mortar_info_in));

  evolution::dg::subcell::neighbor_reconstructed_face_solution<
      Dim,
      typename metavars::SubcellOptions::DgComputeSubcellNeighborPackagedData>(
      make_not_null(&box));
  const auto& mortar_data = db::get<evolution::dg::Tags::MortarData<Dim>>(box);
  for (size_t d = 0; d < Dim; ++d) {
    CAPTURE(d);
    const bool d_is_odd = (d % 2 != 0);
    const DirectionalId<Dim> dg_id{
        Direction<Dim>{d, d_is_odd ? Side::Lower : Side::Upper},
        ElementId<Dim>{2 * d}};
    const DirectionalId<Dim> fd_id{
        Direction<Dim>{d, d_is_odd ? Side::Upper : Side::Lower},
        ElementId<Dim>{2 * d + 1}};
    CAPTURE(dg_id);
    CAPTURE(fd_id);
    REQUIRE(mortar_data.contains(dg_id));
    REQUIRE(mortar_data.contains(fd_id));
    CHECK(mortar_data.at(dg_id).neighbor().mortar_data ==
          std::optional(DataVector(2 * Dim + 1, static_cast<double>(d) + 7.0)));
    CHECK(mortar_data.at(fd_id).neighbor().mortar_data ==
          (d == 2 ? std::nullopt
                  : std::optional(DataVector(2 * Dim + 1,
                                             static_cast<double>(d) + 4.0))));
    CHECK(mortar_data.at(dg_id).neighbor().mortar_mesh ==
          std::optional(mortar_mesh));
    CHECK(mortar_data.at(fd_id).neighbor().mortar_mesh ==
          (d == 2 ? std::nullopt : std::optional(reconstructed_mesh)));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborReconstructedFaceSolution",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
