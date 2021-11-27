// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
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
using NeighborDataMap =
    FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                 std::pair<Direction<Dim>, ElementId<Dim>>,
                 evolution::dg::subcell::NeighborData,
                 boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
template <size_t Dim>
using NeighborReconstructionMap =
    FixedHashMap<maximum_number_of_neighbors(Dim),
                 std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
                 boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;

template <size_t Dim>
using MortarData = std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                              std::optional<std::vector<double>>, ::TimeStepId>;

template <size_t Dim>
using MortarDataMap =
    FixedHashMap<maximum_number_of_neighbors(Dim),
                 std::pair<Direction<Dim>, ElementId<Dim>>, MortarData<Dim>,
                 boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  struct SubcellOptions {
    struct DgComputeSubcellNeighborPackagedData {
      template <typename DbTagsList>
      static NeighborReconstructionMap<Dim> apply(
          const db::DataBox<DbTagsList>& box,
          const std::vector<
              std::pair<Direction<volume_dim>, ElementId<volume_dim>>>&
              mortars_to_reconstruct_to) {
        const NeighborDataMap<Dim>& neighbor_data =
            db::get<evolution::dg::subcell::Tags::
                        NeighborDataForReconstructionAndRdmpTci<Dim>>(box);

        // We just simply copy over the data sent since it doesn't actually
        // matter what we fill the packaged data with in the test, just that
        // this function is called and that we can retrieve the correct data
        // from the stored NeighborData.
        NeighborReconstructionMap<Dim> neighbor_package_data{};
        for (const auto& mortar_id : mortars_to_reconstruct_to) {
          neighbor_package_data[mortar_id] =
              neighbor_data.at(mortar_id).data_for_reconstruction;
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
  const std::pair self_id{Direction<Dim>::lower_xi(),
                          ElementId<Dim>::external_boundary_id()};

  // Local neighbor data only holds local max/min values of evolved vars.
  evolution::dg::subcell::NeighborData local_neighbor_data{};
  local_neighbor_data.max_variables_values = std::vector<double>{1.0, 2.0};
  local_neighbor_data.min_variables_values = std::vector<double>{-2.0, 0.1};
  NeighborDataMap<Dim> neighbor_data_map{};
  neighbor_data_map[self_id] = std::move(local_neighbor_data);
  auto box =
      db::create<tmpl::list<evolution::dg::subcell::Tags::
                                NeighborDataForReconstructionAndRdmpTci<Dim>,
                            VolumeDouble>>(std::move(neighbor_data_map), 2.5);

  std::pair<const TimeStepId, MortarDataMap<Dim>> mortar_data_from_neighbors{};
  for (size_t d = 0; d < Dim; ++d) {
    const Mesh<Dim - 1> dg_face_mesh{2 + 2 * Dim, Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto};
    const Mesh<Dim - 1> fd_face_mesh{2 + 2 * Dim + 1,
                                     Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::CellCentered};
    std::vector<double> fd_recons_and_rdmp_data(2 * Dim + 1 + 4, 4.0);
    std::vector<double> dg_recons_and_rdmp_data(2 * Dim + 1 + 4, 7.0);
    for (size_t i = 0; i < 4; ++i) {
      dg_recons_and_rdmp_data[2 * Dim + 1 + i] =
          (i > 1 ? -1.0 : 1.0) * (d + 1.0) * 7.0 * (i + 5.0);
      fd_recons_and_rdmp_data[2 * Dim + 1 + i] =
          (i > 1 ? -1.0 : 1.0) * (d + 1.0) * 7.0 * (i + 50.0);
    }
    std::vector<double> dg_flux_data(2 * Dim + 1);
    if (d % 2 == 0) {
      mortar_data_from_neighbors.second[std::pair{
          Direction<Dim>{d, Side::Upper}, ElementId<Dim>{2 * d}}] =
          MortarData<Dim>{
              dg_face_mesh, dg_recons_and_rdmp_data, dg_flux_data, {}};
      mortar_data_from_neighbors.second[std::pair{
          Direction<Dim>{d, Side::Lower}, ElementId<Dim>{2 * d + 1}}] =
          MortarData<Dim>{
              fd_face_mesh, fd_recons_and_rdmp_data, std::nullopt, {}};
    } else {
      mortar_data_from_neighbors.second[std::pair{
          Direction<Dim>{d, Side::Lower}, ElementId<Dim>{2 * d}}] =
          MortarData<Dim>{
              dg_face_mesh, dg_recons_and_rdmp_data, dg_flux_data, {}};
      mortar_data_from_neighbors.second[std::pair{
          Direction<Dim>{d, Side::Upper}, ElementId<Dim>{2 * d + 1}}] =
          MortarData<Dim>{
              fd_face_mesh, fd_recons_and_rdmp_data, std::nullopt, {}};
    }
  }
  evolution::dg::subcell::neighbor_reconstructed_face_solution<metavars>(
      make_not_null(&box), make_not_null(&mortar_data_from_neighbors));
  for (size_t d = 0; d < Dim; ++d) {
    std::pair id{Direction<Dim>{d, Side::Lower}, ElementId<Dim>{2 * d + 1}};
    if (d % 2 != 0) {
      id = std::pair{Direction<Dim>{d, Side::Upper}, ElementId<Dim>{2 * d + 1}};
    }
    REQUIRE(std::get<1>(mortar_data_from_neighbors.second.at(id)).has_value());
    REQUIRE(std::get<2>(mortar_data_from_neighbors.second.at(id)).has_value());
    CHECK(*std::get<2>(mortar_data_from_neighbors.second.at(id)) ==
          std::vector<double>{
              std::get<1>(mortar_data_from_neighbors.second.at(id))->begin(),
              std::prev(
                  std::get<1>(mortar_data_from_neighbors.second.at(id))->end(),
                  4)});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborReconstructedFaceSolution",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
