// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void test() {
  CAPTURE(Dim);
  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
  // Have upper xi neighbor do DG and lower xi neighbor do FD. For eta do
  // reverse, and zeta do same as xi.
  const Mesh<Dim> dg_mesh{6, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto logical_fd_coords = logical_coordinates(subcell_mesh);
  const size_t number_of_rdmp_vars = 2;
  const size_t number_of_ghost_zones = 3;  // 5th order method

  const DirectionalId<Dim> upper_xi_id{Direction<Dim>::upper_xi(),
                                       ElementId<Dim>{1}};
  const DirectionalId<Dim> lower_xi_id{Direction<Dim>::lower_xi(),
                                       ElementId<Dim>{2}};

  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  // aligned with neighbor so default construct
  neighbors.insert(std::pair{Direction<Dim>::upper_xi(),
                             Neighbors<Dim>{{upper_xi_id.id()}, {}}});
  if constexpr (Dim == 1) {
    neighbors.insert(std::pair{
        Direction<Dim>::lower_xi(),
        Neighbors<Dim>{{lower_xi_id.id()},
                       OrientationMap<Dim>{{{Direction<Dim>::lower_xi()}}}}});
  } else if constexpr (Dim == 2) {
    neighbors.insert(std::pair{
        Direction<Dim>::lower_xi(),
        Neighbors<Dim>{{lower_xi_id.id()},
                       OrientationMap<Dim>{{{Direction<Dim>::lower_xi(),
                                             Direction<Dim>::lower_eta()}}}}});
    neighbors.insert(std::pair{Direction<Dim>::upper_eta(),
                               Neighbors<Dim>{{ElementId<Dim>{3}}, {}}});
  } else if constexpr (Dim == 3) {
    neighbors.insert(std::pair{
        Direction<Dim>::lower_xi(),
        Neighbors<Dim>{{lower_xi_id.id()},
                       OrientationMap<Dim>{{{Direction<Dim>::lower_xi(),
                                             Direction<Dim>::lower_eta(),
                                             Direction<Dim>::upper_zeta()}}}}});
    neighbors.insert(std::pair{Direction<Dim>::upper_eta(),
                               Neighbors<Dim>{{ElementId<Dim>{3}}, {}}});

    neighbors.insert(std::pair{
        Direction<Dim>::lower_zeta(),
        Neighbors<Dim>{{ElementId<Dim>{4}},
                       OrientationMap<Dim>{{{Direction<Dim>::lower_xi(),
                                             Direction<Dim>::lower_eta(),
                                             Direction<Dim>::upper_zeta()}}}}});
    neighbors.insert(std::pair{Direction<Dim>::upper_zeta(),
                               Neighbors<Dim>{{ElementId<Dim>{5}}, {}}});
  }
  Interps neighbor_dg_to_fd_interpolants{};
  {
    const auto& orientation_map =
        neighbors.at(lower_xi_id.direction()).orientation();
    tnsr::I<DataVector, Dim, Frame::ElementLogical> oriented_logical_coords{};
    for (size_t i = 0; i < Dim; ++i) {
      oriented_logical_coords.get(i) = orient_variables(
          logical_fd_coords.get(i), subcell_mesh.extents(), orientation_map);
    }
    const auto target_points = evolution::dg::subcell::slice_tensor_for_subcell(
        oriented_logical_coords, subcell_mesh.extents(), number_of_ghost_zones,
        orientation_map(lower_xi_id.direction()), {});
    neighbor_dg_to_fd_interpolants[lower_xi_id] =
        intrp::Irregular<Dim>{dg_mesh, target_points};
  }
  if constexpr (Dim == 3) {
    const DirectionalId<Dim> lower_zeta_id{Direction<Dim>::lower_zeta(),
                                           ElementId<Dim>{4}};
    const auto& orientation_map =
        neighbors.at(lower_zeta_id.direction()).orientation();
    tnsr::I<DataVector, Dim, Frame::ElementLogical> oriented_logical_coords{};
    for (size_t i = 0; i < Dim; ++i) {
      oriented_logical_coords.get(i) = orient_variables(
          logical_fd_coords.get(i), subcell_mesh.extents(), orientation_map);
    }
    const auto target_points = evolution::dg::subcell::slice_tensor_for_subcell(
        oriented_logical_coords, subcell_mesh.extents(), number_of_ghost_zones,
        orientation_map(lower_zeta_id.direction()).opposite(), {});
    neighbor_dg_to_fd_interpolants[lower_zeta_id] =
        intrp::Irregular<Dim>{dg_mesh, target_points};
  }

  DataVector received_fd_data{subcell_mesh.number_of_grid_points() +
                              2 * number_of_rdmp_vars};
  alg::iota(received_fd_data, 0.0);
  DataVector received_dg_data{dg_mesh.number_of_grid_points() +
                              2 * number_of_rdmp_vars};
  alg::iota(received_dg_data, *std::prev(received_fd_data.end()) + 1.0);

  const Element<Dim> element{ElementId<Dim>{0}, neighbors};

  DataVector expected_neighbor_data_from_upper_xi{received_fd_data.size() -
                                                  2 * number_of_rdmp_vars};
  std::copy(received_fd_data.begin(),
            std::prev(received_fd_data.end(), 2 * number_of_rdmp_vars),
            expected_neighbor_data_from_upper_xi.begin());
  const DataVector expected_neighbor_data_from_lower_xi = [&dg_mesh, &neighbors,
                                                           &number_of_rdmp_vars,
                                                           &received_dg_data,
                                                           &subcell_mesh]() {
    (void)number_of_rdmp_vars;  // workaround clang bug unused warning
    // Need the view so the size is correct
    const DataVector view_received_data(
        received_dg_data.data(),
        received_dg_data.size() - 2 * number_of_rdmp_vars);
    DataVector oriented_data{view_received_data.size()};
    orient_variables(
        make_not_null(&oriented_data), view_received_data, dg_mesh.extents(),
        neighbors.at(Direction<Dim>::lower_xi()).orientation().inverse_map());
    // We've now got the data in the local orientation, so now we need to
    // project it to the ghost cells.
    // Note: assume isotropic meshes
    auto projection_matrices =
        make_array<Dim>(std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            Spectral::Quadrature::CellCentered)));
    projection_matrices[0] =
        std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            number_of_ghost_zones, Side::Upper));

    DataVector expected_data{subcell_mesh.extents().slice_away(0).product() *
                             number_of_ghost_zones};
    apply_matrices(make_not_null(&expected_data), projection_matrices,
                   oriented_data, dg_mesh.extents());
    return expected_data;
  }();

  DirectionalIdMap<Dim, evolution::dg::subcell::GhostData> neighbor_data{};
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      DataVector{std::numeric_limits<double>::min(),
                 std::numeric_limits<double>::min()},
      DataVector{std::numeric_limits<double>::max(),
                 std::numeric_limits<double>::max()}};
  // Do upper-xi neighbor first. This is just aligned FD
  evolution::dg::subcell::insert_neighbor_rdmp_and_volume_data(
      make_not_null(&rdmp_tci_data), make_not_null(&neighbor_data),
      received_fd_data, number_of_rdmp_vars, upper_xi_id,
      subcell_mesh,  // neighbor mesh is the same as my mesh since both are
                     // doing subcell
      element, subcell_mesh, number_of_ghost_zones,
      neighbor_dg_to_fd_interpolants);

  const auto get_neighbor_data =
      [&neighbor_data](const auto mortar_id) -> const DataVector& {
    return neighbor_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
  };
  {
    DataVector expected_max_rdmp_tci_data{number_of_rdmp_vars};
    DataVector expected_min_rdmp_tci_data{number_of_rdmp_vars};
    std::copy(std::prev(received_fd_data.end(), 2 * number_of_rdmp_vars),
              std::prev(received_fd_data.end(), number_of_rdmp_vars),
              expected_max_rdmp_tci_data.begin());
    std::copy(std::prev(received_fd_data.end(), number_of_rdmp_vars),
              received_fd_data.end(), expected_min_rdmp_tci_data.begin());
    CHECK(rdmp_tci_data.max_variables_values == expected_max_rdmp_tci_data);
    CHECK(rdmp_tci_data.min_variables_values == expected_min_rdmp_tci_data);

    REQUIRE(neighbor_data.size() == 1);
    REQUIRE(neighbor_data.find(upper_xi_id) != neighbor_data.end());
    CHECK(get_neighbor_data(upper_xi_id) ==
          expected_neighbor_data_from_upper_xi);
  }

  // Do lower-xi neighbor. This is unaligned DG.
  evolution::dg::subcell::insert_neighbor_rdmp_and_volume_data(
      make_not_null(&rdmp_tci_data), make_not_null(&neighbor_data),
      received_dg_data, number_of_rdmp_vars, lower_xi_id, dg_mesh, element,
      subcell_mesh, number_of_ghost_zones, neighbor_dg_to_fd_interpolants);

  {
    DataVector expected_max_rdmp_tci_data{number_of_rdmp_vars};
    DataVector expected_min_rdmp_tci_data{number_of_rdmp_vars};
    std::copy(std::prev(received_dg_data.end(), 2 * number_of_rdmp_vars),
              std::prev(received_dg_data.end(), number_of_rdmp_vars),
              expected_max_rdmp_tci_data.begin());
    std::copy(std::prev(received_fd_data.end(), number_of_rdmp_vars),
              received_fd_data.end(), expected_min_rdmp_tci_data.begin());
    CHECK(rdmp_tci_data.max_variables_values == expected_max_rdmp_tci_data);
    CHECK(rdmp_tci_data.min_variables_values == expected_min_rdmp_tci_data);

    REQUIRE(neighbor_data.size() == 2);
    REQUIRE(neighbor_data.find(upper_xi_id) != neighbor_data.end());
    REQUIRE(neighbor_data.find(lower_xi_id) != neighbor_data.end());
    CHECK(get_neighbor_data(upper_xi_id) ==
          expected_neighbor_data_from_upper_xi);
    CHECK_ITERABLE_APPROX(get_neighbor_data(lower_xi_id),
                          expected_neighbor_data_from_lower_xi);
  }

  if constexpr (Dim > 1) {
    // Do upper-eta neighbor. This is aligned DG.
    const DirectionalId<Dim> upper_eta_id{Direction<Dim>::upper_eta(),
                                          ElementId<Dim>{3}};

    DataVector aligned_received_dg_data{dg_mesh.number_of_grid_points() +
                                        2 * number_of_rdmp_vars};
    alg::iota(aligned_received_dg_data,
              *std::prev(received_dg_data.end()) + 1.0);
    evolution::dg::subcell::insert_neighbor_rdmp_and_volume_data(
        make_not_null(&rdmp_tci_data), make_not_null(&neighbor_data),
        aligned_received_dg_data, number_of_rdmp_vars, upper_eta_id, dg_mesh,
        element, subcell_mesh, number_of_ghost_zones, {});

    DataVector expected_max_rdmp_tci_data{number_of_rdmp_vars};
    DataVector expected_min_rdmp_tci_data{number_of_rdmp_vars};
    std::copy(
        std::prev(aligned_received_dg_data.end(), 2 * number_of_rdmp_vars),
        std::prev(aligned_received_dg_data.end(), number_of_rdmp_vars),
        expected_max_rdmp_tci_data.begin());
    std::copy(std::prev(received_fd_data.end(), number_of_rdmp_vars),
              received_fd_data.end(), expected_min_rdmp_tci_data.begin());
    CHECK(rdmp_tci_data.max_variables_values == expected_max_rdmp_tci_data);
    CHECK(rdmp_tci_data.min_variables_values == expected_min_rdmp_tci_data);

    REQUIRE(neighbor_data.size() == 3);
    REQUIRE(neighbor_data.find(upper_xi_id) != neighbor_data.end());
    REQUIRE(neighbor_data.find(lower_xi_id) != neighbor_data.end());
    REQUIRE(neighbor_data.find(upper_eta_id) != neighbor_data.end());
    CHECK(get_neighbor_data(upper_xi_id) ==
          expected_neighbor_data_from_upper_xi);
    CHECK_ITERABLE_APPROX(get_neighbor_data(lower_xi_id),
                          expected_neighbor_data_from_lower_xi);

    auto projection_matrices =
        make_array<Dim>(std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            Spectral::Quadrature::CellCentered)));
    projection_matrices[1] =
        std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            number_of_ghost_zones, Side::Lower));

    DataVector view_aligned_received_dg_data(aligned_received_dg_data.data(),
                                             dg_mesh.number_of_grid_points());
    DataVector expected_data{subcell_mesh.extents().slice_away(0).product() *
                             number_of_ghost_zones};
    apply_matrices(make_not_null(&expected_data), projection_matrices,
                   view_aligned_received_dg_data, dg_mesh.extents());
    CHECK(get_neighbor_data(upper_eta_id) == expected_data);
  }

  {
    // Check that not inserting but updating in a direction that already has FD
    // data does nothing. That is, even the pointer should stay the same.
    const double* expected_pointer = get_neighbor_data(lower_xi_id).data();
    evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
        make_not_null(&neighbor_data), get_neighbor_data(lower_xi_id),
        number_of_rdmp_vars, lower_xi_id, subcell_mesh, element, subcell_mesh,
        number_of_ghost_zones, neighbor_dg_to_fd_interpolants);
    CHECK(get_neighbor_data(lower_xi_id).data() == expected_pointer);
  }

  if constexpr (Dim > 2) {
    // Check that a neighbor being aligned DG and unaligned DG both work when
    // not inserting.

    // Do upper-zeta neighbor. This is aligned DG.
    const DirectionalId<Dim> upper_zeta_id{Direction<Dim>::upper_zeta(),
                                           ElementId<Dim>{5}};
    DataVector aligned_received_dg_data{dg_mesh.number_of_grid_points()};
    alg::iota(aligned_received_dg_data,
              *std::prev(received_dg_data.end()) + 1.0);
    neighbor_data[upper_zeta_id] = evolution::dg::subcell::GhostData{1};
    neighbor_data[upper_zeta_id].neighbor_ghost_data_for_reconstruction() =
        aligned_received_dg_data;
    evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
        make_not_null(&neighbor_data), get_neighbor_data(upper_zeta_id), 0,
        upper_zeta_id, dg_mesh, element, subcell_mesh, number_of_ghost_zones,
        neighbor_dg_to_fd_interpolants);

    auto projection_matrices =
        make_array<Dim>(std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            Spectral::Quadrature::CellCentered)));
    projection_matrices[2] =
        std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            number_of_ghost_zones, Side::Lower));

    DataVector expected_data{subcell_mesh.extents().slice_away(0).product() *
                             number_of_ghost_zones};
    apply_matrices(make_not_null(&expected_data), projection_matrices,
                   aligned_received_dg_data, dg_mesh.extents());
    CHECK(get_neighbor_data(upper_zeta_id) == expected_data);

    // Do lower-zeta neighbor. This is unaligned DG.
    const DirectionalId<Dim> lower_zeta_id{Direction<Dim>::lower_zeta(),
                                           ElementId<Dim>{4}};
    DataVector unaligned_received_dg_data{dg_mesh.number_of_grid_points()};
    alg::iota(unaligned_received_dg_data,
              *std::prev(aligned_received_dg_data.end()) + 1.0);
    neighbor_data[lower_zeta_id] = evolution::dg::subcell::GhostData{1};
    neighbor_data[lower_zeta_id].neighbor_ghost_data_for_reconstruction() =
        unaligned_received_dg_data;

    evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
        make_not_null(&neighbor_data), get_neighbor_data(lower_zeta_id), 0,
        lower_zeta_id, dg_mesh, element, subcell_mesh, number_of_ghost_zones,
        neighbor_dg_to_fd_interpolants);

    projection_matrices[2] =
        std::cref(evolution::dg::subcell::fd::projection_matrix(
            dg_mesh.slice_through(0), subcell_mesh.extents(0),
            number_of_ghost_zones, Side::Upper));

    DataVector oriented_data{unaligned_received_dg_data.size()};
    orient_variables(
        make_not_null(&oriented_data), unaligned_received_dg_data,
        dg_mesh.extents(),
        neighbors.at(Direction<Dim>::lower_zeta()).orientation().inverse_map());
    apply_matrices(make_not_null(&expected_data), projection_matrices,
                   oriented_data, dg_mesh.extents());
    CHECK_ITERABLE_APPROX(get_neighbor_data(lower_zeta_id), expected_data);
  }

#ifdef SPECTRE_DEBUG
  // Test ASSERTs
  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_neighbor_rdmp_and_volume_data(
          make_not_null(&rdmp_tci_data), make_not_null(&neighbor_data),
          DataVector{}, number_of_rdmp_vars,
          DirectionalId<Dim>{Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
          subcell_mesh, element, subcell_mesh, number_of_ghost_zones,
          neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "received_neighbor_subcell_data must be non-empty"));

  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<true>(
          make_not_null(&neighbor_data), DataVector{}, number_of_rdmp_vars,
          DirectionalId<Dim>{Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
          subcell_mesh, element, subcell_mesh, number_of_ghost_zones,
          neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "neighbor_subcell_data must be non-empty"));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
          make_not_null(&neighbor_data), DataVector{}, number_of_rdmp_vars,
          DirectionalId<Dim>{Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
          subcell_mesh, element, subcell_mesh, number_of_ghost_zones,
          neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "neighbor_subcell_data must be non-empty"));

  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<true>(
          make_not_null(&neighbor_data), received_fd_data, number_of_rdmp_vars,
          upper_xi_id,
          Mesh<Dim>{5, Spectral::Basis::FiniteDifference,
                    Spectral::Quadrature::CellCentered},
          element, subcell_mesh, number_of_ghost_zones,
          neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "must be the same if we are both doing subcell."));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
          make_not_null(&neighbor_data), received_fd_data, number_of_rdmp_vars,
          upper_xi_id,
          Mesh<Dim>{5, Spectral::Basis::FiniteDifference,
                    Spectral::Quadrature::CellCentered},
          element, subcell_mesh, number_of_ghost_zones,
          neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "must be the same if we are both doing subcell."));

  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<true>(
          make_not_null(&neighbor_data), received_dg_data, number_of_rdmp_vars,
          lower_xi_id, dg_mesh, element,
          Mesh<Dim>{4, Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto},
          number_of_ghost_zones, neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "Neighbor subcell mesh computed from the neighbor DG mesh "));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
          make_not_null(&neighbor_data), received_dg_data, number_of_rdmp_vars,
          lower_xi_id, dg_mesh, element,
          Mesh<Dim>{4, Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto},
          number_of_ghost_zones, neighbor_dg_to_fd_interpolants),
      Catch::Matchers::ContainsSubstring(
          "Neighbor subcell mesh computed from the neighbor DG mesh "));

  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<true>(
          make_not_null(&neighbor_data),
          DataVector{2 * number_of_rdmp_vars + 1, 0.0}, number_of_rdmp_vars,
          lower_xi_id, dg_mesh, element, subcell_mesh, number_of_ghost_zones,
          {}),
      Catch::Matchers::ContainsSubstring(
          "The number of DG volume grid points times the number of variables"));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
          make_not_null(&neighbor_data),
          DataVector{2 * number_of_rdmp_vars + 1, 0.0}, number_of_rdmp_vars,
          lower_xi_id, dg_mesh, element, subcell_mesh, number_of_ghost_zones,
          {}),
      Catch::Matchers::ContainsSubstring(
          "The number of DG volume grid points times the number of variables"));

  if constexpr (Dim > 1) {
    Mesh<Dim> non_uniform_mesh{};
    if constexpr (Dim == 2) {
      non_uniform_mesh = Mesh<2>{{{4, 5}},
                                 Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
    } else if constexpr (Dim == 3) {
      non_uniform_mesh = Mesh<3>{{{4, 5, 6}},
                                 Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
    }
    CHECK_THROWS_WITH(
        evolution::dg::subcell::insert_or_update_neighbor_volume_data<true>(
            make_not_null(&neighbor_data), received_fd_data,
            number_of_rdmp_vars,
            DirectionalId<Dim>{Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
            non_uniform_mesh, element, subcell_mesh, number_of_ghost_zones,
            neighbor_dg_to_fd_interpolants),
        Catch::Matchers::ContainsSubstring(
            "The neighbor mesh must be uniform but is"));
    CHECK_THROWS_WITH(
        evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
            make_not_null(&neighbor_data), received_fd_data,
            number_of_rdmp_vars,
            DirectionalId<Dim>{Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
            non_uniform_mesh, element, subcell_mesh, number_of_ghost_zones,
            neighbor_dg_to_fd_interpolants),
        Catch::Matchers::ContainsSubstring(
            "The neighbor mesh must be uniform but is"));
  }
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborRdmpAndVolumeData",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
