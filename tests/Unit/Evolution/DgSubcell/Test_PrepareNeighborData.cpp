// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct GhostZoneSize : db::SimpleTag {
  using type = size_t;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;

  struct system {
    using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  };

  struct SubcellOptions {
    template <typename DbTagsList>
    static size_t ghost_zone_size(const db::DataBox<DbTagsList>& box) {
      return db::get<GhostZoneSize>(box);
    }

    struct GhostDataToSlice {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<evolution::dg::subcell::Tags::Inactive<
          typename system::variables_tag>>;
      template <typename T>
      static Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
      apply(const T& subcell_vars) {
        T subcell_vars_to_send = subcell_vars;
        get(get<evolution::dg::subcell::Tags::Inactive<Var1>>(
            subcell_vars_to_send)) *= 2.0;
        return subcell_vars_to_send;
      }
    };
  };
};

template <size_t Dim>
Element<Dim> create_element() {
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    // only populate some directions with neighbors to test that we can handle
    // that case correctly. This is needed for DG-subcell at external boundaries
    if (i % 2 == 0) {
      neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
          Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
      if constexpr (Dim == 3) {
        if (i == 2) {
          neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
              Neighbors<Dim>{
                  {ElementId<Dim>{i + 1, {}}},
                  OrientationMap<Dim>{std::array{
                      Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta(),
                      Direction<Dim>::upper_zeta()}}};
        }
      }
    }
  }

  return Element<Dim>{ElementId<Dim>{0, {}}, neighbors};
}

template <size_t Dim>
std::vector<Direction<Dim>> expected_neighbor_directions() {
  std::vector<Direction<Dim>> neighbor_directions{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    // only populate some directions with neighbors to test that we can handle
    // that case correctly. This is needed for DG-subcell at external boundaries
    if (i % 2 == 0) {
      neighbor_directions.push_back(
          gsl::at(Direction<Dim>::all_directions(), i));
    }
  }
  return neighbor_directions;
}

template <size_t Dim>
void test() {
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const Element<Dim> element = create_element<Dim>();

  Variables<tmpl::list<Var1>> vars{dg_mesh.number_of_grid_points(), 0.0};
  get(get<Var1>(vars)) = get<0>(logical_coordinates(dg_mesh));
  Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
      inactive_vars{subcell_mesh.number_of_grid_points()};
  inactive_vars = evolution::dg::subcell::fd::project(vars, dg_mesh,
                                                      subcell_mesh.extents());

  const size_t ghost_zone_size = 2;
  auto box = db::create<
      tmpl::list<GhostZoneSize, evolution::dg::subcell::Tags::Mesh<Dim>,
                 domain::Tags::Element<Dim>, variables_tag,
                 evolution::dg::subcell::Tags::Inactive<variables_tag>,
                 evolution::dg::subcell::Tags::
                     NeighborDataForReconstructionAndRdmpTci<Dim>>>(
      ghost_zone_size, subcell_mesh, element, vars, inactive_vars,
      typename evolution::dg::subcell::Tags::
          NeighborDataForReconstructionAndRdmpTci<Dim>::type{});
  const auto data_for_neighbors =
      evolution::dg::subcell::prepare_neighbor_data<Metavariables<Dim>>(
          make_not_null(&box));

  const std::pair self_id{Direction<Dim>::lower_xi(),
                          ElementId<Dim>::external_boundary_id()};
  const auto& local_neighbor_data =
      db::get<evolution::dg::subcell::Tags::
                  NeighborDataForReconstructionAndRdmpTci<Dim>>(box);
  CHECK(local_neighbor_data.size() == 1);
  REQUIRE(local_neighbor_data.contains(self_id) == 1);
  CHECK_ITERABLE_APPROX(local_neighbor_data.at(self_id).min_variables_values,
                        std::vector<double>{-1.0});
  CHECK_ITERABLE_APPROX(local_neighbor_data.at(self_id).max_variables_values,
                        std::vector<double>{1.0});

  // Set all directions to false, enable the desired ones below
  DirectionMap<Dim, bool> directions_to_slice{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    directions_to_slice[direction] = false;
  }

  REQUIRE(data_for_neighbors.size() == Dim);
  const size_t num_ghost_points =
      subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size;
  for (const auto& direction : expected_neighbor_directions<Dim>()) {
    REQUIRE(data_for_neighbors.contains(direction));
    REQUIRE(data_for_neighbors.at(direction).size() == num_ghost_points + 2);
    directions_to_slice[direction] = true;
  }

  // do same operation as GhostDataToSlice
  get(get<evolution::dg::subcell::Tags::Inactive<Var1>>(inactive_vars)) *= 2.0;
  auto expected_neighbor_data =
      evolution::dg::subcell::slice_data(inactive_vars, subcell_mesh.extents(),
                                         ghost_zone_size, directions_to_slice);
  if constexpr (Dim == 3) {
    const auto direction = expected_neighbor_directions<Dim>()[1];
    Index<Dim> slice_extents = subcell_mesh.extents();
    slice_extents[direction.dimension()] = ghost_zone_size;
    expected_neighbor_data.at(direction) =
        orient_variables(expected_neighbor_data.at(direction), slice_extents,
                         element.neighbors().at(direction).orientation());
  }

  for (const auto& direction : expected_neighbor_directions<Dim>()) {
    const auto& data_in_direction = data_for_neighbors.at(direction);
    CHECK_ITERABLE_APPROX(
        expected_neighbor_data.at(direction),
        std::vector<double>(data_in_direction.begin(),
                            std::prev(data_in_direction.end(), 2)));
    CHECK(*std::prev(data_in_direction.end(), 2) == approx(1.0));
    CHECK(*std::prev(data_in_direction.end(), 1) == approx(-1.0));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.PrepareNeighborData",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
