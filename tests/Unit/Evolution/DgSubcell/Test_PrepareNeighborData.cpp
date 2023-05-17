// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CartesianProduct.hpp"
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
    using flux_variables = tmpl::list<Var1>;
  };

  struct SubcellOptions {
    template <typename DbTagsList>
    static size_t ghost_zone_size(const db::DataBox<DbTagsList>& box) {
      return db::get<GhostZoneSize>(box);
    }

    struct GhostVariables {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<typename system::variables_tag>;
      template <typename T>
      static DataVector apply(const T& dg_vars, const size_t rdmp_size) {
        DataVector buffer{dg_vars.size() + rdmp_size};
        Variables<tmpl::list<Var1>> subcell_vars_to_send{buffer.data(),
                                                         dg_vars.size()};
        get(get<Var1>(subcell_vars_to_send)) = 2.0 * get(get<Var1>(dg_vars));
        return buffer;
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
FixedHashMap<maximum_number_of_neighbors(Dim),
             std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim>,
             boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
compute_neighbor_meshes(const Element<Dim>& element, const bool all_dg,
                        const Mesh<Dim>& dg_mesh,
                        const Mesh<Dim>& subcell_mesh) {
  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim>,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      result{};
  bool already_set_fd = false;
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      if (not already_set_fd and not all_dg) {
        result.insert(std::pair{std::pair{direction, neighbor}, subcell_mesh});
        already_set_fd = true;
      } else {
        result.insert(std::pair{std::pair{direction, neighbor}, dg_mesh});
      }
    }
  }
  return result;
}

// TestCreator class needed for subcell options specified below
template <size_t Dim>
class TestCreator : public DomainCreator<Dim> {
  Domain<Dim> create_domain() const override { return Domain<Dim>{}; }
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    return {};
  }

  std::vector<std::string> block_names() const override {
    return {"Block0", "Block1"};
  }

  std::vector<std::array<size_t, Dim>> initial_extents() const override {
    return {};
  }

  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override {
    return {};
  }
};

template <size_t Dim>
void test(const bool all_neighbors_are_doing_dg,
          const ::fd::DerivativeOrder fd_derivative_order) {
  CAPTURE(all_neighbors_are_doing_dg);
  CAPTURE(fd_derivative_order);
  CAPTURE(Dim);
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const Element<Dim> element = create_element<Dim>();

  const auto neighbor_meshes = compute_neighbor_meshes(
      element, all_neighbors_are_doing_dg, dg_mesh, subcell_mesh);

  Variables<tmpl::list<Var1>> vars{dg_mesh.number_of_grid_points(), 0.0};
  get(get<Var1>(vars)) = get<0>(logical_coordinates(dg_mesh));
  using flux_tag = ::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>;
  Variables<tmpl::list<flux_tag>> volume_fluxes{dg_mesh.number_of_grid_points(),
                                                0.0};
  for (size_t i = 0; i < Dim; ++i) {
    get<flux_tag>(volume_fluxes).get(i) = logical_coordinates(dg_mesh).get(i);
  }

  const size_t ghost_zone_size = 2;

  const bool always_use_subcell = false;
  const bool use_halo = false;

  // set subcell options
  const evolution::dg::subcell::SubcellOptions& subcell_options =
      evolution::dg::subcell::SubcellOptions{
          evolution::dg::subcell::SubcellOptions{
              1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 5.0, 4.0, always_use_subcell,
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
              use_halo,
              all_neighbors_are_doing_dg
                  ? std::optional{std::vector<std::string>{"Block1"}}
                  : std::optional<std::vector<std::string>>{},
              fd_derivative_order},
          TestCreator<Dim>{}};

  auto box =
      db::create<tmpl::list<GhostZoneSize, domain::Tags::Mesh<Dim>,
                            evolution::dg::subcell::Tags::Mesh<Dim>,
                            domain::Tags::Element<Dim>, variables_tag,
                            evolution::dg::subcell::Tags::DataForRdmpTci,
                            evolution::dg::Tags::NeighborMesh<Dim>,
                            evolution::dg::subcell::Tags::SubcellOptions<Dim>>>(
          ghost_zone_size, dg_mesh, subcell_mesh, element, vars,
          // Set RDMP data since it would've been calculated before already.
          evolution::dg::subcell::RdmpTciData{{1.0}, {-1.0}}, neighbor_meshes,
          subcell_options);

  Mesh<Dim> ghost_data_mesh{};
  DirectionMap<Dim, DataVector> data_for_neighbors{};
  evolution::dg::subcell::prepare_neighbor_data<Metavariables<Dim>>(
      make_not_null(&data_for_neighbors), make_not_null(&ghost_data_mesh),
      make_not_null(&box), volume_fluxes);

  CHECK(ghost_data_mesh ==
        (all_neighbors_are_doing_dg ? dg_mesh : subcell_mesh));

  const auto& rdmp_tci_data =
      db::get<evolution::dg::subcell::Tags::DataForRdmpTci>(box);
  CHECK_ITERABLE_APPROX(rdmp_tci_data.min_variables_values, DataVector{-1.0});
  CHECK_ITERABLE_APPROX(rdmp_tci_data.max_variables_values, DataVector{1.0});

  Variables<tmpl::list<Var1>> expected_vars = vars;
  get(get<Var1>(expected_vars)) *= 2.0;

  DirectionMap<Dim, DataVector> expected_neighbor_data{};

  const bool need_fluxes = fd_derivative_order != ::fd::DerivativeOrder::Two;
  if (all_neighbors_are_doing_dg) {
    DataVector data{expected_vars.size() +
                    (need_fluxes ? volume_fluxes.size() : 0)};
    std::copy(get(get<Var1>(expected_vars)).begin(),
              get(get<Var1>(expected_vars)).end(), data.begin());
    if (need_fluxes) {
      std::copy(volume_fluxes.data(),
                std::next(volume_fluxes.data(),
                          static_cast<std::ptrdiff_t>(volume_fluxes.size())),
                std::next(data.begin(),
                          static_cast<std::ptrdiff_t>(expected_vars.size())));
    }

    for (const auto& direction : expected_neighbor_directions<Dim>()) {
      expected_neighbor_data.insert(std::pair{direction, data});
    }
  } else {
    // Set all directions to false, enable the desired ones below
    std::unordered_set<Direction<Dim>> directions_to_slice{};

    REQUIRE(data_for_neighbors.size() == Dim);
    const size_t num_ghost_points =
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size;
    for (const auto& direction : expected_neighbor_directions<Dim>()) {
      REQUIRE(data_for_neighbors.contains(direction));
      REQUIRE(data_for_neighbors.at(direction).size() ==
              num_ghost_points * (need_fluxes ? (Dim + 1) : 1) + 2);
      directions_to_slice.emplace(direction);
    }

    // do same operation as GhostDataToSlice
    expected_neighbor_data = [&subcell_mesh, &directions_to_slice, &dg_mesh,
                              &expected_vars, need_fluxes, &volume_fluxes]() {
      if (need_fluxes) {
        Variables<tmpl::list<Var1, flux_tag>> expected_var_and_flux{
            expected_vars.number_of_grid_points()};
        get<Var1>(expected_var_and_flux) = get<Var1>(expected_vars);
        get<flux_tag>(expected_var_and_flux) = get<flux_tag>(volume_fluxes);
        return evolution::dg::subcell::slice_data(
            evolution::dg::subcell::fd::project(expected_var_and_flux, dg_mesh,
                                                subcell_mesh.extents()),
            subcell_mesh.extents(), ghost_zone_size, directions_to_slice, 0);
      } else {
        return evolution::dg::subcell::slice_data(
            evolution::dg::subcell::fd::project(expected_vars, dg_mesh,
                                                subcell_mesh.extents()),
            subcell_mesh.extents(), ghost_zone_size, directions_to_slice, 0);
      }
    }();
    if constexpr (Dim == 3) {
      const auto direction = expected_neighbor_directions<Dim>()[1];
      Index<Dim> slice_extents = subcell_mesh.extents();
      slice_extents[direction.dimension()] = ghost_zone_size;
      expected_neighbor_data.at(direction) =
          orient_variables(expected_neighbor_data.at(direction), slice_extents,
                           element.neighbors().at(direction).orientation());
    }
  }

  for (const auto& direction : expected_neighbor_directions<Dim>()) {
    const auto& data_in_direction = data_for_neighbors.at(direction);
    CHECK_ITERABLE_APPROX(
        expected_neighbor_data.at(direction),
        (DataVector{const_cast<double*>(data_in_direction.data()),
                    data_in_direction.size() - 2}));
    CHECK(*std::prev(data_in_direction.end(), 2) == approx(1.0));
    CHECK(*std::prev(data_in_direction.end(), 1) == approx(-1.0));
  }
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.PrepareNeighborData",
                  "[Evolution][Unit]") {
  for (const auto& [all_neighbors_are_doing_dg, fd_deriv_order] :
       cartesian_product(make_array(true, false),
                         make_array(::fd::DerivativeOrder::Two,
                                    ::fd::DerivativeOrder::Four))) {
    test<1>(all_neighbors_are_doing_dg, fd_deriv_order);
    test<2>(all_neighbors_are_doing_dg, fd_deriv_order);
    test<3>(all_neighbors_are_doing_dg, fd_deriv_order);
  }
}
