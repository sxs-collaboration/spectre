// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using initial_tags = tmpl::list<
      evolution::dg::subcell::Tags::Mesh<Dim>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToGrid,
      evolution::dg::Tags::MortarData<Dim>>;

  using initial_compute_tags =
      tmpl::list<evolution::dg::subcell::Tags::LogicalCoordinatesCompute<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags, initial_compute_tags>,
          evolution::dg::subcell::fd::Actions::TakeTimeStep<
              typename Metavariables::TimeDerivative>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
  enum class Phase { Initialization, Exit };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool time_derivative_invoked;

  struct TimeDerivative {
    template <typename DbTagsList>
    static void apply(
        const gsl::not_null<db::DataBox<DbTagsList>*> box,
        const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Grid>&
            cell_centered_logical_to_grid_inv_jacobian,
        const Scalar<DataVector>& cell_centered_det_inv_jacobian) noexcept {
      time_derivative_invoked = true;
      CHECK(db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box) ==
            Mesh<Dim>(5, Spectral::Basis::FiniteDifference,
                      Spectral::Quadrature::CellCentered));
      const auto inv_jacobian =
          db::get<domain::Tags::ElementMap<Dim, Frame::Grid>>(*box)
              .inv_jacobian(db::get<evolution::dg::subcell::Tags::Coordinates<
                                Dim, Frame::Logical>>(*box));
      CHECK(cell_centered_logical_to_grid_inv_jacobian == inv_jacobian);
      const auto det_inv_jacobian = determinant(inv_jacobian);
      CHECK_ITERABLE_APPROX(cell_centered_det_inv_jacobian, det_inv_jacobian);
    }
  };
};

template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::time_derivative_invoked = false;

template <size_t Dim>
auto make_grid_map() {
  using domain::make_coordinate_map_base;
  using domain::CoordinateMaps::Affine;
  if constexpr (Dim == 1) {
    return make_coordinate_map_base<Frame::Logical, Frame::Grid>(
        Affine(-1.0, 1.0, 2.0, 5.0));
  } else if constexpr (Dim == 2) {
    using domain::CoordinateMaps::ProductOf2Maps;
    return make_coordinate_map_base<Frame::Logical, Frame::Grid>(
        ProductOf2Maps<Affine, Affine>(Affine(-1.0, 1.0, -1.0, -0.8),
                                       Affine(-1.0, 1.0, -1.0, -0.8)),
        domain::CoordinateMaps::Wedge<2>(0.5, 0.75, 1.0, 1.0, {}, false));
  } else {
    using domain::CoordinateMaps::ProductOf3Maps;
    return make_coordinate_map_base<Frame::Logical, Frame::Grid>(
        ProductOf3Maps<Affine, Affine, Affine>(Affine(-1.0, 1.0, -1.0, -0.8),
                                               Affine(-1.0, 1.0, -1.0, -0.8),
                                               Affine(-1.0, 1.0, 0.8, 1.0)),
        domain::CoordinateMaps::Wedge<3>(0.5, 0.75, 1.0, 1.0, {}, false));
  }
}

template <size_t Dim>
auto make_inertial_map() {
  return domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      domain::CoordinateMaps::Identity<Dim>{});
}

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  using metavars = Metavariables<Dim>;
  metavars::time_derivative_invoked = false;

  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};

  const Mesh<Dim> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  // Set up nonsense mortar data since we only need to check that it got
  // cleared.
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  std::unordered_map<Key, evolution::dg::MortarData<Dim>, boost::hash<Key>>
      mortar_data{};
  evolution::dg::MortarData<Dim> lower_xi_data{};
  lower_xi_data.insert_local_mortar_data(
      TimeStepId{true, 1, Time{Slab{1.2, 7.8}, {1, 10}}},
      subcell_mesh.slice_away(0), std::vector<double>{1.1, 2.43, 7.8});
  const std::pair lower_id{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};
  mortar_data[lower_id] = lower_xi_data;

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {subcell_mesh,
       ElementMap<Dim, Frame::Grid>{ElementId<Dim>{0}, make_grid_map<Dim>()},
       make_inertial_map<Dim>(), std::nullopt, std::nullopt, mortar_data});

  CHECK(ActionTesting::get_databox_tag<comp,
                                       evolution::dg::Tags::MortarData<Dim>>(
            runner, 0)
            .at(lower_id)
            .local_mortar_data()
            .has_value());

  // Invoke the TakeTimeStep action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  CHECK_FALSE(ActionTesting::get_databox_tag<
                  comp, evolution::dg::Tags::MortarData<Dim>>(runner, 0)
                  .at(lower_id)
                  .local_mortar_data()
                  .has_value());
  CHECK(metavars::time_derivative_invoked);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.Actions.TakeTimeStep",
                  "[Evolution][Unit]") {
  using Affine = domain::CoordinateMaps::Affine;
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Grid, Affine>));
  PUPable_reg(SINGLE_ARG(domain::CoordinateMap<
                         Frame::Logical, Frame::Grid,
                         domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>,
                         domain::CoordinateMaps::Wedge<2>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<
                 Frame::Logical, Frame::Grid,
                 domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>,
                 domain::CoordinateMaps::Wedge<3>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<1>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<2>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<3>>));

  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
