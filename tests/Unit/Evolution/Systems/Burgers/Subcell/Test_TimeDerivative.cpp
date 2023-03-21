// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Burgers/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// These solution tag and metavariables are not strictly required for testing
// subcell time derivative, but needed for compilation since
// BoundaryConditionGhostData requires this to be in box.
struct DummyAnalyticSolutionTag : db::SimpleTag, Tags::AnalyticSolutionOrData {
  using type = Burgers::Solutions::Step;
};

struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = false;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Burgers::BoundaryConditions::BoundaryCondition,
                   Burgers::BoundaryConditions::standard_boundary_conditions>>;
  };
};
}  // namespace

namespace Burgers {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.TimeDerivative",
                  "[Unit][Evolution]") {
  using evolved_vars_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, evolved_vars_tag>;

  DirectionMap<1, Neighbors<1>> neighbors{};
  for (size_t i = 0; i < 2; ++i) {
    neighbors[gsl::at(Direction<1>::all_directions(), i)] =
        Neighbors<1>{{ElementId<1>{i + 1, {}}}, {}};
  }
  const Element<1> element{ElementId<1>{0, {}}, neighbors};

  const size_t num_dg_pts = 5;
  const Mesh<1> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // Perform test with MC reconstruction & Rusanov riemann solver
  using ReconstructionForTest = fd::MonotonisedCentral;
  using BoundaryCorrectionForTest = BoundaryCorrections::Rusanov;

  // Set the testing profile for U.
  // Here we use
  //   * U(x)   = x + 1
  const auto compute_test_solution = [](const auto& coords) {
    using tag = Tags::U;
    Variables<tmpl::list<tag>> vars{get<0>(coords).size(), 0.0};
    get(get<tag>(vars)) += coords.get(0) + 1.0;
    return vars;
  };
  auto logical_coords_subcell = logical_coordinates(subcell_mesh);
  const auto volume_vars_subcell =
      compute_test_solution(logical_coords_subcell);

  // set the ghost data from neighbor
  const ReconstructionForTest reconstructor{};
  evolution::dg::subcell::Tags::GhostDataForReconstruction<1>::type ghost_data =
      TestHelpers::Burgers::fd::compute_ghost_data(
          subcell_mesh, logical_coords_subcell, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_test_solution);

  // Below are also dummy variables required for compilation due to boundary
  // condition FD ghost data. Since the element used here for testing has
  // neighbors in all directions, BoundaryConditionGhostData::apply() is not
  // actually called so it is okay to leave these variables somewhat poorly
  // initialized.
  Domain<1> dummy_domain{};
  std::optional<tnsr::I<DataVector, 1>> dummy_volume_mesh_velocity{};
  typename evolution::dg::Tags::NormalCovectorAndMagnitude<1>::type
      dummy_normal_covector_and_magnitude{};
  const double dummy_time{0.0};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      dummy_functions_of_time{};

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<1>, evolution::dg::subcell::Tags::Mesh<1>,
      evolved_vars_tag, dt_variables_tag,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<1>,
      fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection<System>,
      domain::Tags::ElementMap<1, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<1, Frame::ElementLogical>,
      evolution::dg::Tags::MortarData<1>, domain::Tags::Domain<1>,
      domain::Tags::MeshVelocity<1, Frame::Inertial>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<1>, ::Tags::Time,
      domain::Tags::FunctionsOfTimeInitialize, DummyAnalyticSolutionTag,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars>>>(
      element, subcell_mesh, volume_vars_subcell,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      ghost_data,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<ReconstructionForTest>()},
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<BoundaryCorrectionForTest>()},
      ElementMap<1, Frame::Grid>{
          ElementId<1>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<1>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{}),
      logical_coords_subcell,
      typename evolution::dg::Tags::MortarData<1>::type{},
      std::move(dummy_domain), dummy_volume_mesh_velocity,
      dummy_normal_covector_and_magnitude, dummy_time,
      clone_unique_ptrs(dummy_functions_of_time),
      Burgers::Solutions::Step{2.0, 1.0, 0.0}, DummyEvolutionMetaVars{});

  {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{});
    InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
        cell_centered_logical_to_grid_inv_jacobian{};
    const auto cell_centered_logical_to_inertial_inv_jacobian =
        coordinate_map.inv_jacobian(logical_coords_subcell);
    for (size_t i = 0; i < cell_centered_logical_to_grid_inv_jacobian.size();
         ++i) {
      cell_centered_logical_to_grid_inv_jacobian[i] =
          cell_centered_logical_to_inertial_inv_jacobian[i];
    }
    subcell::TimeDerivative::apply(
        make_not_null(&box), cell_centered_logical_to_grid_inv_jacobian,
        determinant(cell_centered_logical_to_grid_inv_jacobian));
  }

  const auto& dt_vars = db::get<dt_variables_tag>(box);

  // Analytic time derivative of U for the testing profile
  //   * dt(U) = -(1+x)
  const auto compute_test_derivative = [](const auto& coords) {
    using tag = ::Tags::dt<Tags::U>;
    Variables<tmpl::list<tag>> dt_expected{get<0>(coords).size(), 0.0};
    get(get<tag>(dt_expected)) += -1.0 - coords.get(0);
    return dt_expected;
  };

  CHECK_ITERABLE_APPROX(get<::Tags::dt<Tags::U>>(dt_vars),
                        get<::Tags::dt<Tags::U>>(
                            compute_test_derivative(logical_coords_subcell)));
}
}  // namespace Burgers
