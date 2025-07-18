// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/SetPiAndPhiFromConstraints.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using evolved_vars_tags =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags = db::AddSimpleTags<
      ::Tags::Time, ::Tags::Variables<evolved_vars_tags>, domain::Tags::Mesh<3>,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::ActiveGrid>;
  using compute_tags = db::AddComputeTags<
      evolution::dg::subcell::Tags::MeshCompute<3>,
      evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<grmhd::GhValenciaDivClean::SetPiAndPhiFromConstraints>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
};

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean."
    "SetPiAndPhiFromConstraints",
    "[Unit][Evolution][Actions]") {
  MAKE_GENERATOR(generator);
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  register_classes_with_charm<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Grid,
                            domain::CoordinateMaps::Identity<3>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<3>>,
      gh::gauges::DampedHarmonic>();

  std::uniform_real_distribution<> metric_dist(0.1, 1.);
  std::uniform_real_distribution<> deriv_dist(-1.e-5, 1.e-5);

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto make_vars = [&generator, &deriv_dist,
                          &metric_dist](const Mesh<3>& mesh) {
    const size_t num_points = mesh.number_of_grid_points();
    Variables<evolved_vars_tags> evolved_vars{mesh.number_of_grid_points()};
    get<gh::Tags::Pi<DataVector, 3>>(evolved_vars) =
        make_with_random_values<tnsr::aa<DataVector, 3, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&deriv_dist), num_points);
    get<gh::Tags::Phi<DataVector, 3>>(evolved_vars) =
        make_with_random_values<tnsr::iaa<DataVector, 3, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&deriv_dist), num_points);
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved_vars) =
        make_with_random_values<tnsr::aa<DataVector, 3, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&metric_dist), num_points);
    get<0, 0>(get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved_vars)) +=
        -2.0;
    for (size_t i = 0; i < 3; ++i) {
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved_vars)
          .get(i + 1, i + 1) += 4.0;
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved_vars)
          .get(i + 1, 0) *= 0.01;
    }
    return evolved_vars;
  };

  const auto initial_dg_vars = make_vars(dg_mesh);

  // Testing SetPiAndPhiFromConstraints = False.
  {
    MockRuntimeSystem runner{
        {std::unique_ptr<gh::gauges::GaugeCondition>(
            std::make_unique<gh::gauges::DampedHarmonic>(
                100., std::array{1.2, 1.5, 1.7}, std::array{2, 4, 6}))},
        {false}};
    ActionTesting::emplace_component_and_initialize<component>(
        &runner, 0,
        {0., initial_dg_vars, dg_mesh,
         ElementMap<3, Frame::Grid>{
             ElementId<3>{0},
             domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                 domain::CoordinateMaps::Identity<3>{})},
         domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
             domain::CoordinateMaps::Identity<3>{}),
         std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
         logical_coordinates(dg_mesh), evolution::dg::subcell::ActiveGrid::Dg});

    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
    ActionTesting::next_action<component>(make_not_null(&runner), 0);

    // Should be exact since we didn't compute anything
    const auto& box = ActionTesting::get_databox<component>(runner, 0);
    CHECK(get<gh::Tags::Pi<DataVector, 3>>(initial_dg_vars) ==
          db::get<gh::Tags::Pi<DataVector, 3>>(box));
    CHECK(get<gh::Tags::Phi<DataVector, 3>>(initial_dg_vars) ==
          db::get<gh::Tags::Phi<DataVector, 3>>(box));
  }

  MockRuntimeSystem runner{
      {std::unique_ptr<gh::gauges::GaugeCondition>(
          std::make_unique<gh::gauges::DampedHarmonic>(
              100., std::array{1.2, 1.5, 1.7}, std::array{2, 4, 6}))},
      {true}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {0., initial_dg_vars, dg_mesh,
       ElementMap<3, Frame::Grid>{
           ElementId<3>{0},
           domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
               domain::CoordinateMaps::Identity<3>{})},
       domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
           domain::CoordinateMaps::Identity<3>{}),
       std::unordered_map<
           std::string,
           std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
       logical_coordinates(dg_mesh), evolution::dg::subcell::ActiveGrid::Dg});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& box = ActionTesting::get_databox<component>(make_not_null(&runner), 0);

  const auto initial_subcell_vars =
      make_vars(db::get<evolution::dg::subcell::Tags::Mesh<3>>(box));

  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  const auto check = [&box](const Mesh<3>& mesh, const auto& initial_vars) {
    CAPTURE(mesh);
    tnsr::aa<DataVector, 3, Frame::Inertial> expected_pi =
        get<gh::Tags::Pi<DataVector, 3>>(initial_vars);
    tnsr::iaa<DataVector, 3, Frame::Inertial> expected_phi =
        get<gh::Tags::Phi<DataVector, 3>>(initial_vars);
    gh::gauges::SetPiAndPhiFromConstraints<
        ghmhd::GhValenciaDivClean::InitialData::
            analytic_solutions_and_data_list,
        3>::impl(make_not_null(&expected_pi), make_not_null(&expected_phi), 0.,
                 mesh, db::get<domain::Tags::ElementMap<3, Frame::Grid>>(box),
                 db::get<domain::CoordinateMaps::Tags::CoordinateMap<
                     3, Frame::Grid, Frame::Inertial>>(box),
                 db::get<domain::Tags::FunctionsOfTime>(box),
                 logical_coordinates(mesh),
                 get<gr::Tags::SpacetimeMetric<DataVector, 3>>(initial_vars),
                 db::get<gh::gauges::Tags::GaugeCondition>(box), true);

    const auto& pi = db::get<gh::Tags::Pi<DataVector, 3>>(box);
    CHECK(pi == expected_pi);
    const auto& phi = db::get<gh::Tags::Phi<DataVector, 3>>(box);
    CHECK(phi == expected_phi);
  };

  // Check that initial grid is fine.
  check(db::get<domain::Tags::Mesh<3>>(box), initial_dg_vars);

  // Switch to subcell grid and check we compute Pi correctly
  db::mutate<evolution::dg::subcell::Tags::ActiveGrid,
             ::Tags::Variables<evolved_vars_tags>>(
      [&initial_subcell_vars](const auto active_grid_ptr,
                              const auto variables_ptr) {
        *active_grid_ptr = evolution::dg::subcell::ActiveGrid::Subcell;
        *variables_ptr = initial_subcell_vars;
      },
      make_not_null(&box));
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  check(db::get<evolution::dg::subcell::Tags::Mesh<3>>(box),
        initial_subcell_vars);

  // // Switch back to DG and check we compute Pi correctly
  db::mutate<evolution::dg::subcell::Tags::ActiveGrid,
             ::Tags::Variables<evolved_vars_tags>>(
      [&initial_dg_vars](const auto active_grid_ptr, const auto variables_ptr) {
        *active_grid_ptr = evolution::dg::subcell::ActiveGrid::Dg;
        *variables_ptr = initial_dg_vars;
      },
      make_not_null(&box));
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  check(db::get<domain::Tags::Mesh<3>>(box), initial_dg_vars);
}
}  // namespace
