// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/BackgroundGrVars.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct MetavariablesForTest {
  using component_list = tmpl::list<>;
  using initial_data_list = tmpl::list<RelativisticEuler::Solutions::TovStar>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>>;
  };
};

struct SystemForTest {
  static constexpr size_t volume_dim = 3;

  // A disparate set of GR variables were chosen here to make sure that the
  // action allocates and assigns metric variables without missing any tags
  using spacetime_variables_tag = ::Tags::Variables<
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>>>;
  using flux_spacetime_variables_tag =
      ::Tags::Variables<tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, 3>>>;
  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, 3>;
};

// A free function returning a moving brick domain
template <bool mesh_is_moving>
domain::creators::Brick create_a_brick(const size_t num_dg_pts,
                                       const double initial_time) {
  auto time_dependence_ptr = [&]() {
    if constexpr (mesh_is_moving) {
      const std::array<double, 3> mesh_velocity{1, 2, 3};
      return std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          initial_time, mesh_velocity);
    } else {
      return nullptr;
    }
  }();
  const auto lower_bounds = make_array<3, double>(3.0);
  const auto upper_bounds = make_array<3, double>(5.0);
  const auto refinement_levels = make_array<3, size_t>(0);
  return domain::creators::Brick(lower_bounds, upper_bounds, refinement_levels,
                                 make_array<3, size_t>(num_dg_pts),
                                 make_array<3, bool>(true),
                                 std::move(time_dependence_ptr));
}

// if `testing_runtime_initial_data` == false, test for the compile time types
// of initial data
template <bool test_for_moving_mesh, bool testing_runtime_initial_data>
void test(const gsl::not_null<std::mt19937*> gen) {
  // The test is done as follows :
  //
  // - Create a 3D element (brick) for the test. If `test_for_moving_mesh` ==
  //    `true`, the coordinate map of the brick is set to be time-dependent.
  // - Use Kerr-Schild or TOV solution as the background metric, depending on
  //   the type (compile or runtime) of initial data to test.
  // - Create a box for running the `BackgroundGrVars` mutator.
  // - Run the mutator at the initial time, check results
  // - Change inertial coordinates and time to a random later moment, and test
  //    the mutator again.
  //

  const double initial_time = 0.5;
  // make the random time strictly different from the initial time
  std::uniform_real_distribution<> distribution_time(1.0, 2.0);
  const double random_time{
      make_with_random_values<double>(gen, make_not_null(&distribution_time))};

  // Create a 3D element [3.0, 5.0]^3  for the test
  const size_t num_dg_pts = 5;
  const auto brick = [&]() {
    if constexpr (test_for_moving_mesh) {
      return create_a_brick<true>(num_dg_pts, initial_time);
    } else {
      return create_a_brick<false>(num_dg_pts, initial_time);
    }
  }();
  const auto domain = brick.create_domain();
  const auto element_id = ElementId<3>{0};
  Element<3> element = domain::Initialization::create_initial_element(
      element_id, domain.blocks().at(0),
      std::vector<std::array<size_t, 3>>{{0, 0, 0}});

  const Mesh<3> mesh{num_dg_pts, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  const auto compute_inertial_coords = [&brick, &domain, &element_id,
                                        &mesh](const double time) {
    const auto& block = domain.blocks()[element_id.block_id()];
    const auto element_map = ElementMap<3, Frame::Grid>{
        element_id, block.is_time_dependent()
                        ? block.moving_mesh_logical_to_grid_map().get_clone()
                        : block.stationary_map().get_to_grid_frame()};
    std::unique_ptr<
        ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
        grid_to_inertial_map;
    if (block.is_time_dependent()) {
      grid_to_inertial_map =
          block.moving_mesh_grid_to_inertial_map().get_clone();
    } else {
      grid_to_inertial_map =
          ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              ::domain::CoordinateMaps::Identity<3>{});
    }
    return (*grid_to_inertial_map)(element_map(logical_coordinates(mesh)), time,
                                   brick.functions_of_time());
  };

  const auto initial_inertial_coords = compute_inertial_coords(initial_time);

  using gr_variables_tag =
      ::Tags::Variables<tmpl::remove_duplicates<tmpl::append<
          typename SystemForTest::spacetime_variables_tag::tags_list,
          typename SystemForTest::flux_spacetime_variables_tag::tags_list,
          tmpl::list<typename SystemForTest::inverse_spatial_metric_tag>>>>;

  const auto solution = []() {
    if constexpr (testing_runtime_initial_data) {
      return RelativisticEuler::Solutions::TovStar{
          1.0e-3,
          EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.get_clone(),
          RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
    } else {
      return gr::Solutions::KerrSchild{1.0, make_array<3, double>(0.0),
                                       make_array<3, double>(0.0)};
    }
  }();

  // Note the argument `gr_variables_tag` when creating a box. Since we want to
  // test that the dg::BackgroundGrVars mutator properly initializes (allocate +
  // assign) the background GR variables, use an empty Variables object here for
  // creation.
  auto box = [&initial_time, &brick, &element, &mesh, &initial_inertial_coords,
              &solution]() {
    if constexpr (testing_runtime_initial_data) {
      return db::create<db::AddSimpleTags<
          ::Tags::Time, domain::Tags::Domain<3>, domain::Tags::Element<3>,
          domain::Tags::Mesh<3>, domain::Tags::Coordinates<3, Frame::Inertial>,
          gr_variables_tag, evolution::initial_data::Tags::InitialData>>(
          initial_time, brick.create_domain(), element, mesh,
          initial_inertial_coords, typename gr_variables_tag::type{},
          solution.get_clone());
    } else {
      return db::create<db::AddSimpleTags<
          ::Tags::Time, domain::Tags::Domain<3>, domain::Tags::Element<3>,
          domain::Tags::Mesh<3>, domain::Tags::Coordinates<3, Frame::Inertial>,
          gr_variables_tag,
          ::Tags::AnalyticSolution<gr::Solutions::KerrSchild>>>(
          initial_time, brick.create_domain(), element, mesh,
          initial_inertial_coords, typename gr_variables_tag::type{}, solution);
    }
  }();

  // Apply the mutator for initialization phase, and check that it has put
  // correct values of GR variables in the box.
  db::mutate_apply<evolution::dg::BackgroundGrVars<
      SystemForTest, MetavariablesForTest, testing_runtime_initial_data>>(
      make_not_null(&box));

  const auto expected_initial_gr_vars = solution.variables(
      initial_inertial_coords, initial_time, gr_variables_tag::tags_list{});
  tmpl::for_each<gr_variables_tag::tags_list>(
      [&box, &expected_initial_gr_vars](const auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const auto& gr_vars_in_box = get<gr_variables_tag>(box);
        CHECK_ITERABLE_APPROX(get<tag>(expected_initial_gr_vars),
                              get<tag>(gr_vars_in_box));
      });

  // Mutate time and inertial coords to those at t = `random_time` and apply the
  // mutator again.. Then check that the mutator has evaluated correct values of
  // GR variables at a later random time.
  const auto inertial_coords = compute_inertial_coords(random_time);
  db::mutate<::Tags::Time, domain::Tags::Coordinates<3, Frame::Inertial>>(
      [&random_time, &inertial_coords](const auto time_ptr,
                                       const auto inertial_coords_ptr) {
        *time_ptr = random_time;
        *inertial_coords_ptr = inertial_coords;
      },
      make_not_null(&box));

  db::mutate_apply<evolution::dg::BackgroundGrVars<
      SystemForTest, MetavariablesForTest, testing_runtime_initial_data>>(
      make_not_null(&box));

  if constexpr (test_for_moving_mesh) {
    const auto expected_gr_vars = solution.variables(
        inertial_coords, random_time, gr_variables_tag::tags_list{});
    tmpl::for_each<gr_variables_tag::tags_list>(
        [&box, &expected_gr_vars](const auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          const auto& gr_vars_in_box = get<gr_variables_tag>(box);
          CHECK_ITERABLE_APPROX(get<tag>(expected_gr_vars),
                                get<tag>(gr_vars_in_box));
        });
  } else {
    tmpl::for_each<gr_variables_tag::tags_list>(
        [&box, &expected_initial_gr_vars](const auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          const auto& gr_vars_in_box = get<gr_variables_tag>(box);
          CHECK_ITERABLE_APPROX(get<tag>(expected_initial_gr_vars),
                                get<tag>(gr_vars_in_box));
        });
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.BackgroundGrVars", "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  test<false, false>(make_not_null(&gen));
  test<false, true>(make_not_null(&gen));
  test<true, false>(make_not_null(&gen));
  test<true, true>(make_not_null(&gen));
}

}  // namespace
