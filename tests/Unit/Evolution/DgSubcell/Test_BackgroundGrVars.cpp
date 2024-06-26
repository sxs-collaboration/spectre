// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/BackgroundGrVars.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
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
template <bool MeshIsMoving>
domain::creators::Brick create_a_brick(const size_t num_dg_pts,
                                       const double initial_time) {
  auto time_dependence_ptr = [&]() {
    if constexpr (MeshIsMoving) {
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

// A free function that creates and returns face-centered subcell mesh
std::array<Mesh<3>, 3> create_face_centered_meshes(
    const Mesh<3> cell_centered_mesh) {
  std::array<Mesh<3>, 3> face_centered_meshes{};
  for (size_t dim = 0; dim < 3; ++dim) {
    const auto basis = make_array<3>(cell_centered_mesh.basis(0));
    auto quadrature = make_array<3>(cell_centered_mesh.quadrature(0));
    auto extents = make_array<3>(cell_centered_mesh.extents(0));
    gsl::at(extents, dim) = cell_centered_mesh.extents(0) + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    gsl::at(face_centered_meshes, dim) = Mesh<3>{extents, basis, quadrature};
  }
  return face_centered_meshes;
}

// if `TestRuntimeInitialData` == false, test for the compile time initial
// data
template <bool TestMovingMesh, bool TestRuntimeInitialData,
          bool ComputeOnlyOnRollback>
void test(const gsl::not_null<std::mt19937*> gen, const bool did_rollback) {
  //
  // The test is done as follows :
  //
  // - Create a 3D element (brick) for the test. If `TestMovingMesh ` ==
  //    `true`, the coordinate map of the brick is set to be time-dependent.
  // - Use Kerr-Schild or TOV solution as the background metric, depending on
  //   the type (compile or runtime) of initial data to test.
  // - Create a box for running the `BackgroundGrVars` mutator.
  // - Run the mutator at the initial time (=initialization phase), then check
  //   the result
  // - Change inertial coordinates and time to a random later moment, and test
  //   the mutator again.
  //

  CAPTURE(TestMovingMesh);
  CAPTURE(TestRuntimeInitialData);
  CAPTURE(ComputeOnlyOnRollback);
  CAPTURE(did_rollback);

  const double initial_time = 0.5;
  // make a (random) later time strictly different from the initial time
  std::uniform_real_distribution<> distribution_time(1.0, 2.0);
  const double later_time{
      make_with_random_values<double>(gen, make_not_null(&distribution_time))};

  // Create a 3D element [3.0, 5.0]^3  for the test
  const size_t num_dg_pts = 3;
  const auto brick = [&]() {
    if constexpr (TestMovingMesh) {
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

  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh<3>(dg_mesh);

  const auto face_centered_meshes = create_face_centered_meshes(subcell_mesh);

  const auto& block = domain.blocks()[element_id.block_id()];

  const auto element_map = ElementMap<3, Frame::Grid>{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};

  std::unique_ptr<::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      grid_to_inertial_map;
  if (block.is_time_dependent()) {
    grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map().get_clone();
  } else {
    grid_to_inertial_map =
        ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<3>{});
  }

  const auto compute_inertial_coords = [&grid_to_inertial_map, &element_map,
                                        &brick](const Mesh<3> mesh,
                                                const double time) {
    return (*grid_to_inertial_map)(element_map(logical_coordinates(mesh)), time,
                                   brick.functions_of_time());
  };

  const auto subcell_initial_inertial_coords =
      compute_inertial_coords(subcell_mesh, initial_time);
  const auto subcell_later_inertial_coords =
      compute_inertial_coords(subcell_mesh, later_time);

  std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 3>
      face_centered_initial_inertial_coords{};
  std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 3>
      face_centered_later_inertial_coords{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(face_centered_initial_inertial_coords, i) =
        compute_inertial_coords(gsl::at(face_centered_meshes, i), initial_time);
    gsl::at(face_centered_later_inertial_coords, i) =
        compute_inertial_coords(gsl::at(face_centered_meshes, i), later_time);
  }

  using gr_variables_tag =
      ::Tags::Variables<SystemForTest::spacetime_variables_tag::tags_list>;
  using inactive_gr_variables_tag =
      evolution::dg::subcell::Tags::Inactive<gr_variables_tag>;
  using subcell_face_gr_variables_tag =
      evolution::dg::subcell::Tags::OnSubcellFaces<
          typename SystemForTest::flux_spacetime_variables_tag, 3>;

  const auto solution = []() {
    if constexpr (TestRuntimeInitialData) {
      return RelativisticEuler::Solutions::TovStar{
          1.0e-3,
          EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.get_clone(),
          RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
    } else {
      return gr::Solutions::KerrSchild{1.0, make_array<3, double>(0.0),
                                       make_array<3, double>(0.0)};
    }
  }();

  const auto dg_gr_vars = [&compute_inertial_coords, &dg_mesh, &initial_time,
                           &solution]() {
    gr_variables_tag::type gr_vars{dg_mesh.number_of_grid_points()};

    gr_vars.assign_subset(evolution::Initialization::initial_data(
        solution, compute_inertial_coords(dg_mesh, initial_time), initial_time,
        typename gr_variables_tag::tags_list{}));

    return gr_vars;
  }();

  auto box = [&block, &brick, &dg_gr_vars, &element, &element_id,
              &grid_to_inertial_map, &initial_time, &solution,
              &subcell_initial_inertial_coords, &subcell_mesh]() {
    // Bug in GCC 13.2 where having this class created in-place below causes an
    // internal compiler error
    typename subcell_face_gr_variables_tag::type face_gr_vars{};
    // Since we want to test that the BackgroundGrVars action properly
    // initializes (allocate + assign) the background GR variables on
    // cell-centered and face-centered coordinates, use an empty subcell GR
    // variables objects here for creating a box. GR variables on DG mesh are
    // considered as initialized.
    if constexpr (TestRuntimeInitialData) {
      return db::create<db::AddSimpleTags<
          ::Tags::Time, domain::Tags::Domain<3>, domain::Tags::Element<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::FunctionsOfTimeInitialize,
          evolution::dg::subcell::Tags::Mesh<3>,
          evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
          gr_variables_tag, inactive_gr_variables_tag,
          subcell_face_gr_variables_tag,
          evolution::dg::subcell::Tags::DidRollback,
          evolution::initial_data::Tags::InitialData>>(
          initial_time, brick.create_domain(), element,
          ElementMap<3, Frame::Grid>{
              element_id,
              block.is_time_dependent()
                  ? block.moving_mesh_logical_to_grid_map().get_clone()
                  : block.stationary_map().get_to_grid_frame()},
          std::move(grid_to_inertial_map),
          clone_unique_ptrs(brick.functions_of_time()), subcell_mesh,
          subcell_initial_inertial_coords, dg_gr_vars,
          typename inactive_gr_variables_tag::type{}, face_gr_vars, false,
          solution.get_clone());
    } else {
      return db::create<db::AddSimpleTags<
          ::Tags::Time, domain::Tags::Domain<3>, domain::Tags::Element<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::FunctionsOfTimeInitialize,
          evolution::dg::subcell::Tags::Mesh<3>,
          evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
          gr_variables_tag, inactive_gr_variables_tag,
          subcell_face_gr_variables_tag,
          evolution::dg::subcell::Tags::DidRollback,
          ::Tags::AnalyticSolution<gr::Solutions::KerrSchild>>>(
          initial_time, brick.create_domain(), element,
          ElementMap<3, Frame::Grid>{
              element_id,
              block.is_time_dependent()
                  ? block.moving_mesh_logical_to_grid_map().get_clone()
                  : block.stationary_map().get_to_grid_frame()},
          std::move(grid_to_inertial_map),
          clone_unique_ptrs(brick.functions_of_time()), subcell_mesh,
          subcell_initial_inertial_coords, dg_gr_vars,
          typename inactive_gr_variables_tag::type{}, face_gr_vars, false,
          solution);
    }
  }();

  // Apply the mutator for initialization phase, and check that it has put
  // correct values of GR variables in the box. In the initialization phase,
  // `inactive_gr_variables_tag` and `subcell_face_gr_variables_tag` must be
  // properly initialized.
  db::mutate_apply<evolution::dg::subcell::BackgroundGrVars<
      SystemForTest, MetavariablesForTest, TestRuntimeInitialData,
      ComputeOnlyOnRollback>>(make_not_null(&box));

  // Compute expected cell-centered and face-centered GR vars
  const auto expected_initial_cell_centered_gr_vars =
      solution.variables(subcell_initial_inertial_coords, initial_time,
                         gr_variables_tag::tags_list{});
  subcell_face_gr_variables_tag::type expected_initial_face_centered_gr_vars{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(expected_initial_face_centered_gr_vars, d)
        .initialize(gsl::at(face_centered_meshes, 0).number_of_grid_points());
    gsl::at(expected_initial_face_centered_gr_vars, d)
        .assign_subset(evolution::Initialization::initial_data(
            solution, gsl::at(face_centered_initial_inertial_coords, d),
            initial_time, subcell_face_gr_variables_tag::tag::tags_list{}));
  }

  // some helper functions
  const auto check_cell_centered_vars = [&box](const auto expected_values,
                                               const bool is_active) {
    tmpl::for_each<gr_variables_tag::tags_list>(
        [&box, &expected_values, &is_active](const auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;

          const auto var_in_box = [&box, &is_active]() {
            if (is_active) {
              return get<tag>(get<gr_variables_tag>(box));
            } else {
              return get<evolution::dg::subcell::Tags::Inactive<tag>>(
                  get<inactive_gr_variables_tag>(box));
            }
          }();
          const auto var_expected = get<tag>(expected_values);

          CHECK_ITERABLE_APPROX(var_in_box, var_expected);
        });
  };
  const auto check_face_centered_vars = [&box](const auto expected_values) {
    tmpl::for_each<subcell_face_gr_variables_tag::tag::tags_list>(
        [&box, &expected_values](const auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          for (size_t d = 0; d < 3; ++d) {
            const auto var_in_box =
                get<tag>(gsl::at(get<subcell_face_gr_variables_tag>(box), d));
            const auto var_expected = get<tag>(gsl::at(expected_values, d));

            CHECK_ITERABLE_APPROX(var_in_box, var_expected);
          }
        });
  };

  // check results for the initialization phase
  check_cell_centered_vars(expected_initial_cell_centered_gr_vars, false);
  check_face_centered_vars(expected_initial_face_centered_gr_vars);

  // Mutate time and inertial coords to those at t = `later_time`, mutate the
  // `DidRollback` tag to `did_rollback`, and apply the mutator again.
  db::mutate<::Tags::Time,
             evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
             evolution::dg::subcell::Tags::DidRollback>(
      [&later_time, &subcell_later_inertial_coords, &did_rollback](
          const auto time_ptr, const auto inertial_coords_ptr,
          const auto did_rollback_ptr) {
        *time_ptr = later_time;
        *inertial_coords_ptr = subcell_later_inertial_coords;
        *did_rollback_ptr = did_rollback;
      },
      make_not_null(&box));

  //
  // Now, we have a number of different cases that need to be handled :
  //
  //  * If `TestMovingMesh ` == false :
  //     this mutator should not change anything
  //
  //  * If `TestMovingMesh ` == true :
  //
  //    - if `did_rollback` == true :
  //      This is the beginning of FD solve after rollback, before
  //      SwapGrTags mutator is applied. Therefore
  //      `inactive_gr_variables_tag` and `subcell_face_gr_variables_tag`
  //      need to be modified.
  //
  //    - if `did_rollback` == false :
  //
  //       - if `ComputeOnlyOnRollback` == true :
  //           This is when the element is doing FD and passing the
  //           `Labels::BeginSubcellAfterDgRollback` label in the action list.
  //           So this mutator should not change anything.
  //
  //       - if `ComputeOnlyOnRollback` == false :
  //           This is when the element is doing FD and just started the FD
  //           solve. In this case, `active_gr_variables_tag` and
  //           `subcell_face_gr_variables_tag` need to be modified. We manually
  //           swap GR tags in the test databox to mimic the situation.
  //
  if (TestMovingMesh and not did_rollback and not ComputeOnlyOnRollback) {
    db::mutate<gr_variables_tag, inactive_gr_variables_tag>(
        [](const auto active_gr_vars_ptr, const auto inactive_gr_vars_ptr) {
          using std::swap;
          swap(*active_gr_vars_ptr, *inactive_gr_vars_ptr);
        },
        make_not_null(&box));
  }

  db::mutate_apply<evolution::dg::subcell::BackgroundGrVars<
      SystemForTest, MetavariablesForTest, TestRuntimeInitialData,
      ComputeOnlyOnRollback>>(make_not_null(&box));

  //
  // Chcek the results. All the `if` branches below are organized in the same
  // order as the cases presented above.
  //
  if constexpr (not TestMovingMesh) {
    check_cell_centered_vars(expected_initial_cell_centered_gr_vars, false);
    check_face_centered_vars(expected_initial_face_centered_gr_vars);
  } else {
    // Compute expected values of cell-centered and face-centered GR vars
    const auto expected_later_cell_centered_gr_vars =
        solution.variables(subcell_later_inertial_coords, later_time,
                           gr_variables_tag::tags_list{});

    subcell_face_gr_variables_tag::type expected_later_face_centered_gr_vars{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(expected_later_face_centered_gr_vars, d)
          .initialize(gsl::at(face_centered_meshes, 0).number_of_grid_points());
      gsl::at(expected_later_face_centered_gr_vars, d)
          .assign_subset(evolution::Initialization::initial_data(
              solution, gsl::at(face_centered_later_inertial_coords, d),
              later_time, subcell_face_gr_variables_tag::tag::tags_list{}));
    }

    if (did_rollback) {
      check_cell_centered_vars(expected_later_cell_centered_gr_vars, false);
      check_face_centered_vars(expected_later_face_centered_gr_vars);
    } else {
      if constexpr (ComputeOnlyOnRollback) {
        check_cell_centered_vars(expected_initial_cell_centered_gr_vars, false);
        check_face_centered_vars(expected_initial_face_centered_gr_vars);
      } else {
        check_cell_centered_vars(expected_later_cell_centered_gr_vars, true);
        check_face_centered_vars(expected_later_face_centered_gr_vars);
      }
    }
  }
}

template <bool TestMovingMesh, bool TestRuntimeInitialData>
void test_for_rollback(const gsl::not_null<std::mt19937*> gen) {
  test<TestMovingMesh, TestRuntimeInitialData, true>(gen, true);
  test<TestMovingMesh, TestRuntimeInitialData, true>(gen, false);

  test<TestMovingMesh, TestRuntimeInitialData, false>(gen, true);
  test<TestMovingMesh, TestRuntimeInitialData, false>(gen, false);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.BackgroundGrVars",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  test_for_rollback<true, true>(make_not_null(&gen));
  test_for_rollback<true, false>(make_not_null(&gen));

  test_for_rollback<false, true>(make_not_null(&gen));
  test_for_rollback<false, false>(make_not_null(&gen));
}

}  // namespace
