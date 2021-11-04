// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Events/ObserveFields.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Utilities/CloneUniquePtrs.hpp"

#include "Helpers/ParallelAlgorithms/Events/ObserveFields.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"  // for Variables

template <size_t>
class Index;
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
namespace observers::Actions {
struct ContributeVolumeData;
}  // namespace observers::Actions

namespace {
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace TestHelpers::dg::Events::ObserveFields;

template <size_t Dim>
auto make_map() {
  using domain::make_coordinate_map_base;
  using Translation = domain::CoordinateMaps::TimeDependent::Translation<Dim>;
  return make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      Translation("Translation"));
}

template <typename System, bool AlwaysHasAnalyticSolutions>
void test_observe(
    const std::unique_ptr<Event> observe,
    const std::optional<Mesh<System::volume_dim>>& interpolating_mesh,
    const evolution::dg::subcell::ActiveGrid active_grid) {
  // The subcell code doesn't yet support interpolation.
  REQUIRE_FALSE(interpolating_mesh.has_value());
  CAPTURE(active_grid);

  using metavariables = Metavariables<System, AlwaysHasAnalyticSolutions>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  static constexpr auto volume_dim = System::volume_dim;
  using dg_coordinates_tag =
      ::domain::Tags::Coordinates<volume_dim, Frame::Grid>;
  using subcell_coordinates_tag =
      ::evolution::dg::subcell::Tags::Coordinates<volume_dim, Frame::Grid>;

  const ElementId<volume_dim> element_id(2);
  const typename element_component::array_index array_index(element_id);
  const std::string element_name = get_output(element_id);
  const double observation_time = 2.0;
  const typename System::solution_for_test analytic_solution{};
  using solution_variables = typename System::solution_for_test::vars_for_test;
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  const std::array<DataVector, 4> init_func{{{volume_dim, 1.0},
                                             {volume_dim, -2.0},
                                             {volume_dim, 2.0},
                                             {volume_dim, 0.0}}};
  std::unordered_map<std::string, FoftPtr> functions_of_time{};
  functions_of_time["Translation"] =
      std::make_unique<Polynomial>(0, init_func, 1.0e100);

  const auto grid_to_inertial_map = make_map<volume_dim>();

  const Mesh<volume_dim> dg_mesh(5, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto);
  const auto subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto active_mesh = active_grid == evolution::dg::subcell::ActiveGrid::Dg
                               ? dg_mesh
                               : subcell_mesh;
  const intrp::RegularGrid interpolant(
      active_mesh, interpolating_mesh.value_or(active_mesh));

  Variables<tmpl::list<dg_coordinates_tag>> dg_coords_vars{
      dg_mesh.number_of_grid_points()};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(dg_coords_vars.data(),
            dg_coords_vars.data() + dg_coords_vars.size(), 1.0);
  Variables<tmpl::list<subcell_coordinates_tag>> subcell_coords_vars(
      subcell_mesh.number_of_grid_points());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(subcell_coords_vars.data(),
            subcell_coords_vars.data() + subcell_coords_vars.size(),
            1.0 + dg_mesh.number_of_grid_points());

  const auto active_grid_coords =
      active_grid == evolution::dg::subcell::ActiveGrid::Dg
          ? get<dg_coordinates_tag>(dg_coords_vars)
          : get<subcell_coordinates_tag>(subcell_coords_vars);
  const auto active_inertial_coords = (*grid_to_inertial_map)(
      active_grid_coords, observation_time, functions_of_time);

  Variables<typename System::all_vars_for_test> vars(
      active_mesh.number_of_grid_points());
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);
  const Variables<db::wrap_tags_in<Tags::Analytic, solution_variables>>
      solutions{variables_from_tagged_tuple(analytic_solution.variables(
          active_inertial_coords, observation_time, solution_variables{}))};
  const Variables<solution_variables> errors =
      vars.template extract_subset<solution_variables>() - solutions;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner(
      tuples::TaggedTuple<
          Tags::AnalyticSolution<typename System::solution_for_test>>{
          std::move(analytic_solution)});
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      element_id);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavariables>,
                        ObservationTimeTag, domain::Tags::Mesh<volume_dim>,
                        evolution::dg::subcell::Tags::Mesh<volume_dim>,
                        evolution::dg::subcell::Tags::ActiveGrid,
                        dg_coordinates_tag, subcell_coordinates_tag,
                        Tags::Variables<typename decltype(vars)::tags_list>,
                        ::domain::CoordinateMaps::Tags::CoordinateMap<
                            volume_dim, Frame::Grid, Frame::Inertial>,
                        ::domain::Tags::FunctionsOfTimeInitialize>>(
      metavariables{}, observation_time, dg_mesh, subcell_mesh, active_grid,
      get<dg_coordinates_tag>(dg_coords_vars),
      get<subcell_coordinates_tag>(subcell_coords_vars), vars,
      grid_to_inertial_map->get_clone(), clone_unique_ptrs(functions_of_time));

  // Reset to empty
  MockContributeVolumeData::results = MockContributeVolumeData::Results{};
  observe->run(box,
               ActionTesting::cache<element_component>(runner, array_index),
               array_index, std::add_pointer_t<element_component>{});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeVolumeData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.subfile_name == "/element_data");
  CHECK(results.array_component_id ==
        observers::ArrayComponentId(
            std::add_pointer_t<element_component>{},
            Parallel::ArrayIndex<ElementId<volume_dim>>(array_index)));
  CHECK(results.received_extents.size() == volume_dim);
  CHECK(std::equal(results.received_extents.begin(),
                   results.received_extents.end(),
                   interpolating_mesh.value_or(active_mesh).extents().begin()));
  CHECK(std::equal(results.received_basis.begin(), results.received_basis.end(),
                   interpolating_mesh.value_or(active_mesh).basis().begin()));
  CHECK(std::equal(
      results.received_quadrature.begin(), results.received_quadrature.end(),
      interpolating_mesh.value_or(active_mesh).quadrature().begin()));

  size_t num_components_observed = 0;
  // gcc 6.4.0 gets confused if we try to capture tensor_data by
  // reference and fails to compile because it wants it to be
  // non-const, so we capture a pointer instead.
  const auto check_component =
      [&element_name, &num_components_observed,
       tensor_data = &results.in_received_tensor_data,
       &interpolant](const std::string& component, const DataVector& expected) {
        CAPTURE(*tensor_data);
        CAPTURE(component);
        const DataVector interpolated_expected =
            interpolant.interpolate(expected);
        const auto it = alg::find_if(
            *tensor_data,
            [name = element_name + "/" + component](const TensorComponent& tc) {
              return tc.name == name;
            });
        CHECK(it != tensor_data->end());
        if (component.substr(0, 6) == "Tensor" or
            component.substr(6, 7) == "Tensor2") {
          CHECK(std::get<std::vector<float>>(it->data) ==
                std::vector<float>{interpolated_expected.begin(),
                                   interpolated_expected.end()});
        } else {
          CHECK(std::get<DataVector>(it->data) == interpolated_expected);
        }
        ++num_components_observed;
      };
  for (size_t i = 0; i < volume_dim; ++i) {
    check_component(
        std::string("InertialCoordinates_") + gsl::at({'x', 'y', 'z'}, i),
        active_inertial_coords.get(i));
  }
  System::check_data([&check_component, &vars](const std::string& name,
                                               auto tag,
                                               const auto... indices) {
    check_component(name, get<decltype(tag)>(vars).get(indices...));
  });
  if (AlwaysHasAnalyticSolutions) {
    System::solution_for_test::check_data(
        [&check_component, &errors](const std::string& name, auto tag,
                                    const auto... indices) {
          check_component(name, get<decltype(tag)>(errors).get(indices...));
        });
  }
  CHECK(results.in_received_tensor_data.size() == num_components_observed);

  CHECK(observe->needs_evolved_variables());
}

template <typename System, bool AlwaysHasAnalyticSolutions = true>
void test_system(
    const std::string& mesh_creation_string,
    const std::optional<Mesh<System::volume_dim>>& interpolating_mesh = {}) {
  INFO(pretty_type::get_name<System>());
  CAPTURE(AlwaysHasAnalyticSolutions);
  CAPTURE(mesh_creation_string);
  using metavariables = Metavariables<System, AlwaysHasAnalyticSolutions>;
  for (const auto active_grid : {evolution::dg::subcell::ActiveGrid::Dg,
                                 evolution::dg::subcell::ActiveGrid::Subcell}) {
    test_observe<System, AlwaysHasAnalyticSolutions>(
        std::make_unique<typename System::ObserveEvent>(
            System::make_test_object(interpolating_mesh)),
        interpolating_mesh, active_grid);
  }
  INFO("create/serialize");
  Parallel::register_factory_classes_with_charm<metavariables>();
  const std::string creation_string =
      System::creation_string_for_test + mesh_creation_string;
  const auto factory_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, metavariables>(
          creation_string);
  for (const auto active_grid : {evolution::dg::subcell::ActiveGrid::Subcell,
                                 evolution::dg::subcell::ActiveGrid::Dg}) {
    test_observe<System, AlwaysHasAnalyticSolutions>(
        serialize_and_deserialize(factory_event), interpolating_mesh,
        active_grid);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.ObserveFields", "[Unit][Evolution]") {
  const std::string interpolating_mesh_str = "  InterpolateToMesh: None";
  INVOKE_TEST_FUNCTION(
      test_system, (interpolating_mesh_str, std::nullopt),
      (ScalarSystem<evolution::dg::subcell::Events::ObserveFields>,
       ComplicatedSystem<evolution::dg::subcell::Events::ObserveFields>),
      (true));
  INVOKE_TEST_FUNCTION(
      test_system, (interpolating_mesh_str, std::nullopt),
      (ScalarSystem<evolution::dg::subcell::Events::ObserveFields>,
       ComplicatedSystem<evolution::dg::subcell::Events::ObserveFields>),
      (false));
}
