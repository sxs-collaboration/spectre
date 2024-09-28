// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Rectilinear.hpp"
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
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Events/ObserveFields.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// for Variables

template <size_t>
class Index;

namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace observers::Actions {
struct ContributeVolumeData;
}  // namespace observers::Actions

namespace {
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace TestHelpers::dg::Events::ObserveFields;

template <typename System, typename = void>
struct prim_vars_impl {
  using type = tmpl::list<>;
  static constexpr bool has_prims = false;
};

template <typename System>
struct prim_vars_impl<System,
                      std::void_t<typename System::primitive_variables_tag>> {
  using type = typename System::primitive_variables_tag::tags_list;
  static constexpr bool has_prims = true;
};

template <typename System>
using prim_vars_list = typename prim_vars_impl<System>::type;

template <typename System, typename ArraySectionIdTag = void,
          typename ObserveEvent>
void test_observe(
    const std::unique_ptr<ObserveEvent> observe,
    const std::optional<Mesh<System::volume_dim>>& interpolating_mesh,
    const bool has_analytic_solutions, const bool test_specific_blocks,
    const std::optional<std::string>& section = std::nullopt) {
  INFO(test_specific_blocks);
  using metavariables = Metavariables<System, false>;
  constexpr size_t volume_dim = System::volume_dim;
  using DataType = typename System::data_type;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  using coordinates_tag =
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>;

  const ElementId<volume_dim> element_id(0);
  const Element<volume_dim> element(element_id, {});
  // NOTE: The coordinate map is not actually what is used to compute the
  // coordinates.
  const domain::creators::Rectilinear<volume_dim> rectilinear{
      make_array<volume_dim>(-2.0), make_array<volume_dim>(2.0),
      make_array<volume_dim>(0_st), make_array<volume_dim>(5_st),
      make_array<volume_dim>(true)};
  const typename element_component::array_index array_index(element_id);
  const Mesh<volume_dim> mesh(5, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);

  const intrp::RegularGrid interpolant(mesh, interpolating_mesh.value_or(mesh));
  const double observation_time = 2.0;
  Variables<typename System::variables_tag::tags_list> vars(
      mesh.number_of_grid_points());
  Variables<tmpl::list<coordinates_tag>> coordinate_vars(
      mesh.number_of_grid_points());
  Variables<prim_vars_list<System>> prim_vars(mesh.number_of_grid_points());
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);
  std::iota(coordinate_vars.data(),
            coordinate_vars.data() + coordinate_vars.size(),
            static_cast<double>(vars.size()));
  if constexpr (not std::is_same_v<tmpl::list<>, prim_vars_list<System>>) {
    std::iota(prim_vars.data(), prim_vars.data() + prim_vars.size(),
              static_cast<double>(vars.size() + coordinate_vars.size()));
  }

  const typename System::solution_for_test analytic_solution{};
  using solution_variables = typename System::solution_for_test::vars_for_test;
  const Variables<
      db::wrap_tags_in<::Tags::detail::AnalyticImpl, solution_variables>>
      solutions{variables_from_tagged_tuple(
          analytic_solution.variables(get<coordinates_tag>(coordinate_vars),
                                      observation_time, solution_variables{}))};
  const Variables<solution_variables> errors =
      [&vars, &prim_vars, &solutions]() -> Variables<solution_variables> {
    if constexpr (prim_vars_impl<System>::has_prims) {
      (void)vars;
      return prim_vars.template extract_subset<solution_variables>() -
             solutions;
    } else {
      (void)prim_vars;
      return vars.template extract_subset<solution_variables>() - solutions;
    }
  }();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner(
      tuples::TaggedTuple<
          ::Tags::AnalyticSolution<typename System::solution_for_test>>{
          std::move(analytic_solution)});
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      element_id);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavariables>,
      domain::Tags::Domain<volume_dim>, domain::Tags::Element<volume_dim>,
      domain::Tags::Mesh<volume_dim>,
      ::Tags::Variables<typename decltype(vars)::tags_list>,
      ::Tags::Variables<typename decltype(prim_vars)::tags_list>,
      coordinates_tag, ::Tags::AnalyticSolutions<solution_variables>,
      observers::Tags::ObservationKey<ArraySectionIdTag>>>(
      metavariables{}, rectilinear.create_domain(), element, mesh, vars,
      prim_vars, get<coordinates_tag>(coordinate_vars),
      [&solutions, &has_analytic_solutions]() {
        return has_analytic_solutions ? std::make_optional(solutions)
                                      : std::nullopt;
      }(),
      section);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const std::string expected_subfile_name{
      "/element_data" +
      (std::is_same_v<ArraySectionIdTag, void> ? ""
                                               : section.value_or("Unused"))};
  const observers::ObservationKey expected_observation_key_for_reg(
      expected_subfile_name + ".vol");
  if (std::is_same_v<ArraySectionIdTag, void> or section.has_value()) {
    CHECK(ids_to_register->first == observers::TypeOfObservation::Volume);
    CHECK(ids_to_register->second == expected_observation_key_for_reg);
  } else {
    CHECK_FALSE(ids_to_register.has_value());
  }

  auto obs_box = make_observation_box<tmpl::push_back<
      tmpl::filter<
          typename System::ObserveEvent::compute_tags_for_observation_box,
          db::is_compute_tag<tmpl::_1>>,
      ::Events::Tags::ObserverMeshCompute<volume_dim>>>(make_not_null(&box));
  observe->run(make_not_null(&obs_box),
               ActionTesting::cache<element_component>(runner, array_index),
               array_index, std::add_pointer_t<element_component>{},
               {"TimeName", observation_time});

  if (not std::is_same_v<ArraySectionIdTag, void> and not section.has_value()) {
    CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));
    return;
  }

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeVolumeData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.observation_id.observation_key() ==
        expected_observation_key_for_reg);
  CHECK(results.subfile_name == expected_subfile_name);
  CHECK(results.array_component_id ==
        Parallel::make_array_component_id<element_component>(array_index));
  CHECK(results.received_volume_data.extents.size() == volume_dim);
  CHECK(std::equal(results.received_volume_data.extents.begin(),
                   results.received_volume_data.extents.end(),
                   interpolating_mesh.value_or(mesh).extents().begin()));
  CHECK(std::equal(results.received_volume_data.basis.begin(),
                   results.received_volume_data.basis.end(),
                   interpolating_mesh.value_or(mesh).basis().begin()));
  CHECK(std::equal(results.received_volume_data.quadrature.begin(),
                   results.received_volume_data.quadrature.end(),
                   interpolating_mesh.value_or(mesh).quadrature().begin()));

  size_t num_components_observed = 0;
  // gcc 6.4.0 gets confused if we try to capture tensor_data by
  // reference and fails to compile because it wants it to be
  // non-const, so we capture a pointer instead.
  const auto check_component_impl = [&num_components_observed,
                                     tensor_data = &results.received_volume_data
                                                        .tensor_components,
                                     &interpolant](const std::string& component,
                                                   const DataVector& expected) {
    CAPTURE(*tensor_data);
    CAPTURE(component);
    const DataVector interpolated_expected = interpolant.interpolate(expected);
    const auto it =
        alg::find_if(*tensor_data, [&component](const TensorComponent& tc) {
          return tc.name == component;
        });
    REQUIRE(it != tensor_data->end());
    CAPTURE(component);
    if (component.substr(0, 6) == "Tensor" or
        component.substr(0, 7) == "Tensor2" or
        component.substr(0, 14) == "Error(Tensor2)") {
      CHECK(std::get<std::vector<float>>(it->data) ==
            std::vector<float>{interpolated_expected.begin(),
                               interpolated_expected.end()});
    } else {
      CHECK(std::get<DataVector>(it->data) == interpolated_expected);
    }
    ++num_components_observed;
  };
  const auto check_component = [&check_component_impl](
                                   const std::string& component,
                                   const auto& expected) {
    if constexpr (std::is_same_v<std::decay_t<decltype(expected)>,
                                 ComplexDataVector>) {
      check_component_impl("Re(" + component + ")", real(expected));
      check_component_impl("Im(" + component + ")", imag(expected));
    } else {
      check_component_impl(component, expected);
    }
  };
  for (size_t i = 0; i < volume_dim; ++i) {
    check_component(
        std::string("InertialCoordinates_") + gsl::at({'x', 'y', 'z'}, i),
        get<coordinates_tag>(coordinate_vars).get(i));
  }
  System::check_data([&check_component, &prim_vars, &vars](
                         const std::string& name, auto tag_v,
                         const auto... indices) {
    using tag = decltype(tag_v);
    if constexpr (std::is_same_v<tag, TestHelpers::dg::Events::ObserveFields::
                                          Tags::ScalarVarTimesTwo<DataType>>) {
      check_component(
          name, DataType{2.0 * get<typename System::ScalarVar>(vars).get()});
    } else if constexpr (std::is_same_v<
                             tag, TestHelpers::dg::Events::ObserveFields::Tags::
                                      ScalarVarTimesThree<DataType>>) {
      check_component(
          name, DataType{3.0 * get<typename System::ScalarVar>(vars).get()});
    } else {
      if constexpr (tmpl::list_contains_v<
                        typename std::decay_t<decltype(prim_vars)>::tags_list,
                        tag>) {
        check_component(name, get<tag>(prim_vars).get(indices...));
      } else {
        check_component(name, get<tag>(vars).get(indices...));
      }
    }
  });
  if (has_analytic_solutions) {
    System::solution_for_test::check_data(
        [&check_component, &errors](const std::string& name, auto tag,
                                    const auto... indices) {
          check_component(name, get<decltype(tag)>(errors).get(indices...));
        });
  }
  CHECK(results.received_volume_data.tensor_components.size() ==
        num_components_observed);

  CHECK(static_cast<const Event&>(*observe).is_ready(
      box, ActionTesting::cache<element_component>(runner, array_index),
      array_index, std::add_pointer_t<element_component>{}));
  CHECK(observe->needs_evolved_variables());
}

template <typename System>
void test_system(
    const std::string& mesh_creation_string,
    const std::optional<Mesh<System::volume_dim>>& interpolating_mesh = {},
    const bool has_analytic_solutions = true,
    const std::optional<std::string>& section = std::nullopt) {
  INFO(pretty_type::get_name<System>());
  CAPTURE(has_analytic_solutions);
  CAPTURE(mesh_creation_string);
  using ArraySectionIdTag = typename System::array_section_id;
  INFO(pretty_type::get_name<ArraySectionIdTag>());
  CAPTURE(section);
  using metavariables = Metavariables<System, false>;
  for (const bool test_specific_blocks : {false, true}) {
    test_observe<System, ArraySectionIdTag>(
        std::make_unique<typename System::ObserveEvent>(
            System::make_test_object(interpolating_mesh)),
        interpolating_mesh, has_analytic_solutions, test_specific_blocks,
        section);
    INFO("create/serialize");
    register_factory_classes_with_charm<metavariables>();
    {
      const std::string creation_string =
          System::creation_string_for_test + mesh_creation_string;
      const auto factory_event =
          TestHelpers::test_creation<std::unique_ptr<Event>, metavariables>(
              creation_string);
      auto serialized_event = serialize_and_deserialize(factory_event);
      test_observe<System, ArraySectionIdTag>(
          std::move(serialized_event), interpolating_mesh,
          has_analytic_solutions, test_specific_blocks, section);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveFields", "[Unit][Evolution]") {
  {
    INFO("No Interpolation");
    const std::string interpolating_mesh_str = "  InterpolateToMesh: None";
    using system_no_section = ScalarSystem<dg::Events::ObserveFields, void>;
    using system_with_section =
        ScalarSystem<dg::Events::ObserveFields, TestSectionIdTag>;
    using system_complex =
        ScalarSystem<dg::Events::ObserveFields, void, ComplexDataVector>;
    INVOKE_TEST_FUNCTION(
        test_system, (interpolating_mesh_str, std::nullopt, true, std::nullopt),
        (system_no_section, system_with_section, system_complex,
         ComplicatedSystem<dg::Events::ObserveFields>));
    INVOKE_TEST_FUNCTION(
        test_system, (interpolating_mesh_str, std::nullopt, true, "Section0"),
        (system_no_section, system_with_section,
         ComplicatedSystem<dg::Events::ObserveFields>));
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, std::nullopt, false),
                         (ScalarSystem<dg::Events::ObserveFields>,
                          ComplicatedSystem<dg::Events::ObserveFields>));
  }

  {
    INFO("Interpolate to finer grid");
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 12\n"
        "    Basis: Legendre\n"
        "    Quadrature: GaussLobatto";
    const Mesh<1> interpolating_mesh_1d{12, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto};
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, interpolating_mesh_1d, true),
                         (ScalarSystem<dg::Events::ObserveFields>));
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, interpolating_mesh_1d, false),
                         (ScalarSystem<dg::Events::ObserveFields>));
    const Mesh<2> interpolating_mesh_2d{12, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto};
    test_system<ComplicatedSystem<dg::Events::ObserveFields>>(
        interpolating_mesh_str, interpolating_mesh_2d);
  }

  {
    INFO("Interpolate to coarser grid");
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 3\n"
        "    Basis: Legendre\n"
        "    Quadrature: GaussLobatto";
    const Mesh<1> interpolating_mesh_1{3, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto};
    const Mesh<2> interpolating_mesh_2{3, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto};
    test_system<ScalarSystem<dg::Events::ObserveFields>>(interpolating_mesh_str,
                                                         interpolating_mesh_1);
    test_system<ComplicatedSystem<dg::Events::ObserveFields>>(
        interpolating_mesh_str, interpolating_mesh_2);
  }

  {
    INFO("Interpolate to different basis");
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 5\n"
        "    Basis: Chebyshev\n"
        "    Quadrature: GaussLobatto";
    const Mesh<1> interpolating_mesh_1{5, Spectral::Basis::Chebyshev,
                                       Spectral::Quadrature::GaussLobatto};
    const Mesh<2> interpolating_mesh_2{5, Spectral::Basis::Chebyshev,
                                       Spectral::Quadrature::GaussLobatto};
    test_system<ScalarSystem<dg::Events::ObserveFields>>(interpolating_mesh_str,
                                                         interpolating_mesh_1);
    test_system<ComplicatedSystem<dg::Events::ObserveFields>>(
        interpolating_mesh_str, interpolating_mesh_2);
  }

  {
    INFO("Interpolate to different quadrature");
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 5\n"
        "    Basis: Legendre\n"
        "    Quadrature: Gauss";
    const Mesh<1> interpolating_mesh_1{5, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::Gauss};
    const Mesh<2> interpolating_mesh_2{5, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::Gauss};
    test_system<ScalarSystem<dg::Events::ObserveFields>>(interpolating_mesh_str,
                                                         interpolating_mesh_1);
    test_system<ComplicatedSystem<dg::Events::ObserveFields>>(
        interpolating_mesh_str, interpolating_mesh_2);
  }

  {
    INFO("Interpolate to different extents, basis and quadrature");
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 8\n"
        "    Basis: FiniteDifference\n"
        "    Quadrature: CellCentered";
    const Mesh<1> interpolating_mesh_1{8, Spectral::Basis::FiniteDifference,
                                       Spectral::Quadrature::CellCentered};
    const Mesh<2> interpolating_mesh_2{8, Spectral::Basis::FiniteDifference,
                                       Spectral::Quadrature::CellCentered};
    test_system<ScalarSystem<dg::Events::ObserveFields>>(interpolating_mesh_str,
                                                         interpolating_mesh_1);
    test_system<ComplicatedSystem<dg::Events::ObserveFields>>(
        interpolating_mesh_str, interpolating_mesh_2);
  }

  {
    INFO("Interpolate to non-uniform mesh");
    // test nonuniform mesh, these cannot be parsed yet
    const Mesh<2> interpolating_mesh(
        {3, 9}, {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto});

    test_observe<ComplicatedSystem<dg::Events::ObserveFields>>(
        std::make_unique<typename ComplicatedSystem<
            dg::Events::ObserveFields>::ObserveEvent>(
            ComplicatedSystem<dg::Events::ObserveFields>::make_test_object(
                interpolating_mesh)),
        interpolating_mesh, true, false);
  }
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<
          typename ScalarSystem<dg::Events::ObserveFields>::ObserveEvent>(
          "SubfileName: VolumeData\n"
          "CoordinatesFloatingPointType: Double\n"
          "VariablesToObserve: [NotAVar]\n"
          "FloatingPointTypes: [Double]\n"
          "InterpolateToMesh: None\n"
          "BlocksToObserve: All\n"),
      Catch::Matchers::ContainsSubstring("Invalid selection: NotAVar"));

  CHECK_THROWS_WITH(
      TestHelpers::test_creation<
          typename ScalarSystem<dg::Events::ObserveFields>::ObserveEvent>(
          "SubfileName: VolumeData\n"
          "CoordinatesFloatingPointType: Double\n"
          "VariablesToObserve: [Scalar, Scalar]\n"
          "FloatingPointTypes: [Double]\n"
          "InterpolateToMesh: None\n"
          "BlocksToObserve: All\n"),
      Catch::Matchers::ContainsSubstring("Scalar specified multiple times"));
}
