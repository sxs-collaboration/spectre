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
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
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

struct ObservationTimeTag : db::SimpleTag {
  using type = double;
};

struct MockContributeVolumeData {
  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    observers::ArrayComponentId array_component_id{};
    std::vector<TensorComponent> in_received_tensor_data{};
    std::vector<size_t> received_extents{};
    std::vector<Spectral::Basis> received_basis{};
    std::vector<Spectral::Quadrature> received_quadrature{};
  };
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, size_t Dim>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const observers::ArrayComponentId& array_component_id,
                    std::vector<TensorComponent>&& in_received_tensor_data,
                    const Index<Dim>& received_extents,
                    const std::array<Spectral::Basis, Dim>& received_basis,
                    const std::array<Spectral::Quadrature, Dim>&
                        received_quadrature) noexcept {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.array_component_id = array_component_id;
    results.in_received_tensor_data = in_received_tensor_data;
    results.received_extents.assign(received_extents.indices().begin(),
                                    received_extents.indices().end());
    results.received_basis.assign(received_basis.begin(), received_basis.end());
    results.received_quadrature.assign(received_quadrature.begin(),
                                       received_quadrature.end());
  }
};

MockContributeVolumeData::Results MockContributeVolumeData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::system::volume_dim>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeVolumeData>;
  using with_these_simple_actions = tmpl::list<MockContributeVolumeData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename System>
struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<Tags::AnalyticSolution<typename System::solution_for_test>>;
  enum class Phase { Initialization, Testing, Exit };
};

// Test systems

struct ScalarSystem {
  struct ScalarVar : db::SimpleTag {
    static std::string name() noexcept { return "Scalar"; }
    using type = Scalar<DataVector>;
  };

  static constexpr size_t volume_dim = 1;
  using variables_tag = Tags::Variables<tmpl::list<ScalarVar>>;

  template <typename CheckComponent>
  static void check_data(const CheckComponent& check_component) noexcept {
    check_component("Scalar", ScalarVar{});
  }

  using all_vars_for_test = tmpl::list<ScalarVar>;
  struct solution_for_test {
    using vars_for_test = variables_tag::tags_list;

    template <typename CheckComponent>
    static void check_data(const CheckComponent& check_component) noexcept {
      check_component("Error(Scalar)", ScalarVar{});
    }
    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(

        const tnsr::I<DataVector, 1>& x, const double t,
        const vars_for_test /*meta*/) noexcept {
      return {Scalar<DataVector>{1.0 - t * get<0>(x)}};
    }

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };

  using ObserveEvent =
      dg::Events::ObserveFields<volume_dim, ObservationTimeTag,
                                all_vars_for_test,
                                solution_for_test::vars_for_test>;
  static constexpr auto creation_string_for_test =
      "ObserveFields:\n"
      "  SubfileName: element_data\n"
      "  VariablesToObserve: [Scalar]\n";
  static ObserveEvent make_test_object(
      const std::optional<Mesh<volume_dim>>& interpolating_mesh) noexcept {
    return ObserveEvent{"element_data", {"Scalar"}, interpolating_mesh};
  }
};

struct ComplicatedSystem {
  struct ScalarVar : db::SimpleTag {
    static std::string name() noexcept { return "Scalar"; }
    using type = Scalar<DataVector>;
  };

  struct VectorVar : db::SimpleTag {
    static std::string name() noexcept { return "Vector"; }
    using type = tnsr::I<DataVector, 2>;
  };

  struct TensorVar : db::SimpleTag {
    static std::string name() noexcept { return "Tensor"; }
    using type = tnsr::ii<DataVector, 2>;
  };

  struct TensorVar2 : db::SimpleTag {
    static std::string name() noexcept { return "Tensor2"; }
    using type = tnsr::ii<DataVector, 2>;
  };

  struct UnobservedVar : db::SimpleTag {
    static std::string name() noexcept { return "Unobserved"; }
    using type = Scalar<DataVector>;
  };

  struct UnobservedVar2 : db::SimpleTag {
    static std::string name() noexcept { return "Unobserved2"; }
    using type = Scalar<DataVector>;
  };

  static constexpr size_t volume_dim = 2;
  using variables_tag =
      Tags::Variables<tmpl::list<TensorVar, ScalarVar, UnobservedVar>>;
  using primitive_variables_tag =
      Tags::Variables<tmpl::list<VectorVar, TensorVar2, UnobservedVar2>>;

  template <typename CheckComponent>
  static void check_data(const CheckComponent& check_component) noexcept {
    check_component("Scalar", ScalarVar{});
    check_component("Tensor_xx", TensorVar{}, 0, 0);
    check_component("Tensor_yx", TensorVar{}, 0, 1);
    check_component("Tensor_yy", TensorVar{}, 1, 1);
    check_component("Vector_x", VectorVar{}, 0);
    check_component("Vector_y", VectorVar{}, 1);
    check_component("Tensor2_xx", TensorVar2{}, 0, 0);
    check_component("Tensor2_yx", TensorVar2{}, 0, 1);
    check_component("Tensor2_yy", TensorVar2{}, 1, 1);
  }

  using all_vars_for_test = tmpl::list<TensorVar, ScalarVar, UnobservedVar,
                                       VectorVar, TensorVar2, UnobservedVar2>;
  struct solution_for_test {
    using vars_for_test = primitive_variables_tag::tags_list;

    template <typename CheckComponent>
    static void check_data(const CheckComponent& check_component) noexcept {
      check_component("Error(Vector)_x", VectorVar{}, 0);
      check_component("Error(Vector)_y", VectorVar{}, 1);
      check_component("Error(Tensor2)_xx", TensorVar2{}, 0, 0);
      check_component("Error(Tensor2)_yx", TensorVar2{}, 0, 1);
      check_component("Error(Tensor2)_yy", TensorVar2{}, 1, 1);
    }

    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 2>& x, const double t,
        const vars_for_test /*meta*/) noexcept {
      auto vector = make_with_value<tnsr::I<DataVector, 2>>(x, 0.0);
      auto tensor = make_with_value<tnsr::ii<DataVector, 2>>(x, 0.0);
      auto unobserved = make_with_value<Scalar<DataVector>>(x, 0.0);
      // Arbitrary functions
      get<0>(vector) = 1.0 - t * get<0>(x);
      get<1>(vector) = 1.0 - t * get<1>(x);
      get<0, 0>(tensor) = get<0>(x) + get<1>(x);
      get<0, 1>(tensor) = get<0>(x) - get<1>(x);
      get<1, 1>(tensor) = get<0>(x) * get<1>(x);
      get(unobserved) = 2.0 * get<0>(x);
      return {std::move(vector), std::move(tensor), std::move(unobserved)};
    }

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };

  using ObserveEvent =
      dg::Events::ObserveFields<volume_dim, ObservationTimeTag,
                                all_vars_for_test,
                                solution_for_test::vars_for_test>;
  static constexpr auto creation_string_for_test =
      "ObserveFields:\n"
      "  SubfileName: element_data\n"
      "  VariablesToObserve: [Scalar, Vector, Tensor, Tensor2]\n";

  static ObserveEvent make_test_object(
      const std::optional<Mesh<volume_dim>>& interpolating_mesh) noexcept {
    return ObserveEvent("element_data",
                        {"Scalar", "Vector", "Tensor", "Tensor2"},
                        interpolating_mesh);
  }
};

template <typename System, bool AlwaysHasAnalyticSolutions,
          typename ObserveEvent>
void test_observe(
    const std::unique_ptr<ObserveEvent> observe,
    const std::optional<Mesh<System::volume_dim>>& interpolating_mesh,
    const bool has_analytic_solutions) noexcept {
  using metavariables = Metavariables<System>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  static constexpr auto volume_dim = System::volume_dim;
  using coordinates_tag =
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>;

  const ElementId<volume_dim> element_id(2);
  const typename element_component::array_index array_index(element_id);
  const std::string element_name = get_output(element_id);
  const Mesh<volume_dim> mesh(5, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);

  const intrp::RegularGrid interpolant(mesh, interpolating_mesh.value_or(mesh));
  const double observation_time = 2.0;
  Variables<
      tmpl::push_back<typename System::all_vars_for_test, coordinates_tag>>
      vars(mesh.number_of_grid_points());
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  const typename System::solution_for_test analytic_solution{};
  using solution_variables = typename System::solution_for_test::vars_for_test;
  const Variables<db::wrap_tags_in<Tags::Analytic, solution_variables>>
      solutions{variables_from_tagged_tuple(analytic_solution.variables(
          get<coordinates_tag>(vars), observation_time, solution_variables{}))};
  const Variables<solution_variables> errors =
      vars.template extract_subset<solution_variables>() - solutions;

  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<System>>;
  MockRuntimeSystem runner(
      tuples::TaggedTuple<
          Tags::AnalyticSolution<typename System::solution_for_test>>{
          std::move(analytic_solution)});
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      element_id);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const auto box = db::create<db::AddSimpleTags<
      ObservationTimeTag, domain::Tags::Mesh<volume_dim>,
      Tags::Variables<typename decltype(vars)::tags_list>,
      tmpl::conditional_t<
          AlwaysHasAnalyticSolutions,
          ::Tags::AnalyticSolutions<solution_variables>,
          ::Tags::AnalyticSolutionsOptional<solution_variables>>>>(
      observation_time, mesh, vars, [&solutions, &has_analytic_solutions]() {
        if constexpr (AlwaysHasAnalyticSolutions) {
          (void)has_analytic_solutions;
          // NOLINTNEXTLINE(performance-no-automatic-move)
          return solutions;
        } else {
          return has_analytic_solutions ? std::make_optional(solutions)
                                        : std::nullopt;
        }
      }());

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
                   interpolating_mesh.value_or(mesh).extents().begin()));
  CHECK(std::equal(results.received_basis.begin(), results.received_basis.end(),
                   interpolating_mesh.value_or(mesh).basis().begin()));
  CHECK(std::equal(results.received_quadrature.begin(),
                   results.received_quadrature.end(),
                   interpolating_mesh.value_or(mesh).quadrature().begin()));

  size_t num_components_observed = 0;
  // gcc 6.4.0 gets confused if we try to capture tensor_data by
  // reference and fails to compile because it wants it to be
  // non-const, so we capture a pointer instead.
  const auto check_component =
      [&element_name, &num_components_observed,
       tensor_data = &results.in_received_tensor_data, &interpolant](
          const std::string& component, const DataVector& expected) noexcept {
        CAPTURE(*tensor_data);
        CAPTURE(component);
        const auto it =
            alg::find_if(*tensor_data, [name = element_name + "/" + component](
                                           const TensorComponent& tc) noexcept {
              return tc.name == name;
            });
        CHECK(it != tensor_data->end());
        if (it != tensor_data->end()) {
          CHECK(it->data == interpolant.interpolate(expected));
        }
        ++num_components_observed;
      };
  for (size_t i = 0; i < volume_dim; ++i) {
    check_component(
        std::string("InertialCoordinates_") + gsl::at({'x', 'y', 'z'}, i),
        get<coordinates_tag>(vars).get(i));
  }
  System::check_data([&check_component, &vars](const std::string& name,
                                               auto tag,
                                               const auto... indices) noexcept {
    check_component(name, get<decltype(tag)>(vars).get(indices...));
  });
  if (AlwaysHasAnalyticSolutions or has_analytic_solutions) {
    System::solution_for_test::check_data(
        [&check_component, &errors](const std::string& name, auto tag,
                                    const auto... indices) noexcept {
          check_component(name, get<decltype(tag)>(errors).get(indices...));
        });
  }
  CHECK(results.in_received_tensor_data.size() == num_components_observed);

  CHECK(observe->needs_evolved_variables());
}

template <typename System, bool AlwaysHasAnalyticSolutions = true>
void test_system(
    const std::string& mesh_creation_string,
    const std::optional<Mesh<System::volume_dim>>& interpolating_mesh = {},
    const bool has_analytic_solutions = true) noexcept {
  INFO(pretty_type::get_name<System>());
  CAPTURE(AlwaysHasAnalyticSolutions);
  CAPTURE(has_analytic_solutions);
  CAPTURE(mesh_creation_string);
  test_observe<System, AlwaysHasAnalyticSolutions>(
      std::make_unique<typename System::ObserveEvent>(
          System::make_test_object(interpolating_mesh)),
      interpolating_mesh, has_analytic_solutions);
  INFO("create/serialize");
  using EventType = Event<tmpl::list<dg::Events::Registrars::ObserveFields<
      System::volume_dim, ObservationTimeTag,
      typename System::all_vars_for_test,
      typename System::solution_for_test::vars_for_test>>>;
  Parallel::register_derived_classes_with_charm<EventType>();
  const std::string creation_string =
      System::creation_string_for_test + mesh_creation_string;
  const auto factory_event =
      TestHelpers::test_creation<std::unique_ptr<EventType>>(creation_string);
  auto serialized_event = serialize_and_deserialize(factory_event);
  test_observe<System, AlwaysHasAnalyticSolutions>(
      std::move(serialized_event), interpolating_mesh, has_analytic_solutions);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveFields", "[Unit][Evolution]") {
  {
    INFO("No Interpolation")
    const std::string interpolating_mesh_str = "  InterpolateToMesh: None";
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, std::nullopt, true),
                         (ScalarSystem, ComplicatedSystem), (true));
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, std::nullopt, false),
                         (ScalarSystem, ComplicatedSystem), (true, false));
  }

  {
    INFO("Interpolate to finer grid")
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 12\n"
        "    Basis: Legendre\n"
        "    Quadrature: GaussLobatto";
    const Mesh<1> interpolating_mesh_1d{12, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto};
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, interpolating_mesh_1d, true),
                         (ScalarSystem), (true));
    INVOKE_TEST_FUNCTION(test_system,
                         (interpolating_mesh_str, interpolating_mesh_1d, false),
                         (ScalarSystem), (true, false));
    const Mesh<2> interpolating_mesh_2d{12, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto};
    test_system<ComplicatedSystem>(interpolating_mesh_str,
                                   interpolating_mesh_2d);
  }

  {
    INFO("Interpolate to coarser grid")
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 3\n"
        "    Basis: Legendre\n"
        "    Quadrature: GaussLobatto";
    const Mesh<1> interpolating_mesh_1{3, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto};
    const Mesh<2> interpolating_mesh_2{3, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto};
    test_system<ScalarSystem>(interpolating_mesh_str, interpolating_mesh_1);
    test_system<ComplicatedSystem>(interpolating_mesh_str,
                                   interpolating_mesh_2);
  }

  {
    INFO("Interpolate to different basis")
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 5\n"
        "    Basis: Chebyshev\n"
        "    Quadrature: GaussLobatto";
    const Mesh<1> interpolating_mesh_1{5, Spectral::Basis::Chebyshev,
                                       Spectral::Quadrature::GaussLobatto};
    const Mesh<2> interpolating_mesh_2{5, Spectral::Basis::Chebyshev,
                                       Spectral::Quadrature::GaussLobatto};
    test_system<ScalarSystem>(interpolating_mesh_str, interpolating_mesh_1);
    test_system<ComplicatedSystem>(interpolating_mesh_str,
                                   interpolating_mesh_2);
  }

  {
    INFO("Interpolate to different quadrature")
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 5\n"
        "    Basis: Legendre\n"
        "    Quadrature: Gauss";
    const Mesh<1> interpolating_mesh_1{5, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::Gauss};
    const Mesh<2> interpolating_mesh_2{5, Spectral::Basis::Legendre,
                                       Spectral::Quadrature::Gauss};
    test_system<ScalarSystem>(interpolating_mesh_str, interpolating_mesh_1);
    test_system<ComplicatedSystem>(interpolating_mesh_str,
                                   interpolating_mesh_2);
  }

  {
    INFO("Interpolate to different extents, basis and quadrature")
    const std::string interpolating_mesh_str =
        "  InterpolateToMesh:\n"
        "    Extents: 8\n"
        "    Basis: FiniteDifference\n"
        "    Quadrature: CellCentered";
    const Mesh<1> interpolating_mesh_1{8, Spectral::Basis::FiniteDifference,
                                       Spectral::Quadrature::CellCentered};
    const Mesh<2> interpolating_mesh_2{8, Spectral::Basis::FiniteDifference,
                                       Spectral::Quadrature::CellCentered};
    test_system<ScalarSystem>(interpolating_mesh_str, interpolating_mesh_1);
    test_system<ComplicatedSystem>(interpolating_mesh_str,
                                   interpolating_mesh_2);
  }

  {
    INFO("Interpolate to non-uniform mesh")
    // test nonuniform mesh, these cannot be parsed yet
    const Mesh<2> interpolating_mesh(
        {3, 9}, {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto});

    test_observe<ComplicatedSystem, true>(
        std::make_unique<typename ComplicatedSystem::ObserveEvent>(
            ComplicatedSystem::make_test_object(interpolating_mesh)),
        interpolating_mesh, true);
  }
}

// [[OutputRegex, NotAVar is not an available variable.*Scalar]]
SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveFields.bad_field",
                  "[Unit][Evolution]") {
  ERROR_TEST();
  TestHelpers::test_creation<ScalarSystem::ObserveEvent>(
      "SubfileName: VolumeData\n"
      "VariablesToObserve: [NotAVar]\n"
      "InterpolateToMesh: None");
}

// [[OutputRegex, Scalar specified multiple times]]
SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveFields.repeated_field",
                  "[Unit][Evolution]") {
  ERROR_TEST();
  TestHelpers::test_creation<ScalarSystem::ObserveEvent>(
      "SubfileName: VolumeData\n"
      "VariablesToObserve: [Scalar, Scalar]\n"
      "InterpolateToMesh: None");
}
