// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>  // IWYU pragma: keep  // for std::move
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Observe.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"  // for Variables
// IWYU pragma: no_include "Utilities/Registration.hpp"

template <size_t>
class Index;
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare dg::Events::Observe
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
namespace observers {
namespace Actions {
struct ContributeReductionData;
struct ContributeVolumeData;
}  // namespace Actions
}  // namespace observers

namespace {

struct MockContributeVolumeData {
  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    observers::ArrayComponentId array_component_id{};
    std::vector<TensorComponent> in_received_tensor_data{};
    std::vector<size_t> received_extents{};
  };
  static Results results;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t Dim>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const observers::ArrayComponentId& array_component_id,
                    std::vector<TensorComponent>&& in_received_tensor_data,
                    const Index<Dim>& received_extents) noexcept {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.array_component_id = array_component_id;
    results.in_received_tensor_data = in_received_tensor_data;
    results.received_extents.assign(received_extents.indices().begin(),
                                    received_extents.indices().end());
  }
};

MockContributeVolumeData::Results MockContributeVolumeData::results{};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    double time;
    size_t number_of_grid_points;
    std::vector<double> errors;
  };
  static Results results;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) noexcept {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.reduction_names = reduction_names;
    results.time = std::get<0>(reduction_data.data());
    results.number_of_grid_points = std::get<1>(reduction_data.data());
    results.errors.clear();
    tmpl::for_each<tmpl::range<size_t, 2, sizeof...(Ts)>>([&reduction_data](
        const auto index_v) noexcept {
      constexpr size_t index = tmpl::type_from<decltype(index_v)>::value;
      results.errors.push_back(std::get<index>(reduction_data.data()));
    });
  }
};

MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename System>
struct Metavariables;

template <typename System>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables<System>;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<System::volume_dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using action_list = tmpl::list<>;
};

template <typename System>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables<System>>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeVolumeData,
                 observers::Actions::ContributeReductionData>;
  using with_these_simple_actions =
      tmpl::list<MockContributeVolumeData, MockContributeReductionData>;

  using metavariables = Metavariables<System>;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using action_list = tmpl::list<>;
};

template <typename System>
struct Metavariables {
  using system = System;
  using component_list =
      tmpl::list<ElementComponent<System>, MockObserverComponent<System>>;
  using const_global_cache_tag_list = tmpl::list<
      OptionTags::AnalyticSolution<typename System::solution_for_test>>;

  struct ObservationType {};
  using element_observation_type = ObservationType;
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
  static void check_volume_data(
      const CheckComponent& check_component) noexcept {
    check_component("Scalar", ScalarVar{});
  }

  using all_vars_for_test = tmpl::list<ScalarVar>;
  struct solution_for_test {
    using vars_for_test = variables_tag::tags_list;

    template <typename CheckComponent>
    static void check_volume_data(
        const CheckComponent& check_component) noexcept {
      check_component("ErrorScalar", ScalarVar{});
    }

    template <typename CheckTensor>
    static void check_reduction_data(const CheckTensor& check_tensor) noexcept {
      check_tensor("ErrorScalar", ScalarVar{});
    }

    tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 1>& x, const double t,
        const vars_for_test /*meta*/) const noexcept {
      return {Scalar<DataVector>{1.0 - t * get<0>(x)}};
    }

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };

  using ObserveEvent = dg::Events::Observe<volume_dim, all_vars_for_test,
                                           solution_for_test::vars_for_test>;
  static constexpr auto creation_string_for_test = "  Observe";
  static ObserveEvent make_test_object() noexcept { return ObserveEvent{}; }
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
  static void check_volume_data(
      const CheckComponent& check_component) noexcept {
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
    static void check_volume_data(
        const CheckComponent& check_component) noexcept {
      check_component("ErrorVector_x", VectorVar{}, 0);
      check_component("ErrorVector_y", VectorVar{}, 1);
      check_component("ErrorTensor2_xx", TensorVar2{}, 0, 0);
      check_component("ErrorTensor2_yx", TensorVar2{}, 0, 1);
      check_component("ErrorTensor2_yy", TensorVar2{}, 1, 1);
    }

    template <typename CheckTensor>
    static void check_reduction_data(const CheckTensor& check_tensor) noexcept {
      check_tensor("ErrorVector", VectorVar{});
      check_tensor("ErrorTensor2", TensorVar2{});
      check_tensor("ErrorUnobserved2", UnobservedVar2{});
    }

    tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 2>& x, const double t,
        const vars_for_test /*meta*/) const noexcept {
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

  using ObserveEvent = dg::Events::Observe<volume_dim, all_vars_for_test,
                                           solution_for_test::vars_for_test>;
  static constexpr auto creation_string_for_test =
      "  Observe:\n"
      "    VariablesToObserve: [Scalar, Vector, Tensor, Tensor2]";
  static ObserveEvent make_test_object() noexcept {
    return ObserveEvent({"Scalar", "Vector", "Tensor", "Tensor2"});
  }
};

template <typename System, typename ObserveEvent>
void test_observe(const std::unique_ptr<ObserveEvent> observe) noexcept {
  constexpr size_t volume_dim = System::volume_dim;
  using element_component = ElementComponent<System>;
  using observer_component = MockObserverComponent<System>;
  using coordinates_tag = Tags::Coordinates<volume_dim, Frame::Inertial>;

  const ElementId<volume_dim> element_id(2);
  const typename element_component::array_index array_index(element_id);
  const std::string element_name = get_output(element_id);
  const Mesh<volume_dim> mesh(5, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
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
  const Variables<solution_variables> errors =
      vars.template extract_subset<solution_variables>() -
      variables_from_tagged_tuple(analytic_solution.variables(
          get<coordinates_tag>(vars), observation_time, solution_variables{}));

  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<System>>;

  typename MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<typename MockRuntimeSystem::template MockDistributedObjectsTag<
      element_component>>(dist_objects)
      .emplace(array_index,
               ActionTesting::MockDistributedObject<element_component>{});
  tuples::get<typename MockRuntimeSystem::template MockDistributedObjectsTag<
      observer_component>>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<observer_component>{});
  MockRuntimeSystem runner({std::move(analytic_solution)},
                           std::move(dist_objects));

  const auto box = db::create<
      db::AddSimpleTags<Tags::TimeId, Tags::Mesh<volume_dim>,
                        Tags::Variables<typename decltype(vars)::tags_list>>,
      db::AddComputeTags<Tags::Time>>(
      TimeId(true, 0, Slab(0., observation_time).end()), mesh, vars);

  observe->run(box, runner.cache(), array_index,
               std::add_pointer_t<element_component>{});

  // Process the volume and reduction data
  runner.template invoke_queued_simple_action<observer_component>(0);
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& volume_results = MockContributeVolumeData::results;
  CHECK(volume_results.observation_id.value() == observation_time);
  CHECK(volume_results.subfile_name == "/element_data");
  CHECK(volume_results.array_component_id ==
        observers::ArrayComponentId(
            std::add_pointer_t<element_component>{},
            Parallel::ArrayIndex<ElementIndex<volume_dim>>(array_index)));
  CHECK(volume_results.received_extents.size() == volume_dim);
  CHECK(std::equal(volume_results.received_extents.begin(),
                   volume_results.received_extents.end(),
                   mesh.extents().begin()));

  size_t num_components_observed = 0;
  // gcc 6.4.0 gets confused if we try to capture tensor_data by
  // reference and fails to compile because it wants it to be
  // non-const, so we capture a pointer instead.
  const auto check_component = [
    &element_name, &num_components_observed,
    tensor_data = &volume_results.in_received_tensor_data
  ](const std::string& component, const DataVector& expected) noexcept {
    CAPTURE(*tensor_data);
    CAPTURE(component);
    const auto it =
        alg::find_if(*tensor_data, [name = element_name + "/" + component](
                                       const TensorComponent& tc) noexcept {
          return tc.name == name;
        });
    CHECK(it != tensor_data->end());
    if (it != tensor_data->end()) {
      CHECK(it->data == expected);
    }
    ++num_components_observed;
  };
  for (size_t i = 0; i < volume_dim; ++i) {
    check_component(
        std::string("InertialCoordinates_") + gsl::at({'x', 'y', 'z'}, i),
        get<coordinates_tag>(vars).get(i));
  }
  System::check_volume_data([&check_component, &vars](
      const std::string& name, auto tag, const auto... indices) noexcept {
    check_component(name, get<decltype(tag)>(vars).get(indices...));
  });
  System::solution_for_test::check_volume_data([&check_component, &errors](
      const std::string& name, auto tag, const auto... indices) noexcept {
    check_component(name, get<decltype(tag)>(errors).get(indices...));
  });
  CHECK(volume_results.in_received_tensor_data.size() ==
        num_components_observed);

  const auto& reduction_results = MockContributeReductionData::results;
  CHECK(reduction_results.observation_id.value() == observation_time);
  CHECK(reduction_results.subfile_name == "/element_data");
  CHECK(reduction_results.reduction_names[0] == "Time");
  CHECK(reduction_results.time == observation_time);
  CHECK(reduction_results.reduction_names[1] == "NumberOfPoints");
  CHECK(reduction_results.number_of_grid_points ==
        mesh.number_of_grid_points());
  CHECK(reduction_results.reduction_names.size() ==
        reduction_results.errors.size() + 2);

  size_t num_tensors_observed = 0;
  // Clang 6 believes the capture of reduction_results to be
  // incorrect, presumably because it is checking the storage duration
  // of the object referenced by reduction_results, rather than
  // reduction_results itself.  gcc (correctly, I believe) requires
  // the capture.
#if defined(__clang__) && __clang_major__ > 4
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-lambda-capture"
#endif  // __clang__
  System::solution_for_test::check_reduction_data([
    &errors, &num_tensors_observed, &reduction_results
  ](const std::string& name, auto tag) noexcept {
#if defined(__clang__) && __clang_major__ > 4
#pragma GCC diagnostic pop
#endif  // __clang__
    double expected = 0.0;
    for (const auto& component : get<decltype(tag)>(errors)) {
      // The rest of the RMS calculation is done later by the writer.
      expected += alg::accumulate(square(component), 0.0);
    }

    CAPTURE(reduction_results.reduction_names);
    CAPTURE(name);
    const auto it = alg::find(reduction_results.reduction_names, name);
    CHECK(it != reduction_results.reduction_names.end());
    if (it != reduction_results.reduction_names.end()) {
      CHECK(reduction_results
                .errors[static_cast<size_t>(
                            it - reduction_results.reduction_names.begin()) -
                        2] == expected);
    }
    ++num_tensors_observed;
  });
  CHECK(reduction_results.errors.size() == num_tensors_observed);
}

template <typename System>
void test_system() noexcept {
  INFO(pretty_type::get_name<System>());
  test_observe<System>(std::make_unique<typename System::ObserveEvent>(
      System::make_test_object()));

  INFO("create/serialize");
  using EventType = Event<tmpl::list<dg::Events::Registrars::Observe<
      System::volume_dim, typename System::all_vars_for_test,
      typename System::solution_for_test::vars_for_test>>>;
  Parallel::register_derived_classes_with_charm<EventType>();
  const auto factory_event =
      test_factory_creation<EventType>(System::creation_string_for_test);
  auto serialized_event = serialize_and_deserialize(factory_event);
  test_observe<System>(std::move(serialized_event));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.Observe", "[Unit][Evolution]") {
  test_system<ScalarSystem>();
  test_system<ComplicatedSystem>();
}

// [[OutputRegex, NotAVar is not a variable in the system.*Scalar]]
SPECTRE_TEST_CASE("Unit.Evolution.dG.Observe.bad_field", "[Unit][Evolution]") {
  ERROR_TEST();
  test_creation<ScalarSystem::ObserveEvent>("  VariablesToObserve: [NotAVar]");
}

// [[OutputRegex, Scalar specified multiple times]]
SPECTRE_TEST_CASE("Unit.Evolution.dG.Observe.repeated_field",
                  "[Unit][Evolution]") {
  ERROR_TEST();
  test_creation<ScalarSystem::ObserveEvent>(
      "  VariablesToObserve: [Scalar, Scalar]");
}
