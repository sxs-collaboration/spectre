// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"  // for Variables

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare dg::Events::ObserveErrorNorms
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
namespace observers::Actions {
struct ContributeReductionData;
}  // namespace observers::Actions

namespace {

struct ObservationTimeTag : db::SimpleTag {
  using type = double;
};

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

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    observers::ArrayComponentId /*sender_array_id*/,
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

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
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
  using vars_for_test = tmpl::list<ScalarVar>;
  struct solution_for_test {
    template <typename CheckTensor>
    static void check_data(const CheckTensor& check_tensor) noexcept {
      check_tensor("Error(Scalar)", ScalarVar{});
    }

    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 1>& x, const double t,
        const vars_for_test /*meta*/) noexcept {
      return {Scalar<DataVector>{1.0 - t * get<0>(x)}};
    }

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };
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

  static constexpr size_t volume_dim = 2;
  using vars_for_test = tmpl::list<VectorVar, TensorVar2>;
  struct solution_for_test {
    template <typename CheckTensor>
    static void check_data(const CheckTensor& check_tensor) noexcept {
      check_tensor("Error(Vector)", VectorVar{});
      check_tensor("Error(Tensor2)", TensorVar2{});
    }

    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 2>& x, const double t,
        const vars_for_test /*meta*/) noexcept {
      auto vector = make_with_value<tnsr::I<DataVector, 2>>(x, 0.0);
      auto tensor = make_with_value<tnsr::ii<DataVector, 2>>(x, 0.0);
      // Arbitrary functions
      get<0>(vector) = 1.0 - t * get<0>(x);
      get<1>(vector) = 1.0 - t * get<1>(x);
      get<0, 0>(tensor) = get<0>(x) + get<1>(x);
      get<0, 1>(tensor) = get<0>(x) - get<1>(x);
      get<1, 1>(tensor) = get<0>(x) * get<1>(x);
      return {std::move(vector), std::move(tensor)};
    }

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };
};

template <typename System, typename ObserveEvent>
void test_observe(const std::unique_ptr<ObserveEvent> observe) noexcept {
  constexpr size_t volume_dim = System::volume_dim;
  using metavariables = Metavariables<System>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  using coordinates_tag =
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>;

  const typename element_component::array_index array_index(0);
  const size_t num_points = 5;
  const double observation_time = 2.0;
  Variables<tmpl::push_back<typename System::vars_for_test, coordinates_tag>>
      vars(num_points);
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  const typename System::solution_for_test analytic_solution{};
  using solution_variables = typename System::vars_for_test;
  const Variables<db::wrap_tags_in<Tags::Analytic, solution_variables>>
      solutions{variables_from_tagged_tuple(analytic_solution.variables(
          get<coordinates_tag>(vars), observation_time, solution_variables{}))};
  const Variables<solution_variables> errors =
      vars.template extract_subset<solution_variables>() - solutions;

  ActionTesting::MockRuntimeSystem<metavariables> runner(
      tuples::TaggedTuple<
          Tags::AnalyticSolution<typename System::solution_for_test>>{
          std::move(analytic_solution)});
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_component<observer_component>(&runner, 0);

  const auto box = db::create<db::AddSimpleTags<
      ObservationTimeTag, Tags::Variables<typename decltype(vars)::tags_list>,
      db::add_tag_prefix<Tags::Analytic, Tags::Variables<solution_variables>>>>(
      observation_time, vars, solutions);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const observers::ObservationKey expected_observation_key_for_reg(
      "/reduction0.dat");
  CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
  CHECK(ids_to_register->second == expected_observation_key_for_reg);

  observe->run(box, runner.cache(), array_index,
               std::add_pointer_t<element_component>{});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.subfile_name == "/reduction0");
  CHECK(results.reduction_names[0] == "ObservationTimeTag");
  CHECK(results.time == observation_time);
  CHECK(results.reduction_names[1] == "NumberOfPoints");
  CHECK(results.number_of_grid_points == num_points);
  CHECK(results.reduction_names.size() == results.errors.size() + 2);

  size_t num_tensors_observed = 0;
  // Clang 6 believes the capture of results to be
  // incorrect, presumably because it is checking the storage duration
  // of the object referenced by reduction_results, rather than
  // reduction_results itself.  gcc (correctly, I believe) requires
  // the capture.
#if defined(__clang__) && __clang_major__ > 4
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-lambda-capture"
#endif  // __clang__
  System::solution_for_test::check_data([
    &errors, &num_tensors_observed, &results
  ](const std::string& name, auto tag) noexcept {
#if defined(__clang__) && __clang_major__ > 4
#pragma GCC diagnostic pop
#endif  // __clang__
    double expected = 0.0;
    for (const auto& component : get<decltype(tag)>(errors)) {
      // The rest of the RMS calculation is done later by the writer.
      expected += alg::accumulate(square(component), 0.0);
    }

    CAPTURE(results.reduction_names);
    CAPTURE(name);
    const auto it = alg::find(results.reduction_names, name);
    CHECK(it != results.reduction_names.end());
    if (it != results.reduction_names.end()) {
      CHECK(results.errors[static_cast<size_t>(
                               it - results.reduction_names.begin()) -
                           2] == expected);
    }
    ++num_tensors_observed;
  });
  CHECK(results.errors.size() == num_tensors_observed);
}

template <typename System>
void test_system() noexcept {
  INFO(pretty_type::get_name<System>());
  test_observe<System>(
      std::make_unique<dg::Events::ObserveErrorNorms<
          ObservationTimeTag, typename System::vars_for_test>>("reduction0"));

  INFO("create/serialize");
  using EventType = Event<tmpl::list<dg::Events::Registrars::ObserveErrorNorms<
      ObservationTimeTag, typename System::vars_for_test>>>;
  Parallel::register_derived_classes_with_charm<EventType>();
  const auto factory_event = TestHelpers::test_factory_creation<EventType>(
      "ObserveErrorNorms:\n"
      "  SubfileName: reduction0");
  auto serialized_event = serialize_and_deserialize(factory_event);
  test_observe<System>(std::move(serialized_event));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveErrorNorms", "[Unit][Evolution]") {
  test_system<ScalarSystem>();
  test_system<ComplicatedSystem>();
}
