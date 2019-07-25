// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelBackend/Actions/TerminatePhase.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {
struct TemporalId {};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Var1"; }
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "Var2"; }
};

template <size_t Dim>
struct Var3 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "Var3"; }
};

struct Var4 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Var4"; }
};

template <size_t Dim>
struct Var4Compute : Var4, db::ComputeTag {
  static Scalar<DataVector> function(
      const Scalar<DataVector>& var_1, const tnsr::I<DataVector, Dim>& var_2,
      const tnsr::i<DataVector, Dim>& var_3) noexcept {
    return Scalar<DataVector>{square(get(var_1)) +
                              get(dot_product(var_2, var_3))};
  }

  using argument_tags = tmpl::list<Var1, Var2<Dim>, Var3<Dim>>;
};

template <size_t Dim>
struct InitializeVars {
  using initialization_option_tags =
      tmpl::list<Initialization::Tags::InitialTime>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTagsList, Initialization::Tags::InitialTime>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> distribution(0.0, 1.0);
    const auto nn_generator = make_not_null(&generator);
    const auto nn_distribution = make_not_null(&distribution);

    const DataVector used_for_size(5);
    auto var_1 = make_with_random_values<Scalar<DataVector>>(
        nn_generator, nn_distribution, used_for_size);
    auto var_2 = make_with_random_values<tnsr::I<DataVector, Dim>>(
        nn_generator, nn_distribution, used_for_size);
    auto var_3 = make_with_random_values<tnsr::i<DataVector, Dim>>(
        nn_generator, nn_distribution, used_for_size);

    using simple_tags =
        db::AddSimpleTags<tmpl::list<Var1, Var2<Dim>, Var3<Dim>>>;
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeVars, simple_tags>(
            std::move(box), std::move(var_1), std::move(var_2),
            std::move(var_3)));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTagsList, Initialization::Tags::InitialTime>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;

  static constexpr size_t dim = metavariables::dim;

  using initialization_actions =
      tmpl::list<InitializeVars<dim>,
                 Initialization::Actions::AddComputeTags<Var4Compute<dim>>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using add_options_to_databox = Parallel::ForwardAllOptionsToDataBox<
      Initialization::option_tags<initialization_actions>>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        initialization_actions>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using temporal_id = TemporalId;

  static constexpr size_t dim = Dim;

  enum class Phase { Initialization, Exit };
};

template <size_t Dim>
void test_add_compute_tag() noexcept {
  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<Dim>>;
  using component = Component<Metavariables<Dim>>;

  MockRuntimeSystem runner{{}};
  const double initial_time = 12.20;
  ActionTesting::emplace_component<component>(&runner, 0, initial_time);
  runner.set_phase(Metavariables<Dim>::Phase::Initialization);
  // Initialize simple tags
  runner.template next_action<component>(0);
  // Initialize compute tags
  runner.template next_action<component>(0);

  const Scalar<DataVector> expected_var_4(
      square(get(ActionTesting::get_databox_tag<component, Var1>(runner, 0))) +
      get(dot_product(
          ActionTesting::get_databox_tag<component, Var2<Dim>>(runner, 0),
          ActionTesting::get_databox_tag<component, Var3<Dim>>(runner, 0))));
  CHECK(ActionTesting::get_databox_tag<component, Var4>(runner, 0) ==
        expected_var_4);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Initialization.AddComputeTags",
                  "[Unit][ParallelAlgorithms]") {
  test_add_compute_tag<1>();
  test_add_compute_tag<2>();
  test_add_compute_tag<3>();
}
