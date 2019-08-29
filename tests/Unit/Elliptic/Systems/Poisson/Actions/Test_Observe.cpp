// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"                               // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/Actions/Observe.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/Tags.hpp"             // IWYU pragma: keep
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"  // IWYU pragma: keep
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace PUP {
class er;
}  // namespace PUP
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare Tensor

namespace {
template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<2>;
  using simple_tags = db::AddSimpleTags<LinearSolver::Tags::IterationId,
                                        Tags::Mesh<2>, Poisson::Tags::Field,
                                        Tags::Coordinates<2, Frame::Inertial>>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterWithObservers<
                         Poisson::Actions::Observe>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Poisson::Actions::Observe>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked = observers::Observer<Metavariables>;
  using simple_tags =
      typename observers::Actions::Initialize<Metavariables>::simple_tags;
  using compute_tags =
      typename observers::Actions::Initialize<Metavariables>::compute_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<observers::Actions::Initialize<Metavariables>>>>;
};

template <typename Metavariables>
struct MockObserverWriterComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<observers::Tags::ReductionFileName,
                                             observers::Tags::VolumeFileName>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>>;
};

struct AnalyticSolution {
  tuples::TaggedTuple<Poisson::Tags::Field> variables(
      const tnsr::I<DataVector, 2>& x,
      tmpl::list<Poisson::Tags::Field> /*meta*/) const noexcept {
    return {Scalar<DataVector>(2. * get<0>(x) + get<1>(x))};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct AnalyticSolutionTag {
  using type = AnalyticSolution;
};

struct Metavariables {
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockObserverComponent<Metavariables>,
                                    MockObserverWriterComponent<Metavariables>>;
  using analytic_solution_tag = AnalyticSolutionTag;
  using const_global_cache_tags = tmpl::list<analytic_solution_tag>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  /// [collect_reduction_data_tags]
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<Poisson::Actions::Observe>>;
  /// [collect_reduction_data_tags]

  enum class Phase { Initialization, RegisterWithObserver, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Poisson.Actions.Observe",
                  "[Unit][Elliptic][Actions]") {
  using obs_component = MockObserverComponent<Metavariables>;
  using obs_writer = MockObserverWriterComponent<Metavariables>;
  using element_comp = MockElementArray<Metavariables>;

  tuples::TaggedTuple<AnalyticSolutionTag, observers::Tags::ReductionFileName,
                      observers::Tags::VolumeFileName>
      cache_data{};
  const auto& reduction_file_name =
      tuples::get<observers::Tags::ReductionFileName>(cache_data) =
          "./Unit.Elliptic.Systems.Poisson.Actions_ReductionData";
  const auto& volume_file_name =
      tuples::get<observers::Tags::VolumeFileName>(cache_data) =
          "./Unit.Elliptic.Systems.Poisson.Actions_VolumeData";
  ActionTesting::MockRuntimeSystem<Metavariables> runner{cache_data};

  runner.set_phase(Metavariables::Phase::Initialization);
  ActionTesting::emplace_component<obs_component>(&runner, 0);
  ActionTesting::emplace_component<obs_writer>(&runner, 0);
  ActionTesting::next_action<obs_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);

  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 2}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    ActionTesting::emplace_component_and_initialize<element_comp>(
        &runner, ElementIndex<2>{id},
        {db::item_type<LinearSolver::Tags::IterationId>{1},
         Mesh<2>{{{3, 2}},
                 Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto},
         Scalar<DataVector>{{{{1., 2., 3., 4., 5., 6.}}}},
         tnsr::I<DataVector, 2, Frame::Inertial>{
             {{{0., 1., 2., 3., 4., 5.}, {-1., -2., -3., -4., -5., -6.}}}}});
  }
  runner.set_phase(Metavariables::Phase::RegisterWithObserver);

  // Register elements
  for (const auto& id : element_ids) {
    // Observe action
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    // TerminatePhase action
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    CHECK(ActionTesting::get_terminate<element_comp>(runner, id));
    // Invoke the simple_action RegisterSenderWithSelf that was called on the
    // observer component by the RegisterWithObservers action.
    runner.invoke_queued_simple_action<obs_component>(0);
    // Invoke the simple_action RegisterReductionContributorWithObserverWriter
    // and RegisterVolumeContributorWithObserverWriter.
    runner.invoke_queued_simple_action<obs_writer>(0);
    runner.invoke_queued_simple_action<obs_writer>(0);
  }

  runner.set_phase(Metavariables::Phase::Testing);
  const std::string reduction_h5_file_name = reduction_file_name + ".h5";
  if (file_system::check_if_file_exists(reduction_h5_file_name)) {
    file_system::rm(reduction_h5_file_name, true);
  }
  const std::string volume_h5_file_name = volume_file_name + "0.h5";
  if (file_system::check_if_file_exists(volume_h5_file_name)) {
    file_system::rm(volume_h5_file_name, true);
  }

  // Contribute the reduction data
  for (const auto& id : element_ids) {
    runner.next_action<element_comp>(id);
    // Invoke both the reduction and volume observation actions
    runner.invoke_queued_simple_action<obs_component>(0);
    runner.invoke_queued_simple_action<obs_component>(0);
  }

  // Invoke the simple action 'ContributeVolumeDataToWriter' to move the volume
  // data to the Writer parallel component.
  runner.invoke_queued_simple_action<obs_writer>(0);
  // Invoke the threaded actions to write both reduction and volume data to
  // disk
  runner.invoke_queued_threaded_action<obs_writer>(0);
  runner.invoke_queued_threaded_action<obs_writer>(0);

  // Check that the H5 file was written correctly.
  {
    const auto reduction_file =
        h5::H5File<h5::AccessType::ReadOnly>(reduction_h5_file_name);
    const auto& reduction_data_group =
        reduction_file.get<h5::Dat>("/element_data");
    const Matrix reduction_data = reduction_data_group.get_data();
    const auto& reduction_legend = reduction_data_group.get_legend();

    const std::vector<std::string> expected_reduction_legend{
        "Iteration", "NumberOfPoints", "L2Error"};
    CHECK(reduction_legend == expected_reduction_legend);

    // Iteration
    CHECK(reduction_data(0, 0) == 1);
    // NumberOfPoints
    CHECK(reduction_data(0, 1) == 18);
    // L2Error
    CHECK(reduction_data(0, 2) == approx(2.));

    const auto volume_file =
        h5::H5File<h5::AccessType::ReadOnly>(volume_h5_file_name);
    const auto& volume_data_group =
        volume_file.get<h5::VolumeData>("/element_data");

    const auto volume_observation_ids =
        volume_data_group.list_observation_ids();
    // The observation id is probably a hash of the temporal id, but we neither
    // know nor care. But there should be exactly one observation and its value
    // should be the linear solver iteration number.
    CHECK(volume_observation_ids.size() == 1);
    const auto observation_id = volume_observation_ids[0];
    CHECK(volume_data_group.get_observation_value(observation_id) == 1.);

    const auto grid_names = volume_data_group.get_grid_names(observation_id);
    // The grids are probably the element ids, but we also don't care about
    // that. But there should be one grid per element.
    CHECK(grid_names.size() == 3);

    CHECK(volume_data_group.get_extents(observation_id) ==
          std::vector<std::vector<size_t>>{{3, 2}, {3, 2}, {3, 2}});
    const std::vector<std::string> expected_components{
        "Field", "FieldAnalytic", "FieldError", "InertialCoordinates_x",
        "InertialCoordinates_y"};
    const auto components =
        volume_data_group.list_tensor_components(observation_id);
    CHECK(alg::all_of(components,
                      [&expected_components](const std::string& comp) {
                        return alg::found(expected_components, comp);
                      }));
    // Because there were three grids with the same data, we
    // just copy the data 3 times when comparing to the written
    // contiguous data
    auto triple_data = [](const DataVector& data) {
      DataVector triple_dv = DataVector(data.size() * 3);
      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < data.size(); j++) {
          triple_dv[i * data.size() + j] = data[j];
        }
      }
      return triple_dv;
    };
    CHECK(volume_data_group.get_tensor_component(observation_id, "Field") ==
          triple_data(DataVector{1., 2., 3., 4., 5., 6.}));
    CHECK(volume_data_group.get_tensor_component(observation_id,
                                                 "FieldAnalytic") ==
          triple_data(DataVector{-1., 0., 1., 2., 3., 4.}));
    CHECK(volume_data_group.get_tensor_component(
              observation_id, "FieldError") == triple_data(DataVector(6, 2.)));
    CHECK(volume_data_group.get_tensor_component(observation_id,
                                                 "InertialCoordinates_x") ==
          triple_data(DataVector{0., 1., 2., 3., 4., 5.}));
    CHECK(volume_data_group.get_tensor_component(observation_id,
                                                 "InertialCoordinates_y") ==
          triple_data(DataVector{-1., -2., -3., -4., -5., -6.}));
  }

  if (file_system::check_if_file_exists(reduction_h5_file_name)) {
    file_system::rm(reduction_h5_file_name, true);
  }
  if (file_system::check_if_file_exists(volume_h5_file_name)) {
    file_system::rm(volume_h5_file_name, true);
  }
}
