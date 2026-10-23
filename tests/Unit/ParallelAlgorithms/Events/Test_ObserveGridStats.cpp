// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Events/ObserveGridStats.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ReductionData = Events::ObserveGridStats_detail::ReductionData;
using FormatGridStatsOutput =
    Events::ObserveGridStats_detail::FormatGridStatsOutput;

static_assert(
    tt::assert_conforms_to_v<FormatGridStatsOutput,
                             observers::protocols::ReductionDataFormatter>);

// Whether the element is doing FD. Mimics the subcell `ActiveGrid`.
struct IsFd : db::SimpleTag {
  using type = bool;
};

// Mimics the subcell `ObserverMeshCompute`: elements doing FD are observed on
// a cell-centered FD mesh with 2N-1 points per dimension.
struct TestObserverMeshCompute : Events::Tags::ObserverMesh<3>, db::ComputeTag {
  using base = Events::Tags::ObserverMesh<3>;
  using return_type = Mesh<3>;
  using argument_tags = tmpl::list<domain::Tags::Mesh<3>, IsFd>;
  static void function(const gsl::not_null<Mesh<3>*> observer_mesh,
                       const Mesh<3>& dg_mesh, const bool is_fd) {
    if (not is_fd) {
      *observer_mesh = dg_mesh;
      return;
    }
    std::array<size_t, 3> fd_extents{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(fd_extents, d) = 2 * dg_mesh.extents(d) - 1;
    }
    *observer_mesh = Mesh<3>{fd_extents, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  }
};

using ObserveGridStatsEvent =
    Events::ObserveGridStats<3, tmpl::list<TestObserverMeshCompute>>;

struct MockContributeReductionData {
  struct Results {
    std::unordered_map<std::string, ReductionData> reduction_data;
    std::unordered_map<std::string, std::vector<std::string>> legends;
    std::unordered_map<std::string, bool> observe_per_core;
    std::unordered_map<std::string, std::optional<FormatGridStatsOutput>>
        formatters;
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static Results results;

  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename Formatter>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    Parallel::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& legend,
                    ReductionData&& reduction_data,
                    std::optional<Formatter> formatter,
                    const bool observe_per_core) {
    CHECK(observation_id.observation_key() ==
          observers::ObservationKey(subfile_name + ".dat"));
    CHECK(observation_id.value() == 1.5);
    static_assert(std::is_same_v<Formatter, FormatGridStatsOutput>);
    results.formatters[subfile_name] = std::move(formatter);
    results.legends[subfile_name] = legend;
    results.observe_per_core[subfile_name] = observe_per_core;
    if (results.reduction_data.contains(subfile_name)) {
      results.reduction_data.at(subfile_name)
          .combine(std::move(reduction_data));
    } else {
      results.reduction_data.emplace(subfile_name, std::move(reduction_data));
    }
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<ObserveGridStatsEvent>>>;
  };
};

// Four disjoint unit cubes named Shell0, Shell1, Shell2, and Cube, with the
// three shells in the group Shells.
Domain<3> make_test_domain() {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      maps{};
  for (size_t block_id = 0; block_id < 4; ++block_id) {
    const double lower_x = 2.0 * static_cast<double>(block_id);
    maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine{-1.0, 1.0, lower_x, lower_x + 1.0},
                     Affine{-1.0, 1.0, 0.0, 1.0},
                     Affine{-1.0, 1.0, 0.0, 1.0}}));
  }
  return Domain<3>{std::move(maps),
                   {},
                   {"Shell0", "Shell1", "Shell2", "Cube"},
                   {{"Shells", {"Shell0", "Shell1", "Shell2"}}}};
}

struct TestElement {
  ElementId<3> id;
  Mesh<3> dg_mesh;
  bool is_fd;
};

// The elements to observe. Block Shell2 has no elements.
// - Shell0, DG: extents (4, 5, 6), refinement levels (1, 0, 2), 120 points
// - Shell0, FD: extents (7, 7, 7), refinement levels (1, 0, 2), 343 points
// - Shell1, DG: extents (3, 3, 3), refinement levels (0, 0, 0), 27 points
// - Cube, FD: extents (9, 9, 9), refinement levels (1, 1, 1), 729 points
std::vector<TestElement> make_test_elements() {
  return {
      {ElementId<3>{0, {{SegmentId{1, 0}, SegmentId{0, 0}, SegmentId{2, 3}}}},
       Mesh<3>{{{4, 5, 6}},
               Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       false},
      {ElementId<3>{0, {{SegmentId{1, 1}, SegmentId{0, 0}, SegmentId{2, 1}}}},
       Mesh<3>{4, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       true},
      {ElementId<3>{1},
       Mesh<3>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       false},
      {ElementId<3>{3, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}},
       Mesh<3>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       true}};
}

struct ExpectedStats {
  size_t num_elements;
  size_t num_points;
  double fraction_of_elements_fd;
  double fraction_of_points_fd;
  std::vector<double> avg_points_per_dim;
  std::vector<size_t> min_points_per_dim;
  std::vector<size_t> max_points_per_dim;
  std::vector<double> avg_refinement_level;
  std::vector<size_t> min_refinement_level;
  std::vector<size_t> max_refinement_level;
};

void check_stats(ReductionData reduction_data, const ExpectedStats& expected) {
  reduction_data.finalize();
  const auto& data = reduction_data.data();
  CHECK(std::get<0>(data) == 1.5);
  CHECK(std::get<1>(data) == expected.num_elements);
  CHECK(std::get<2>(data) == expected.num_points);
  CHECK(std::get<3>(data) == approx(expected.fraction_of_elements_fd));
  CHECK(std::get<4>(data) == approx(expected.fraction_of_points_fd));
  CHECK_ITERABLE_APPROX(std::get<5>(data), expected.avg_points_per_dim);
  CHECK(std::get<6>(data) == expected.min_points_per_dim);
  CHECK(std::get<7>(data) == expected.max_points_per_dim);
  CHECK_ITERABLE_APPROX(std::get<8>(data), expected.avg_refinement_level);
  CHECK(std::get<9>(data) == expected.min_refinement_level);
  CHECK(std::get<10>(data) == expected.max_refinement_level);
}

// Creates the event from options, registers and runs it on all test
// elements, and checks that the elements contribute to the expected subfiles.
// Returns the combined reduction data for each subfile.
std::unordered_map<std::string, ReductionData> observe(
    const bool print_to_terminal, const std::string& blocks_to_observe,
    const std::string& reduction_scope,
    const std::vector<std::optional<std::string>>& expected_subfiles) {
  CAPTURE(print_to_terminal);
  CAPTURE(blocks_to_observe);
  CAPTURE(reduction_scope);
  using element_component = ElementComponent<Metavariables>;
  using observer_component = MockObserverComponent<Metavariables>;
  const auto event = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "ObserveGridStats:\n"
          "  PrintToTerminal: " +
          std::string(print_to_terminal ? "true" : "false") +
          "\n"
          "  BlocksToObserve: " +
          blocks_to_observe +
          "\n"
          "  ReductionScope: " +
          reduction_scope + "\n"));
  CHECK_FALSE(event->needs_evolved_variables());

  const auto elements = make_test_elements();
  REQUIRE(expected_subfiles.size() == elements.size());
  MockContributeReductionData::results = {};
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<observer_component>(&runner);
  for (size_t i = 0; i < elements.size(); ++i) {
    const auto& [element_id, dg_mesh, is_fd] = elements[i];
    const auto& expected_subfile = expected_subfiles[i];
    CAPTURE(element_id);
    ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                        element_id);
    auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          domain::Tags::Domain<3>, domain::Tags::Element<3>,
                          domain::Tags::Mesh<3>, IsFd>>(
        Metavariables{}, make_test_domain(),
        Element<3>{element_id, {}, domain::topologies::hypercube<3>}, dg_mesh,
        is_fd);

    const auto registration =
        observers::get_registration_observation_type_and_key(*event, box);
    REQUIRE(registration.has_value() == expected_subfile.has_value());
    if (expected_subfile.has_value()) {
      CHECK(registration->first == observers::TypeOfObservation::Reduction);
      CHECK(registration->second ==
            observers::ObservationKey(*expected_subfile + ".dat"));
    }

    auto obs_box = make_observation_box<
        typename ObserveGridStatsEvent::compute_tags_for_observation_box>(
        make_not_null(&box));
    event->run(make_not_null(&obs_box),
               ActionTesting::cache<element_component>(runner, element_id),
               element_id, std::add_pointer_t<element_component>{},
               {.name = "Time", .value = 1.5});
    if (expected_subfile.has_value()) {
      REQUIRE(
          ActionTesting::number_of_queued_simple_actions<observer_component>(
              runner, 0) == 1);
      ActionTesting::invoke_queued_simple_action<observer_component>(
          make_not_null(&runner), 0);
      CHECK(MockContributeReductionData::results.reduction_data.contains(
          *expected_subfile));
    } else {
      CHECK(ActionTesting::number_of_queued_simple_actions<observer_component>(
                runner, 0) == 0);
    }
  }

  const auto& results = MockContributeReductionData::results;
  const std::vector<std::string> expected_legend{"Time",
                                                 "NumElements",
                                                 "NumPoints",
                                                 "FractionOfElementsFd",
                                                 "FractionOfPointsFd",
                                                 "AvgPointsPerDim_0",
                                                 "AvgPointsPerDim_1",
                                                 "AvgPointsPerDim_2",
                                                 "MinPointsPerDim_0",
                                                 "MinPointsPerDim_1",
                                                 "MinPointsPerDim_2",
                                                 "MaxPointsPerDim_0",
                                                 "MaxPointsPerDim_1",
                                                 "MaxPointsPerDim_2",
                                                 "AvgRefinementLevel_0",
                                                 "AvgRefinementLevel_1",
                                                 "AvgRefinementLevel_2",
                                                 "MinRefinementLevel_0",
                                                 "MinRefinementLevel_1",
                                                 "MinRefinementLevel_2",
                                                 "MaxRefinementLevel_0",
                                                 "MaxRefinementLevel_1",
                                                 "MaxRefinementLevel_2"};
  for (const auto& [subfile_name, legend] : results.legends) {
    CAPTURE(subfile_name);
    CHECK(legend == expected_legend);
    // Only whole-domain reductions are also observed per core
    CHECK(results.observe_per_core.at(subfile_name) ==
          (reduction_scope == "WholeDomain"));
    CHECK(results.formatters.at(subfile_name).has_value() == print_to_terminal);
  }
  return results.reduction_data;
}

void test_whole_domain() {
  const std::string whole{"/GridStats"};
  {
    INFO("All blocks");
    const auto results =
        observe(false, "All", "WholeDomain", {whole, whole, whole, whole});
    REQUIRE(results.size() == 1);
    check_stats(results.at(whole),
                {.num_elements = 4,
                 .num_points = 1219,
                 .fraction_of_elements_fd = 0.5,
                 .fraction_of_points_fd = 1072.0 / 1219.0,
                 .avg_points_per_dim = {23.0 / 4.0, 6.0, 25.0 / 4.0},
                 .min_points_per_dim = {3, 3, 3},
                 .max_points_per_dim = {9, 9, 9},
                 .avg_refinement_level = {0.75, 0.25, 1.25},
                 .min_refinement_level = {0, 0, 0},
                 .max_refinement_level = {1, 1, 2}});
  }
  {
    INFO("Restricted to blocks");
    const auto results =
        observe(false, "[Shell1, Cube]", "WholeDomain", {{}, {}, whole, whole});
    REQUIRE(results.size() == 1);
    check_stats(results.at(whole), {.num_elements = 2,
                                    .num_points = 756,
                                    .fraction_of_elements_fd = 0.5,
                                    .fraction_of_points_fd = 729.0 / 756.0,
                                    .avg_points_per_dim = {6.0, 6.0, 6.0},
                                    .min_points_per_dim = {3, 3, 3},
                                    .max_points_per_dim = {9, 9, 9},
                                    .avg_refinement_level = {0.5, 0.5, 0.5},
                                    .min_refinement_level = {0, 0, 0},
                                    .max_refinement_level = {1, 1, 1}});
  }
}

void test_by_block() {
  const ExpectedStats shell0{.num_elements = 2,
                             .num_points = 463,
                             .fraction_of_elements_fd = 0.5,
                             .fraction_of_points_fd = 343.0 / 463.0,
                             .avg_points_per_dim = {5.5, 6.0, 6.5},
                             .min_points_per_dim = {4, 5, 6},
                             .max_points_per_dim = {7, 7, 7},
                             .avg_refinement_level = {1.0, 0.0, 2.0},
                             .min_refinement_level = {1, 0, 2},
                             .max_refinement_level = {1, 0, 2}};
  const ExpectedStats shell1{.num_elements = 1,
                             .num_points = 27,
                             .fraction_of_elements_fd = 0.0,
                             .fraction_of_points_fd = 0.0,
                             .avg_points_per_dim = {3.0, 3.0, 3.0},
                             .min_points_per_dim = {3, 3, 3},
                             .max_points_per_dim = {3, 3, 3},
                             .avg_refinement_level = {0.0, 0.0, 0.0},
                             .min_refinement_level = {0, 0, 0},
                             .max_refinement_level = {0, 0, 0}};
  const ExpectedStats cube{.num_elements = 1,
                           .num_points = 729,
                           .fraction_of_elements_fd = 1.0,
                           .fraction_of_points_fd = 1.0,
                           .avg_points_per_dim = {9.0, 9.0, 9.0},
                           .min_points_per_dim = {9, 9, 9},
                           .max_points_per_dim = {9, 9, 9},
                           .avg_refinement_level = {1.0, 1.0, 1.0},
                           .min_refinement_level = {1, 1, 1},
                           .max_refinement_level = {1, 1, 1}};
  {
    INFO("ByBlock, all blocks");
    const auto results = observe(false, "All", "ByBlock",
                                 {"/GridStats/Shell0", "/GridStats/Shell0",
                                  "/GridStats/Shell1", "/GridStats/Cube"});
    REQUIRE(results.size() == 3);
    check_stats(results.at("/GridStats/Shell0"), shell0);
    check_stats(results.at("/GridStats/Shell1"), shell1);
    check_stats(results.at("/GridStats/Cube"), cube);
  }
  {
    INFO("ByBlock, restricted to a group");
    const auto results = observe(
        false, "[Shells]", "ByBlock",
        {"/GridStats/Shell0", "/GridStats/Shell0", "/GridStats/Shell1", {}});
    REQUIRE(results.size() == 2);
    check_stats(results.at("/GridStats/Shell0"), shell0);
    check_stats(results.at("/GridStats/Shell1"), shell1);
  }
  {
    INFO("ByBlockOrGroup");
    const auto results = observe(false, "[Shells, Cube]", "ByBlockOrGroup",
                                 {"/GridStats/Shells", "/GridStats/Shells",
                                  "/GridStats/Shells", "/GridStats/Cube"});
    REQUIRE(results.size() == 2);
    check_stats(results.at("/GridStats/Shells"),
                {.num_elements = 3,
                 .num_points = 490,
                 .fraction_of_elements_fd = 1.0 / 3.0,
                 .fraction_of_points_fd = 343.0 / 490.0,
                 .avg_points_per_dim = {14.0 / 3.0, 5.0, 16.0 / 3.0},
                 .min_points_per_dim = {3, 3, 3},
                 .max_points_per_dim = {7, 7, 7},
                 .avg_refinement_level = {2.0 / 3.0, 0.0, 4.0 / 3.0},
                 .min_refinement_level = {0, 0, 0},
                 .max_refinement_level = {1, 0, 2}});
    check_stats(results.at("/GridStats/Cube"), cube);
  }
}

// Formats the reduction data of the subfile from the last `observe` call with
// the formatter that the event sent along, as the observer does
std::string formatted_output(const std::string& subfile_name) {
  const auto& results = MockContributeReductionData::results;
  REQUIRE(results.formatters.at(subfile_name).has_value());
  auto reduction_data = results.reduction_data.at(subfile_name);
  reduction_data.finalize();
  return std::apply(*results.formatters.at(subfile_name),
                    reduction_data.data());
}

void test_print_to_terminal() {
  {
    INFO("WholeDomain");
    observe(true, "All", "WholeDomain",
            {"/GridStats", "/GridStats", "/GridStats", "/GridStats"});
    CHECK(formatted_output("/GridStats") ==
          "Grid stats in /GridStats.dat at Time 1.5:\n"
          "  Elements: 4 (fraction FD 0.5)\n"
          "  Grid points: 1219 (fraction FD 0.879409)\n"
          "  Points per element (avg / min / max):\n"
          "    Dim 0: 5.75 / 3 / 9\n"
          "    Dim 1: 6 / 3 / 9\n"
          "    Dim 2: 6.25 / 3 / 9\n"
          "  Refinement level (avg / min / max):\n"
          "    Dim 0: 0.75 / 0 / 1\n"
          "    Dim 1: 0.25 / 0 / 1\n"
          "    Dim 2: 1.25 / 0 / 2\n");
  }
  {
    INFO("ByBlockOrGroup");
    observe(true, "[Shells, Cube]", "ByBlockOrGroup",
            {"/GridStats/Shells", "/GridStats/Shells", "/GridStats/Shells",
             "/GridStats/Cube"});
    // Each block or group is printed separately and labeled by its subfile
    CHECK(formatted_output("/GridStats/Shells") ==
          "Grid stats in /GridStats/Shells.dat at Time 1.5:\n"
          "  Elements: 3 (fraction FD 0.333333)\n"
          "  Grid points: 490 (fraction FD 0.7)\n"
          "  Points per element (avg / min / max):\n"
          "    Dim 0: 4.66667 / 3 / 7\n"
          "    Dim 1: 5 / 3 / 7\n"
          "    Dim 2: 5.33333 / 3 / 7\n"
          "  Refinement level (avg / min / max):\n"
          "    Dim 0: 0.666667 / 0 / 1\n"
          "    Dim 1: 0 / 0 / 0\n"
          "    Dim 2: 1.33333 / 0 / 2\n");
    CHECK(formatted_output("/GridStats/Cube") ==
          "Grid stats in /GridStats/Cube.dat at Time 1.5:\n"
          "  Elements: 1 (fraction FD 1)\n"
          "  Grid points: 729 (fraction FD 1)\n"
          "  Points per element (avg / min / max):\n"
          "    Dim 0: 9 / 9 / 9\n"
          "    Dim 1: 9 / 9 / 9\n"
          "    Dim 2: 9 / 9 / 9\n"
          "  Refinement level (avg / min / max):\n"
          "    Dim 0: 1 / 1 / 1\n"
          "    Dim 1: 1 / 1 / 1\n"
          "    Dim 2: 1 / 1 / 1\n");
  }
  {
    INFO("Serialize formatter");
    // The formatter is sent to the observer writer, so it must keep its labels
    const auto formatter = serialize_and_deserialize(
        FormatGridStatsOutput{"Iteration", "/GridStats/Shell0"});
    CHECK(formatter.observation_value_name == "Iteration");
    CHECK(formatter.subfile_path == "/GridStats/Shell0");
  }
}

void test_legend() {
  CHECK(Events::ObserveGridStats_detail::legend("Iteration", 1) ==
        std::vector<std::string>{
            "Iteration", "NumElements", "NumPoints", "FractionOfElementsFd",
            "FractionOfPointsFd", "AvgPointsPerDim_0", "MinPointsPerDim_0",
            "MaxPointsPerDim_0", "AvgRefinementLevel_0", "MinRefinementLevel_0",
            "MaxRefinementLevel_0"});
}

// The event is only run in 3D above, so check the contribution of a single
// element in lower dimensions directly
void test_reduction_data_lower_dims() {
  {
    INFO("1D");
    auto data = Events::ObserveGridStats_detail::reduction_data(
        1.5,
        Mesh<1>{4, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        ElementId<1>{0, {{SegmentId{2, 1}}}});
    check_stats(std::move(data), {.num_elements = 1,
                                  .num_points = 4,
                                  .fraction_of_elements_fd = 0.0,
                                  .fraction_of_points_fd = 0.0,
                                  .avg_points_per_dim = {4.0},
                                  .min_points_per_dim = {4},
                                  .max_points_per_dim = {4},
                                  .avg_refinement_level = {2.0},
                                  .min_refinement_level = {2},
                                  .max_refinement_level = {2}});
  }
  {
    INFO("2D");
    auto data = Events::ObserveGridStats_detail::reduction_data(
        1.5,
        Mesh<2>{{{5, 3}},
                Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered},
        ElementId<2>{0, {{SegmentId{1, 0}, SegmentId{0, 0}}}});
    check_stats(std::move(data), {.num_elements = 1,
                                  .num_points = 15,
                                  .fraction_of_elements_fd = 1.0,
                                  .fraction_of_points_fd = 1.0,
                                  .avg_points_per_dim = {5.0, 3.0},
                                  .min_points_per_dim = {5, 3},
                                  .max_points_per_dim = {5, 3},
                                  .avg_refinement_level = {1.0, 0.0},
                                  .min_refinement_level = {1, 0},
                                  .max_refinement_level = {1, 0}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.ObserveGridStats",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();
  test_legend();
  test_reduction_data_lower_dims();
  test_whole_domain();
  test_by_block();
  test_print_to_terminal();
}
