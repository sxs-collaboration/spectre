// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace {
std::vector<double> output_news_data;
std::string data_set_name;
struct MockWriteSimpleData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::vector<std::string>& /*file_legend*/,
                    const std::vector<double>& data_row,
                    const std::string& subfile_name) noexcept {
    data_set_name = subfile_name;
    output_news_data = data_row;
  }
};

struct TestCallWriteNews {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<std::is_same_v<
                Tags::AnalyticBoundaryDataManager, DbTags>...>> = nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    db::get<Tags::AnalyticBoundaryDataManager>(box).write_news(
        cache, db::get<::Tags::TimeStepId>(box).substep_time().value());
  }
};

template <typename Metavariables>
struct mock_observer_writer {
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteSimpleData>;
  using with_these_threaded_actions = tmpl::list<MockWriteSimpleData>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct mock_boundary {
  using simple_tags =
      tmpl::list<Tags::AnalyticBoundaryDataManager, ::Tags::TimeStepId>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<Tags::ObservationLMax>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags,
                                                  db::AddComputeTags<>>>>>;
};

struct metavariables {
  using component_list = tmpl::list<mock_observer_writer<metavariables>,
                                    mock_boundary<metavariables>>;
  using observed_reduction_data_tags = tmpl::list<>;
  enum class Phase { Initialization, Exit };
};

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.AnalyticBoundaryDataManager",
                  "[Unit][Cce]") {
  // set up the analytic data parameters
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<double> parameter_dist{0.1, 1.0};
  const double extraction_radius = 100.0 * parameter_dist(gen);
  const double frequency = parameter_dist(gen);

  const double time = 2.0;

  Solutions::LinearizedBondiSachs analytic_solution{
      {0.01 * parameter_dist(gen), 0.01 * parameter_dist(gen)},
      extraction_radius,
      frequency};
  AnalyticBoundaryDataManager analytic_manager{l_max, extraction_radius,
                                               analytic_solution.get_clone()};
  // test the boundary computation
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      boundary_variables_from_manager{number_of_angular_points};
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      expected_boundary_variables{number_of_angular_points};

  analytic_manager.populate_hypersurface_boundary_data(
      make_not_null(&boundary_variables_from_manager), time);
  const auto analytic_solution_gh_variables = analytic_solution.variables(
      l_max, time,
      tmpl::list<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
                 GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>{});
  create_bondi_boundary_data(
      make_not_null(&expected_boundary_variables),
      get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
          analytic_solution_gh_variables),
      get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(
          analytic_solution_gh_variables),
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          analytic_solution_gh_variables),
      extraction_radius, l_max);
  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&boundary_variables_from_manager,
       &expected_boundary_variables](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        CHECK_ITERABLE_APPROX(get<tag>(boundary_variables_from_manager),
                              get<tag>(expected_boundary_variables));
      });

  Parallel::register_derived_classes_with_charm<
      Cce::Solutions::WorldtubeData>();
  // test writing news
  ActionTesting::MockRuntimeSystem<metavariables> runner{{l_max}};
  runner.set_phase(metavariables::Phase::Initialization);
  ActionTesting::emplace_component<mock_observer_writer<metavariables>>(&runner,
                                                                        0_st);
  ActionTesting::emplace_component_and_initialize<mock_boundary<metavariables>>(
      &runner, 0_st,
      tuples::TaggedTuple<Cce::Tags::AnalyticBoundaryDataManager,
                          ::Tags::TimeStepId>{
          std::move(analytic_manager),
          TimeStepId{true, 0_st, {{2.0, 3.0}, {0_st, 1_st}}}});
  ActionTesting::simple_action<mock_boundary<metavariables>, TestCallWriteNews>(
      make_not_null(&runner), 0_st);
  ActionTesting::invoke_queued_threaded_action<
      mock_observer_writer<metavariables>>(make_not_null(&runner), 0_st);

  const auto expected_news = get<Tags::News>(
      analytic_solution.variables(l_max, time, tmpl::list<Tags::News>{}));
  SpinWeighted<ComplexModalVector, -2> output_news_goldberg_modes{
      square(l_max + 1)};
  for (size_t i = 0; i < square(l_max + 1); ++i) {
    output_news_goldberg_modes.data()[i] = std::complex<double>(
        output_news_data[2 * i + 1], output_news_data[2 * i + 2]);
  }
  SpinWeighted<ComplexModalVector, -2> output_news_libsharp_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  Spectral::Swsh::goldberg_to_libsharp_modes(
      make_not_null(&output_news_libsharp_modes), output_news_goldberg_modes,
      l_max);
  const auto output_news = Spectral::Swsh::inverse_swsh_transform(
      l_max, 1, output_news_libsharp_modes);
  CHECK_ITERABLE_APPROX(output_news.data(), get(expected_news).data());
  CHECK(approx(output_news_data[0]) == time);
  CHECK(data_set_name == "/News_expected");
}
}  // namespace
}  // namespace Cce
