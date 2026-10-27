// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "Parallel/Phase.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
std::vector<double> output_news_data;
std::string data_set_name;
struct MockWriteReductionDataRow {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& /*file_legend*/,
                    const std::tuple<std::vector<double>>& data_row) {
    data_set_name = subfile_name;
    output_news_data = std::get<0>(data_row);
  }
};

struct TestCallWriteNews {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<std::is_same_v<
                Tags::AnalyticBoundaryDataManager, DbTags>...>> = nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    db::get<Tags::AnalyticBoundaryDataManager>(box)
        .template write_news<ParallelComponent>(
            cache, db::get<::Tags::TimeStepId>(box).substep_time());
  }
};

template <typename Metavariables>
struct mock_observer_writer {
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionDataRow>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionDataRow>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct mock_boundary {
  using simple_tags =
      tmpl::list<Tags::AnalyticBoundaryDataManager, ::Tags::TimeStepId>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<Tags::ObservationLMax>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, db::AddComputeTags<>>>>>;
};

struct metavariables {
  using component_list = tmpl::list<mock_observer_writer<metavariables>,
                                    mock_boundary<metavariables>>;
  using observed_reduction_data_tags = tmpl::list<>;
};

// Checks the `Du<Dr<BondiJ>>` boundary value against a fourth-order centered
// finite difference of `Dr<BondiJ>` with a much smaller step, and that it is
// left untouched when no interpolator is supplied.
void test_du_dr_j(const gsl::not_null<std::mt19937*> gen,
                  const double frequency, const double stencil_step) {
  CAPTURE(frequency);
  CAPTURE(stencil_step);
  using du_dr_j_tag = Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>;
  using dr_j_tag = Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>;
  using boundary_variables_type = Variables<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>;
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(*gen);
  UniformCustomDistribution<double> parameter_dist{0.1, 1.0};
  const double extraction_radius = 100.0 * parameter_dist(*gen);
  // Avoid a turning point of the 4 pi oscillation, where the derivative
  // vanishes and a step of 0.5 would give the right answer.
  const double time = 2.1;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const Solutions::LinearizedBondiSachs analytic_solution{
      {0.01 * parameter_dist(*gen), 0.01 * parameter_dist(*gen)},
      extraction_radius,
      frequency};
  const auto dr_j_at = [&analytic_solution, &extraction_radius, &l_max,
                        &number_of_angular_points](const double local_time) {
    const auto gh_variables = analytic_solution.variables(
        l_max, local_time,
        tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                   gh::Tags::Pi<DataVector, 3>,
                   gh::Tags::Phi<DataVector, 3>>{});
    boundary_variables_type boundary_variables{number_of_angular_points};
    create_bondi_boundary_data(
        make_not_null(&boundary_variables),
        get<gh::Tags::Phi<DataVector, 3>>(gh_variables),
        get<gh::Tags::Pi<DataVector, 3>>(gh_variables),
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(gh_variables),
        extraction_radius, l_max);
    return ComplexDataVector{get(get<dr_j_tag>(boundary_variables)).data()};
  };
  const double step = 1.0e-3;
  const ComplexDataVector expected_du_dr_j =
      (8.0 * (dr_j_at(time + step) - dr_j_at(time - step)) -
       (dr_j_at(time + 2.0 * step) - dr_j_at(time - 2.0 * step))) /
      (12.0 * step);

  AnalyticBoundaryDataManager analytic_manager{
      l_max, extraction_radius, analytic_solution.get_clone(), stencil_step};
  CHECK(analytic_manager.get_du_dr_j_stencil_step() == stencil_step);
  const auto populate = [&number_of_angular_points,
                         &time](const AnalyticBoundaryDataManager& manager) {
    boundary_variables_type boundary_variables{number_of_angular_points};
    get(get<du_dr_j_tag>(boundary_variables)).data() =
        std::complex<double>(-3.0, 7.0);
    manager.populate_du_dr_j(make_not_null(&boundary_variables), time);
    return ComplexDataVector{get(get<du_dr_j_tag>(boundary_variables)).data()};
  };

  // Without an interpolator the boundary value is not touched.
  CHECK(populate(analytic_manager) ==
        ComplexDataVector{number_of_angular_points,
                          std::complex<double>(-3.0, 7.0)});

  analytic_manager.set_du_dr_j_interpolator(
      std::make_unique<intrp::BarycentricRationalSpanInterpolator>(10_st,
                                                                   10_st));
  const auto custom_approx =
      Approx::custom().epsilon(1.0e-4).scale(max(abs(expected_du_dr_j)));
  CHECK_ITERABLE_CUSTOM_APPROX(populate(analytic_manager), expected_du_dr_j,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      populate(serialize_and_deserialize(analytic_manager)), expected_du_dr_j,
      custom_approx);

  if (frequency > 1.0) {
    // A step of 0.5 samples the same phase of a 4 pi oscillation at every
    // stencil point, so it misses the nonzero derivative.
    CHECK(max(abs(expected_du_dr_j)) > 0.0);
    AnalyticBoundaryDataManager aliased_manager{
        l_max, extraction_radius, analytic_solution.get_clone(), 0.5};
    aliased_manager.set_du_dr_j_interpolator(
        std::make_unique<intrp::BarycentricRationalSpanInterpolator>(10_st,
                                                                     10_st));
    CHECK(max(abs(populate(aliased_manager) - expected_du_dr_j)) >
          0.5 * max(abs(expected_du_dr_j)));
  }

  analytic_manager.set_du_dr_j_interpolator(nullptr);
  CHECK(populate(analytic_manager) ==
        ComplexDataVector{number_of_angular_points,
                          std::complex<double>(-3.0, 7.0)});
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.AnalyticBoundaryDataManager",
                  "[Unit][Cce]") {
  register_classes_with_charm<Cce::Solutions::LinearizedBondiSachs,
                              intrp::BarycentricRationalSpanInterpolator>();
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
  AnalyticBoundaryDataManager analytic_manager{
      l_max, extraction_radius, analytic_solution.get_clone(), 0.05};
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
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>{});
  create_bondi_boundary_data(
      make_not_null(&expected_boundary_variables),
      get<gh::Tags::Phi<DataVector, 3>>(analytic_solution_gh_variables),
      get<gh::Tags::Pi<DataVector, 3>>(analytic_solution_gh_variables),
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(
          analytic_solution_gh_variables),
      extraction_radius, l_max);
  // `Du<Dr<BondiJ>>` is set by neither `create_bondi_boundary_data` nor
  // `populate_hypersurface_boundary_data` (see `test_du_dr_j`). Both sides
  // remain default-initialized at that tag, so comparing them would be
  // NaN-vs-NaN with `SPECTRE_NAN_INIT`.
  using tags_to_compare = tmpl::list_difference<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>,
      tmpl::list<Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>>>;
  tmpl::for_each<tags_to_compare>([&boundary_variables_from_manager,
                                   &expected_boundary_variables](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    CHECK_ITERABLE_APPROX(get<tag>(boundary_variables_from_manager),
                          get<tag>(expected_boundary_variables));
  });

  // test writing news
  ActionTesting::MockRuntimeSystem<metavariables> runner{{l_max}};
  runner.set_phase(Parallel::Phase::Initialization);
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
  CHECK(data_set_name == "/Cce/News_expected");

  test_du_dr_j(make_not_null(&gen), 0.1, 0.5);
  test_du_dr_j(make_not_null(&gen), 4.0 * std::numbers::pi, 0.005);
}
}  // namespace
}  // namespace Cce
