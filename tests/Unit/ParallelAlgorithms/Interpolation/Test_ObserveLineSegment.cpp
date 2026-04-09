// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveLineSegment.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// Simple DataBoxItems for test.
namespace Tags {
struct TestSolution : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Square : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct SquareCompute : Square, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
}  // namespace Tags

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using const_global_cache_tags =
      tmpl::list<observers::Tags::ReductionFileName>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;

  struct LineA : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<DataVector, volume_dim>,
                   domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points =
        intrp::TargetPoints::LineSegment<LineA, volume_dim, Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveLineSegment<
            tmpl::append<vars_to_interpolate_to_target,
                         tmpl::list<Tags::Square>>,
            LineA>>;
  };

  struct LineB : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<DataVector, volume_dim>,
                   domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points =
        intrp::TargetPoints::LineSegment<LineB, volume_dim, Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveLineSegment<
            tmpl::append<vars_to_interpolate_to_target,
                         tmpl::list<Tags::Square>>,
            LineB>>;
  };

  using observed_reduction_data_tags = tmpl::list<>;

  using interpolator_source_vars =
      tmpl::list<Tags::TestSolution,
                 gr::Tags::SpatialMetric<DataVector, volume_dim>,
                 domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<LineA, LineB>;
  using component_list = tmpl::list<MockObserverWriter<MockMetavariables>>;
};

// test function which will be interpolated
template <size_t Dim>
DataVector test_function(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) {
  DataVector res = sin(coords.get(0));
  if constexpr (Dim > 1) {
    res += cos(coords.get(1));
  }
  if constexpr (Dim > 2) {
    res += 3.5 * coords.get(2);
  }
  return res;
}

template <size_t Dim, typename Spacetime>
void run_test(const intrp::OptionHolders::LineSegment<Dim>& line_segment_opts_A,
              const intrp::OptionHolders::LineSegment<Dim>& line_segment_opts_B,
              const Spacetime& spacetime, const bool expect_nans = false) {
  // Check if either file generated by this test exists and remove them
  // if so. Check for both files existing before the test runs, since
  // both files get written when evaluating the list of post interpolation
  // callbacks below.
  const std::string h5_file_prefix = "Test_ObserveLineSegment";
  const auto h5_file_name = h5_file_prefix + ".h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  using metavars = MockMetavariables<Dim>;

  // Test That ObserveTimeSeriesOnSurface indeed does conform to its protocol
  using callback_A =
      tmpl::front<typename metavars::LineA::post_interpolation_callbacks>;
  using callback_B =
      tmpl::front<typename metavars::LineB::post_interpolation_callbacks>;
  using obs_writer = MockObserverWriter<metavars>;
  using protocol = intrp::protocols::PostInterpolationCallback;
  static_assert(tt::assert_conforms_to_v<callback_A, protocol>);
  static_assert(tt::assert_conforms_to_v<callback_B, protocol>);

  tuples::TaggedTuple<observers::Tags::ReductionFileName> tuple_of_opts{
      h5_file_prefix};

  ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);

  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0_st);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, 0));

  const auto set_box = [&]<typename BoxType, typename TargetTag>(
                           const gsl::not_null<BoxType*> box,
                           const intrp::OptionHolders::LineSegment<Dim>&
                               incoming_options,
                           TargetTag /*target_tag_v*/
                       ) {
    db::mutate<intrp::Tags::LineSegment<TargetTag, Dim>>(
        [&](const gsl::not_null<intrp::OptionHolders::LineSegment<Dim>*>
                options) { (*options) = incoming_options; },
        box);

    auto inertial_coords = intrp::TargetPoints::LineSegment<
        TargetTag, Dim, Frame::Inertial>::points(*box, tmpl::type_<metavars>{});

    db::mutate<domain::Tags::Coordinates<Dim, Frame::Inertial>,
               Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, Dim>>(
        [&](const gsl::not_null<tnsr::I<DataVector, Dim>*> box_inertial_coords,
            const gsl::not_null<Scalar<DataVector>*> test_solution,
            const gsl::not_null<tnsr::ii<DataVector, Dim>*> spatial_metric) {
          (*box_inertial_coords) = std::move(inertial_coords);
          get(*test_solution) = test_function(*box_inertial_coords);
          (*spatial_metric) =
              get<gr::Tags::SpatialMetric<DataVector, Dim>>(spacetime.variables(
                  *box_inertial_coords, 0.0,
                  tmpl::list<gr::Tags::SpatialMetric<DataVector, Dim>>{}));

          if (expect_nans) {
            const ScopedFpeState scoped_fpe{};
            get(*test_solution) = std::numeric_limits<double>::quiet_NaN();
            for (size_t i = 0; i < Dim; i++) {
              box_inertial_coords->get(i) =
                  std::numeric_limits<double>::quiet_NaN();
              for (size_t j = 0; j < Dim; j++) {
                spatial_metric->get(i, j) =
                    std::numeric_limits<double>::quiet_NaN();
              }
            }
          }
        },
        box);
  };

  using BoxAType = db::compute_databox_type<
      tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 intrp::Tags::LineSegment<typename metavars::LineA, Dim>,
                 Tags::SquareCompute>>;
  using BoxBType = db::compute_databox_type<
      tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 intrp::Tags::LineSegment<typename metavars::LineB, Dim>,
                 Tags::SquareCompute>>;

  BoxAType box_a{};
  BoxBType box_b{};

  set_box(make_not_null(&box_a), line_segment_opts_A,
          typename metavars::LineA{});
  set_box(make_not_null(&box_b), line_segment_opts_B,
          typename metavars::LineB{});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<obs_writer>(runner, 0_st);

  tmpl::front<typename metavars::LineA::post_interpolation_callbacks>::apply(
      box_a, cache, temporal_id);
  tmpl::front<typename metavars::LineB::post_interpolation_callbacks>::apply(
      box_b, cache, temporal_id);

  // There should be 2 more threaded actions, so invoke them and check
  // that there are no more.  They should all be on node zero.
  REQUIRE(ActionTesting::number_of_queued_threaded_actions<obs_writer>(runner,
                                                                       0) == 2);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);

  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));

  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);

  auto check_file_contents =
      [&file, &spacetime](const std::string& group_name,
                          const tnsr::I<DataVector, Dim>& interpolated_coords) {
        file.close_current_object();
        const auto& vol_file = file.get<h5::VolumeData>(group_name);
        const auto& obs_ids = vol_file.list_observation_ids();
        CHECK(obs_ids.size() == 1);
        const auto& obs_value = vol_file.get_observation_value(obs_ids.at(0));
        CHECK(obs_value == 0.);

        // error due to low resolution of domain
        Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);

        for (size_t i = 0; i < interpolated_coords.size(); ++i) {
          const auto& written_component = vol_file.get_tensor_component(
              obs_ids.at(0),
              "InertialCoordinates" + interpolated_coords.component_suffix(i));
          const auto& written_dv = std::get<DataVector>(written_component.data);
          CHECK_ITERABLE_CUSTOM_APPROX(written_dv, interpolated_coords.get(i),
                                       custom_approx);
        }

        const auto interpolated_metric =
            get<gr::Tags::SpatialMetric<DataVector, Dim>>(spacetime.variables(
                interpolated_coords, 0.0,
                tmpl::list<gr::Tags::SpatialMetric<DataVector, Dim>>{}));
        for (size_t i = 0; i < interpolated_metric.size(); ++i) {
          const auto& written_component = vol_file.get_tensor_component(
              obs_ids.at(0),
              "SpatialMetric" + interpolated_metric.component_suffix(i));
          const auto& written_dv = std::get<DataVector>(written_component.data);
          CHECK_ITERABLE_CUSTOM_APPROX(written_dv, interpolated_metric[i],
                                       custom_approx);
        }

        const auto interpolated_test_solution =
            test_function(interpolated_coords);

        const auto& written_test_solution_component =
            vol_file.get_tensor_component(obs_ids.at(0), "TestSolution");
        const auto& written_test_solution_dv =
            std::get<DataVector>(written_test_solution_component.data);

        CHECK_ITERABLE_CUSTOM_APPROX(written_test_solution_dv,
                                     interpolated_test_solution, custom_approx);

        const auto interpolated_square = square(interpolated_test_solution);
        const auto& written_square_component =
            vol_file.get_tensor_component(obs_ids.at(0), "Square");
        const auto& written_square_dv =
            std::get<DataVector>(written_square_component.data);

        CHECK_ITERABLE_CUSTOM_APPROX(written_square_dv, interpolated_square,
                                     custom_approx);
      };

  auto check_file_contents_are_nans =
      [&file](const std::string& group_name,
              const tnsr::I<DataVector, Dim>& interpolated_coords) {
        file.close_current_object();
        const auto& vol_file = file.get<h5::VolumeData>(group_name);
        const auto& obs_ids = vol_file.list_observation_ids();
        CHECK(obs_ids.size() == 1);
        const auto& obs_value = vol_file.get_observation_value(obs_ids.at(0));
        CHECK(obs_value == 0.);

        for (size_t i = 0; i < interpolated_coords.size(); ++i) {
          const auto& written_component = vol_file.get_tensor_component(
              obs_ids.at(0),
              "InertialCoordinates" + interpolated_coords.component_suffix(i));
          const auto& written_dv = std::get<DataVector>(written_component.data);
          for (size_t s = 0; s < written_dv.size(); ++s) {
            CHECK_THAT(written_dv[s], Catch::Matchers::IsNaN());
          }
        }

        const auto& written_test_solution_component =
            vol_file.get_tensor_component(obs_ids.at(0), "TestSolution");
        const auto& written_test_solution_dv =
            std::get<DataVector>(written_test_solution_component.data);
        for (double written_test_solution : written_test_solution_dv) {
          CHECK_THAT(written_test_solution, Catch::Matchers::IsNaN());
        }

        const auto& written_square_component =
            vol_file.get_tensor_component(obs_ids.at(0), "Square");
        const auto& written_square_dv =
            std::get<DataVector>(written_square_component.data);
        for (double written_square : written_square_dv) {
          CHECK_THAT(written_square, Catch::Matchers::IsNaN());
        }
      };

  const auto& interpolated_coords_a =
      db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box_a);
  const auto& interpolated_coords_b =
      db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box_b);

  if (expect_nans) {
    check_file_contents_are_nans("/LineA", interpolated_coords_a);
    check_file_contents_are_nans("/LineB", interpolated_coords_b);
  } else {
    check_file_contents("/LineA", interpolated_coords_a);
    check_file_contents("/LineB", interpolated_coords_b);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ObserveLineSegment",
                  "[Unit]") {
  intrp::OptionHolders::LineSegment<1> line_segment_opts_A_1d({{0.0}}, {{1.0}},
                                                              10);
  intrp::OptionHolders::LineSegment<1> line_segment_opts_B_1d({{2.2}}, {{3.1}},
                                                              10);
  intrp::OptionHolders::LineSegment<2> line_segment_opts_A_2d({{0.0, 1.0}},
                                                              {{0.0, 2.0}}, 10);
  intrp::OptionHolders::LineSegment<2> line_segment_opts_B_2d({{1.0, 2.0}},
                                                              {{2.0, 3.1}}, 10);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_A_3d(
      {{0.0, 0.0, 1.0}}, {{0.0, 0.0, 2.0}}, 10);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_B_3d(
      {{1.3, 1.0, 2.0}}, {{1.7, 2.0, 3.1}}, 10);

  gr::Solutions::Minkowski<1> minkowski_1d{};
  gr::Solutions::Minkowski<2> minkowski_2d{};
  gr::Solutions::Minkowski<3> minkowski_3d{};
  gr::Solutions::KerrSchild kerr_schild{1., {0.3, 0.4, 0.1}, {0., 0., 0.}};

  run_test(line_segment_opts_A_1d, line_segment_opts_B_1d, minkowski_1d);
  run_test(line_segment_opts_A_2d, line_segment_opts_B_2d, minkowski_2d);
  run_test(line_segment_opts_A_3d, line_segment_opts_B_3d, minkowski_3d);
  run_test(line_segment_opts_A_3d, line_segment_opts_B_3d, kerr_schild);

  intrp::OptionHolders::LineSegment<1> line_segment_opts_N_1d({{4.2}}, {{6.1}},
                                                              10);
  run_test(line_segment_opts_A_1d, line_segment_opts_N_1d, minkowski_1d, true);
}
}  // namespace
