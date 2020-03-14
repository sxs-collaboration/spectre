// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetLineSegment.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace StrahlkorperTags {
template <typename Frame>
struct Strahlkorper;
}  // namespace StrahlkorperTags
/// \endcond

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
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get(result) = square(get(x));
    return result;
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
};
struct Negate : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct NegateCompute : Negate, db::ComputeTag {
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get(result) = -get(x);
    return result;
  }
  using argument_tags = tmpl::list<Square>;
  using base = Negate;
};
}  // namespace Tags

// Functions for testing whether we have
// interpolated correctly.  These encode the
// number of points and the coordinates (chosen by hand to
// agree with the input options for InterpolationTargets below), and
// the function that should be called (chosen by hand to agree
// with the ComputeItems listed in the InterpolationTargetTags below).
template <typename Tag>
struct TestFunctionHelper;
template <>
struct TestFunctionHelper<Tags::Square> {
  static constexpr size_t npts = 15;
  static double apply(const double a) noexcept { return square(a); }
  static double coords(size_t i) noexcept { return 1.0 + 0.1 * i; }
};
template <>
struct TestFunctionHelper<Tags::Negate> {
  static constexpr size_t npts = 17;
  static double apply(const double a) noexcept { return -square(a); }
  static double coords(size_t i) noexcept { return 1.1 + 0.0875 * i; }
};

size_t num_test_function_calls = 0;
template <typename InterpolationTargetTag, typename DbTagToRetrieve>
struct TestFunction {
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const typename Metavariables::temporal_id::
                        type& /*temporal_id*/) noexcept {
    const auto& interpolation_result = get<DbTagToRetrieve>(box);
    const auto
        expected_interpolation_result = [&interpolation_result]() noexcept {
      auto result =
          make_with_value<Scalar<DataVector>>(interpolation_result, 0.0);
      const size_t n_pts = TestFunctionHelper<DbTagToRetrieve>::npts;
      for (size_t n = 0; n < n_pts; ++n) {
        std::array<double, 3> coords{};
        for (size_t d = 0; d < 3; ++d) {
          gsl::at(coords, d) = TestFunctionHelper<DbTagToRetrieve>::coords(n);
        }
        get(result)[n] = TestFunctionHelper<DbTagToRetrieve>::apply(
            2.0 * coords[0] + 3.0 * coords[1] + 5.0 * coords[2]);
      }
      return result;
    }
    ();
    CHECK_ITERABLE_APPROX(interpolation_result, expected_interpolation_result);
    ++num_test_function_calls;
  }
};

struct TestKerrHorizonIntegral {
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const typename Metavariables::temporal_id::
                        type& /*temporal_id*/) noexcept {
    const auto& interpolation_result = get<Tags::Square>(box);
    const auto& strahlkorper =
        get<StrahlkorperTags::Strahlkorper<Frame::Inertial>>(box);
    const double integral = strahlkorper.ylm_spherepack().definite_integral(
        make_not_null(get(interpolation_result).data()));
    const double expected_integral = 608.0 * M_PI / 3.0;  // by hand
    // The interpolation is not perfect because I use too few grid points.
    Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);
    CHECK(integral == custom_approx(expected_integral));
    ++num_test_function_calls;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::flatten<tmpl::append<
      Parallel::get_const_global_cache_tags_from_actions<
          tmpl::list<typename InterpolationTargetTag::compute_target_points>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Registration, tmpl::list<>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolator>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Registration, tmpl::list<>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using compute_items_on_source = tmpl::list<Tags::SquareCompute>;
    using vars_to_interpolate_to_target = tmpl::list<Tags::Square>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::Actions::LineSegment<InterpolationTargetA, 3>;
    using post_interpolation_callback =
        TestFunction<InterpolationTargetA, Tags::Square>;
  };
  struct InterpolationTargetB {
    using compute_items_on_source = tmpl::list<Tags::SquareCompute>;
    using vars_to_interpolate_to_target = tmpl::list<Tags::Square>;
    using compute_items_on_target = tmpl::list<Tags::NegateCompute>;
    using compute_target_points =
        intrp::Actions::LineSegment<InterpolationTargetB, 3>;
    using post_interpolation_callback =
        TestFunction<InterpolationTargetB, Tags::Negate>;
  };
  struct InterpolationTargetC {
    using compute_items_on_source = tmpl::list<>;
    using vars_to_interpolate_to_target = tmpl::list<Tags::TestSolution>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points =
        intrp::Actions::KerrHorizon<InterpolationTargetC, ::Frame::Inertial>;
    using post_interpolation_callback = TestKerrHorizonIntegral;
  };

  using interpolator_source_vars = tmpl::list<Tags::TestSolution>;
  using interpolation_target_tags =
      tmpl::list<InterpolationTargetA, InterpolationTargetB,
                 InterpolationTargetC>;
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolation_target<MockMetavariables, InterpolationTargetB>,
      mock_interpolation_target<MockMetavariables, InterpolationTargetC>,
      mock_interpolator<MockMetavariables>>;
  enum class Phase { Initialization, Registration, Testing, Exit };
};

// This tests whether all the Actions of Interpolator and InterpolationTarget
// work together as they are supposed to.
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.Integration",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using interp_component = mock_interpolator<metavars>;
  using target_a_component =
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>;
  using target_b_component =
      mock_interpolation_target<metavars, metavars::InterpolationTargetB>;
  using target_c_component =
      mock_interpolation_target<metavars, metavars::InterpolationTargetC>;

  // Options for all InterpolationTargets.
  intrp::OptionHolders::LineSegment<3> line_segment_opts_A(
      {{1.0, 1.0, 1.0}}, {{2.4, 2.4, 2.4}}, 15);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_B(
      {{1.1, 1.1, 1.1}}, {{2.5, 2.5, 2.5}}, 17);
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_C(10, {{0.0, 0.0, 0.0}},
                                                        1.0, {{0.0, 0.0, 0.0}});
  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);
  tuples::TaggedTuple<
      intrp::Tags::LineSegment<metavars::InterpolationTargetA, 3>,
      domain::Tags::Domain<3>,
      intrp::Tags::LineSegment<metavars::InterpolationTargetB, 3>,
      intrp::Tags::KerrHorizon<metavars::InterpolationTargetC>>
      tuple_of_opts(std::move(line_segment_opts_A),
                    domain_creator.create_domain(),
                    std::move(line_segment_opts_B), kerr_horizon_opts_C);

  ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Initialization);
  ActionTesting::emplace_component<interp_component>(&runner, 0);
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<target_a_component>(&runner, 0);
  ActionTesting::next_action<target_a_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<target_b_component>(&runner, 0);
  ActionTesting::next_action<target_b_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<target_c_component>(&runner, 0);
  ActionTesting::next_action<target_c_component>(make_not_null(&runner), 0);
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Registration);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, 0));
  const auto domain = domain_creator.create_domain();

  // Create Element_ids.
  std::vector<ElementId<3>> element_ids{};
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator.initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }

  // Tell the interpolator how many elements there are by registering
  // each one.
  for (size_t i = 0; i < element_ids.size(); ++i) {
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::RegisterElement>(
        make_not_null(&runner), 0);
  }

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  ActionTesting::simple_action<
      target_a_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});
  ActionTesting::simple_action<
      target_b_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              metavars::InterpolationTargetB>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});
  ActionTesting::simple_action<
      target_c_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              metavars::InterpolationTargetC>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    if (block.is_time_dependent()) {
      ERROR("The block must be time-independent");
    }
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.stationary_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));
    db::item_type<
        ::Tags::Variables<typename metavars::interpolator_source_vars>>
        output_vars(mesh.number_of_grid_points());
    auto& test_solution = get<Tags::TestSolution>(output_vars);

    // Fill test_solution with some analytic solution.
    get(test_solution) = 2.0 * get<0>(inertial_coords) +
                         3.0 * get<1>(inertial_coords) +
                         5.0 * get<2>(inertial_coords);

    // Call the InterpolatorReceiveVolumeData action on each element_id.
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData>(
        make_not_null(&runner), 0, temporal_id, element_id, mesh,
        std::move(output_vars));
  }

  // Invoke remaining actions in random order.
  MAKE_GENERATOR(generator);

  auto index_map = ActionTesting::indices_of_components_with_queued_actions<
      metavars::component_list>(make_not_null(&runner), 0_st);
  while (not index_map.empty()) {
    ActionTesting::invoke_random_queued_action<metavars::component_list>(
        make_not_null(&runner), make_not_null(&generator), index_map, 0_st);
    index_map = ActionTesting::indices_of_components_with_queued_actions<
        metavars::component_list>(make_not_null(&runner), 0_st);
  }

  // Check whether test function was called.
  CHECK(num_test_function_calls == 3);

  // Tell one InterpolationTarget that we want to interpolate at the same
  // temporal_id that we already interpolated at.
  // This call should be ignored by the InterpolationTarget...
  ActionTesting::simple_action<
      target_a_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});
  // ...so make sure it was ignored by checking that there isn't anything
  // else in the simple_action queue of the target or the interpolator.
  CHECK(ActionTesting::is_simple_action_queue_empty<target_a_component>(runner,
                                                                        0));
  CHECK(
      ActionTesting::is_simple_action_queue_empty<interp_component>(runner, 0));
}
}  // namespace
