// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Shell.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetLineSegment.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace {

// Simple DataBoxItems for test.
namespace Tags {
struct TestSolution : db::SimpleTag {
  static std::string name() noexcept { return "TestSolution"; }
  using type = Scalar<DataVector>;
};
struct Square : db::SimpleTag {
  static std::string name() noexcept { return "Square"; }
  using type = Scalar<DataVector>;
};
struct SquareComputeItem : Square, db::ComputeTag {
  static std::string name() noexcept { return "Square"; }
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get(result) = square(get(x));
    return result;
  }
  using argument_tags = tmpl::list<TestSolution>;
};
struct Negate : db::SimpleTag {
  static std::string name() noexcept { return "Negate"; }
  using type = Scalar<DataVector>;
};
struct NegateComputeItem : Negate, db::ComputeTag {
  static std::string name() noexcept { return "Negate"; }
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get(result) = -get(x);
    return result;
  }
  using argument_tags = tmpl::list<Square>;
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
  static double apply(const double& a) noexcept { return square(a); }
  static double coords(size_t i) noexcept { return 1.0 + 0.1 * i; }
};
template <>
struct TestFunctionHelper<Tags::Negate> {
  static constexpr size_t npts = 17;
  static double apply(const double& a) noexcept { return -square(a); }
  static double coords(size_t i) noexcept { return 1.1 + 0.0875 * i; }
};

size_t num_test_function_calls = 0;
template <typename InterpolationTargetTag, typename DbTagToRetrieve>
struct TestFunction {
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const typename Metavariables::temporal_id& /*temporal_id*/) noexcept {
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

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using action_list = tmpl::list<>;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag, 3>;
  using initial_databox = db::compute_databox_type<
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables, 3>>;

  using const_global_cache_tag_list = Parallel::get_const_global_cache_tags<
      tmpl::list<intrp::Actions::LineSegment<InterpolationTargetTag, 3>>>;
};

template <typename Metavariables, size_t VolumeDim>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using component_being_mocked = intrp::Interpolator<Metavariables, VolumeDim>;
  using initial_databox =
      db::compute_databox_type<typename intrp::Actions::InitializeInterpolator<
          VolumeDim>::template return_tag_list<Metavariables>>;
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using compute_items_on_source = tmpl::list<Tags::SquareComputeItem>;
    using vars_to_interpolate_to_target = tmpl::list<Tags::Square>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::Actions::LineSegment<InterpolationTargetA, 3>;
    using post_interpolation_callback =
        TestFunction<InterpolationTargetA, Tags::Square>;
    // This tag is also an OptionsTag. The type and help string below
    // refer to the options that are read from the input file.
    using type = typename compute_target_points::options_type;
    static constexpr OptionString help = {"Options for InterpolationTargetA"};
  };
  struct InterpolationTargetB {
    using compute_items_on_source = tmpl::list<Tags::SquareComputeItem>;
    using vars_to_interpolate_to_target = tmpl::list<Tags::Square>;
    using compute_items_on_target = tmpl::list<Tags::NegateComputeItem>;
    using compute_target_points =
        intrp::Actions::LineSegment<InterpolationTargetB, 3>;
    using post_interpolation_callback =
        TestFunction<InterpolationTargetB, Tags::Negate>;
    // This tag is also an OptionsTag. The type and help string below
    // refer to the options that are read from the input file.
    using type = typename compute_target_points::options_type;
    static constexpr OptionString help = {"Options for InterpolationTargetB"};
  };

  using interpolator_source_vars = tmpl::list<Tags::TestSolution>;
  using interpolation_target_tags =
      tmpl::list<InterpolationTargetA, InterpolationTargetB>;
  using temporal_id = Time;
  using domain_frame = Frame::Inertial;
  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolation_target<MockMetavariables, InterpolationTargetB>,
      mock_interpolator<MockMetavariables, 3>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

// This tests whether all the Actions of Interpolator and InterpolationTarget
// work together as they are supposed to.
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.Integration",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTargetA =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars, metavars::InterpolationTargetA>>;
  using MockDistributedObjectsTagTargetB =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars, metavars::InterpolationTargetB>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars, 3>>;
  tuples::get<MockDistributedObjectsTagTargetA>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<mock_interpolation_target<
                   metavars, metavars::InterpolationTargetA>>{});
  tuples::get<MockDistributedObjectsTagTargetB>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<mock_interpolation_target<
                   metavars, metavars::InterpolationTargetB>>{});
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      mock_interpolator<metavars, 3>>{});

  // Options for LineSegment for all InterpolationTargets.
  intrp::OptionHolders::LineSegment<3> line_segment_opts_A(
      {{1.0, 1.0, 1.0}}, {{2.4, 2.4, 2.4}}, 15);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_B(
      {{1.1, 1.1, 1.1}}, {{2.5, 2.5, 2.5}}, 17);
  tuples::TaggedTuple<metavars::InterpolationTargetA,
                      metavars::InterpolationTargetB>
      tuple_of_opts(line_segment_opts_A, line_segment_opts_B);

  MockRuntimeSystem runner{tuple_of_opts, std::move(dist_objects)};

  const auto domain_creator =
      DomainCreators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::InitializeInterpolationTarget<
          metavars::InterpolationTargetA>>(0, domain_creator.create_domain());
  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetB>,
      ::intrp::Actions::InitializeInterpolationTarget<
          metavars::InterpolationTargetB>>(0, domain_creator.create_domain());
  runner.simple_action<mock_interpolator<metavars, 3>,
                       ::intrp::Actions::InitializeInterpolator<3>>(0);

  Slab slab(0.0, 1.0);
  Time temporal_id(slab, 0);
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
    runner.simple_action<mock_interpolator<metavars, 3>,
                         intrp::Actions::RegisterElement>(0);
  }

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<
          metavars::InterpolationTargetA>>(0, std::vector<Time>{temporal_id});
  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetB>,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<
          metavars::InterpolationTargetB>>(0, std::vector<Time>{temporal_id});

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.coordinate_map().get_clone()};
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
    runner.simple_action<mock_interpolator<metavars, 3>,
                         intrp::Actions::InterpolatorReceiveVolumeData>(
        0, temporal_id, element_id, mesh, std::move(output_vars));
  }

  // Now there should be queued actions. Run them.
  auto remaining_simple_actions = [&runner]() noexcept {
    const std::vector<bool> queue_not_empty{
        {not runner
                 .is_simple_action_queue_empty<mock_interpolator<metavars, 3>>(
                     0),
         not runner.is_simple_action_queue_empty<mock_interpolation_target<
             metavars, metavars::InterpolationTargetA>>(0),
         not runner.is_simple_action_queue_empty<mock_interpolation_target<
             metavars, metavars::InterpolationTargetB>>(0)}};
    size_t count = 0;
    for (const auto& not_empty : queue_not_empty) {
      if (not_empty) {
        ++count;
      }
    }
    return std::pair<size_t, std::vector<bool>>(count, queue_not_empty);
  };

  // Invoke remaining actions in random order.
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  CAPTURE(seed);
  std::uniform_int_distribution<size_t> ran(0, 2);
  auto simple_actions_remain = remaining_simple_actions();
  while (simple_actions_remain.first > 0) {
    const size_t index = ran(generator);
    if (not simple_actions_remain.second[index]) {
      continue;
    }
    switch (index) {
      case 0:
        runner.invoke_queued_simple_action<mock_interpolator<metavars, 3>>(0);
        break;
      case 1:
        runner.invoke_queued_simple_action<mock_interpolation_target<
            metavars, metavars::InterpolationTargetA>>(0);
        break;
      case 2:
        runner.invoke_queued_simple_action<mock_interpolation_target<
            metavars, metavars::InterpolationTargetB>>(0);
        break;
      default:
        ERROR("How can we get another index here?");
    }
    simple_actions_remain = remaining_simple_actions();
  }

  // Check whether test function was called.
  CHECK(num_test_function_calls == 2);
}
}  // namespace
