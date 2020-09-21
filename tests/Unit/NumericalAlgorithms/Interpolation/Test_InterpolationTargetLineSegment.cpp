// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetLineSegment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::LineSegment<InterpolationTargetA, 3>;
  };
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 InterpTargetTestHelpers::mock_interpolator<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.LineSegment",
                  "[Unit]") {
  // Options for LineSegment
  intrp::OptionHolders::LineSegment<3> line_segment_opts({{1.0, 1.0, 1.0}},
                                                         {{2.4, 2.4, 2.4}}, 15);

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::LineSegment<3>>(
          "Begin: [1.0, 1.0, 1.0]\n"
          "End: [2.4, 2.4, 2.4]\n"
          "NumberOfPoints: 15");
  CHECK(created_opts == line_segment_opts);

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  const auto expected_block_coord_holders = [&domain_creator]() noexcept {
    const size_t n_pts = 15;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + 0.1 * i;  // Worked out by hand.
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }
  ();

  TestHelpers::db::test_simple_tag<
      intrp::Tags::LineSegment<MockMetavariables::InterpolationTargetA, 3>>(
      "LineSegment");

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables,
      intrp::Tags::LineSegment<MockMetavariables::InterpolationTargetA, 3>>(
      domain_creator, std::move(line_segment_opts),
      expected_block_coord_holders);
}
