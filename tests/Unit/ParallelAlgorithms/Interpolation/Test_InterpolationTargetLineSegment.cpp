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
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <InterpTargetTestHelpers::ValidPoints ValidPoints>
domain::creators::Sphere make_sphere() {
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::All) {
    return {0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
  }
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::None) {
    return {4.9, 8.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
  }
  return {3.4, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
}

template <typename Frame>
struct LineSegmentTag
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::TimeStepId;
  using vars_to_interpolate_to_target = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using compute_items_on_target = tmpl::list<>;
  using compute_target_points =
      ::intrp::TargetPoints::LineSegment<LineSegmentTag, 3, Frame>;
  using post_interpolation_callbacks = tmpl::list<>;
};

template <InterpTargetTestHelpers::ValidPoints ValidPoints>
void test() {
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

  const auto domain_creator = make_sphere<ValidPoints>();

  const auto expected_block_coord_holders = [&domain_creator]() {
    const size_t n_pts = 15;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + 0.1 * i;  // Worked out by hand.
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  TestHelpers::db::test_simple_tag<
      intrp::Tags::LineSegment<LineSegmentTag<Frame::Grid>, 3>>("LineSegment");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::LineSegment<LineSegmentTag<Frame::Inertial>, 3>>(
      "LineSegment");

  InterpTargetTestHelpers::test_interpolation_target<
      LineSegmentTag<Frame::Grid>, 3,
      intrp::Tags::LineSegment<LineSegmentTag<Frame::Grid>, 3>>(
      line_segment_opts, expected_block_coord_holders);
  InterpTargetTestHelpers::test_interpolation_target<
      LineSegmentTag<Frame::Inertial>, 3,
      intrp::Tags::LineSegment<LineSegmentTag<Frame::Inertial>, 3>>(
      line_segment_opts, expected_block_coord_holders);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.LineSegment",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  test<InterpTargetTestHelpers::ValidPoints::All>();
  test<InterpTargetTestHelpers::ValidPoints::Some>();
  test<InterpTargetTestHelpers::ValidPoints::None>();
}
