// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetLineSegment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace {
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_target_points =
        ::intrp::Actions::LineSegment<InterpolationTargetA, 3>;
    using type = compute_target_points::options_type;
  };
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 InterpTargetTestHelpers::mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.LineSegment",
                  "[Unit]") {
  // Options for LineSegment
  intrp::OptionHolders::LineSegment<3> line_segment_opts({{1.0, 1.0, 1.0}},
                                                         {{2.4, 2.4, 2.4}}, 15);

  // Test creation of options
  const auto created_opts = test_creation<intrp::OptionHolders::LineSegment<3>>(
      "  Begin: [1.0, 1.0, 1.0]\n"
      "  End: [2.4, 2.4, 2.4]\n"
      "  NumberOfPoints: 15");
  CHECK(created_opts == line_segment_opts);

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

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

  InterpTargetTestHelpers::test_interpolation_target<MockMetavariables>(
      domain_creator, std::move(line_segment_opts),
      expected_block_coord_holders);
}
