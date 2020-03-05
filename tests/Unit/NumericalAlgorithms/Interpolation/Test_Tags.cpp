// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeType {};
struct SomeTag {
  using type = SomeType;
};
struct Metavars {
  using temporal_id = SomeTag;
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<>;
};
struct InterpolationTargetTag {
  using vars_to_interpolate_to_target = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Interpolation.Tags", "[Unit][NumericalAlgorithms]") {
  TestHelpers::db::test_simple_tag<
      intrp::Tags::IndicesOfFilledInterpPoints<Metavars>>(
      "IndicesOfFilledInterpPoints");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::IndicesOfInvalidInterpPoints<Metavars>>(
      "IndicesOfInvalidInterpPoints");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::InterpolatedVars<InterpolationTargetTag, Metavars>>(
      "InterpolatedVars");
  TestHelpers::db::test_simple_tag<intrp::Tags::TemporalIds<Metavars>>(
      "TemporalIds");
  TestHelpers::db::test_simple_tag<intrp::Tags::CompletedTemporalIds<Metavars>>(
      "CompletedTemporalIds");
  TestHelpers::db::test_simple_tag<intrp::Tags::VolumeVarsInfo<Metavars>>(
      "VolumeVarsInfo");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::InterpolatedVarsHolders<Metavars>>(
      "InterpolatedVarsHolders");
  TestHelpers::db::test_simple_tag<intrp::Tags::NumberOfElements>(
      "NumberOfElements");
}
