// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/PointInfoTag.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavars {
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<>;
};
struct InterpolationTargetTag {
  using vars_to_interpolate_to_target = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Interpolation.Tags", "[Unit][NumericalAlgorithms]") {
  TestHelpers::db::test_simple_tag<intrp::Tags::IndicesOfFilledInterpPoints>(
      "IndicesOfFilledInterpPoints");
  TestHelpers::db::test_simple_tag<intrp::Tags::IndicesOfInvalidInterpPoints>(
      "IndicesOfInvalidInterpPoints");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::InterpolatedVars<InterpolationTargetTag>>(
      "InterpolatedVars");
  TestHelpers::db::test_simple_tag<intrp::Tags::Times>("Times");
  TestHelpers::db::test_simple_tag<intrp::Tags::CompletedTimes>(
      "CompletedTimes");
  TestHelpers::db::test_simple_tag<intrp::Tags::VolumeVarsInfo<Metavars>>(
      "VolumeVarsInfo");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::InterpolatedVarsHolders<Metavars>>(
      "InterpolatedVarsHolders");
  TestHelpers::db::test_simple_tag<intrp::Tags::NumberOfElements>(
      "NumberOfElements");
  TestHelpers::db::test_simple_tag<intrp::Tags::InterpPointInfo<Metavars>>(
      "InterpPointInfo");
}
