// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/TagsMetafunctions.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

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
  using compute_target_points =
      intrp::TargetPoints::Sphere<InterpolationTargetTag, Frame::Inertial>;
};

void test_tags_metafunctions() {
  static_assert(std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                                   gh::Tags::Pi<DataVector, 3>, Frame::Grid>,
                               gh::Tags::Pi<DataVector, 3, Frame::Grid>>,
                "Failed testing replace_frame_in_tag_t");
  static_assert(
      not std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                             gh::Tags::Pi<DataVector, 3>, Frame::Grid>,
                         gh::Tags::Pi<DataVector, 3, Frame::Distorted>>,
      "Failed testing replace_frame_in_tag_t");
  static_assert(
      std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                         gr::Tags::SpacetimeMetric<DataVector, 3>, Frame::Grid>,
                     gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Grid>>,
      "Failed testing replace_frame_in_tag_t");
  static_assert(std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                                   gr::Tags::Lapse<DataVector>, Frame::Grid>,
                               gr::Tags::Lapse<DataVector>>,
                "Failed testing replace_frame_in_tag_t");
  static_assert(
      std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                         Tags::deriv<gr::Tags::Lapse<DataVector>,
                                     tmpl::size_t<3>, Frame::Inertial>,
                         Frame::Grid>,
                     Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                 Frame::Grid>>,
      "Failed testing replace_frame_in_tag_t");
  static_assert(std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                                   gh::Tags::ConstraintGamma0, Frame::Grid>,
                               gh::Tags::ConstraintGamma0>,
                "Failed testing replace_frame_in_tag_t");
  static_assert(
      std::is_same_v<TensorMetafunctions::replace_frame_in_tag_t<
                         Tags::deriv<gh::Tags::Phi<DataVector, 3>,
                                     tmpl::size_t<3>, Frame::Inertial>,
                         Frame::Grid>,
                     Tags::deriv<gh::Tags::Phi<DataVector, 3, Frame::Grid>,
                                 tmpl::size_t<3>, Frame::Grid>>,
      "Failed testing replace_frame_in_tag_t");
  static_assert(
      std::is_same_v<
          TensorMetafunctions::replace_frame_in_taglist<
              tmpl::list<Tags::deriv<gh::Tags::Phi<DataVector, 3>,
                                     tmpl::size_t<3>, Frame::Inertial>,
                         gh::Tags::Pi<DataVector, 3, Frame::Distorted>,
                         Tags::deriv<gr::Tags::Lapse<DataVector>,
                                     tmpl::size_t<3>, Frame::Distorted>>,
              Frame::Grid>,
          tmpl::list<Tags::deriv<gh::Tags::Phi<DataVector, 3, Frame::Grid>,
                                 tmpl::size_t<3>, Frame::Grid>,
                     gh::Tags::Pi<DataVector, 3, Frame::Grid>,
                     Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                 Frame::Grid>>>,
      "Failed testing replace_frame_in_taglist");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Interpolation.Tags", "[Unit][NumericalAlgorithms]") {
  test_tags_metafunctions();
  TestHelpers::db::test_simple_tag<intrp::Tags::DumpVolumeDataOnFailure>(
      "DumpVolumeDataOnFailure");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::IndicesOfFilledInterpPoints<Metavars>>(
      "IndicesOfFilledInterpPoints");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::IndicesOfInvalidInterpPoints<Metavars>>(
      "IndicesOfInvalidInterpPoints");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::InterpolatedVars<InterpolationTargetTag, Metavars>>(
      "InterpolatedVars");
  TestHelpers::db::test_simple_tag<intrp::Tags::CurrentTemporalId<Metavars>>(
      "CurrentTemporalId");
  TestHelpers::db::test_simple_tag<intrp::Tags::TemporalIds<Metavars>>(
      "TemporalIds");
  TestHelpers::db::test_simple_tag<intrp::Tags::CompletedTemporalIds<Metavars>>(
      "CompletedTemporalIds");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::VolumeVarsInfo<Metavars, SomeTag>>("VolumeVarsInfo");
  TestHelpers::db::test_simple_tag<
      intrp::Tags::InterpolatedVarsHolders<Metavars>>(
      "InterpolatedVarsHolders");
  TestHelpers::db::test_simple_tag<intrp::Tags::NumberOfElements<3>>(
      "NumberOfElements");
  TestHelpers::db::test_simple_tag<intrp::Tags::PointInfo<
      InterpolationTargetTag, tmpl::size_t<Metavars::volume_dim>>>("PointInfo");
  TestHelpers::db::test_simple_tag<intrp::Tags::Verbosity>("Verbosity");

  CHECK(
      TestHelpers::test_option_tag<intrp::OptionTags::DumpVolumeDataOnFailure>(
          "true"));
  CHECK_FALSE(
      TestHelpers::test_option_tag<intrp::OptionTags::DumpVolumeDataOnFailure>(
          "false"));
}
