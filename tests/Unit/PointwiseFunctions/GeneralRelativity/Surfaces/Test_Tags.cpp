// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"

namespace {
struct SomeType {};
struct SomeTag : db::SimpleTag {
  using type = SomeType;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Surfaces.Tags",
                  "[ApparentHorizons][Unit]") {
  TestHelpers::db::test_simple_tag<gr::surfaces::Tags::Area>("Area");
  TestHelpers::db::test_simple_tag<gr::surfaces::Tags::IrreducibleMass>(
      "IrreducibleMass");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::OneOverOneFormMagnitude>(
      "OneOverOneFormMagnitude");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::RicciScalar>(
      "RicciScalar");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::MaxRicciScalar>(
      "MaxRicciScalar");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::MinRicciScalar>(
      "MinRicciScalar");
  TestHelpers::db::test_simple_tag<gr::surfaces::Tags::SpinFunction>(
      "SpinFunction");
  TestHelpers::db::test_simple_tag<
      gr::surfaces::Tags::DimensionfulSpinMagnitude>(
      "DimensionfulSpinMagnitude");
  TestHelpers::db::test_simple_tag<gr::surfaces::Tags::ChristodoulouMass>(
      "ChristodoulouMass");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::UnitNormalOneForm<Frame::Inertial>>(
      "UnitNormalOneForm");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::GradUnitNormalOneForm<Frame::Inertial>>(
      "GradUnitNormalOneForm");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::ExtrinsicCurvature<Frame::Inertial>>(
      "ExtrinsicCurvature");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::UnitNormalVector<Frame::Inertial>>("UnitNormalVector");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::EuclideanAreaElement<Frame::Inertial>>(
      "EuclideanAreaElement");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::EuclideanSurfaceIntegral<SomeTag, Frame::Inertial>>(
      "EuclideanSurfaceIntegral(SomeTag)");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::EuclideanSurfaceIntegralVector<SomeTag,
                                                       Frame::Inertial>>(
      "EuclideanSurfaceIntegralVector(SomeTag)");
  TestHelpers::db::test_simple_tag<
      gr::surfaces::Tags::AreaElement<Frame::Inertial>>("AreaElement");
  TestHelpers::db::test_simple_tag<
      gr::surfaces::Tags::SurfaceIntegral<SomeTag, Frame::Inertial>>(
      "SurfaceIntegral(SomeTag)");
  TestHelpers::db::test_simple_tag<
      gr::surfaces::Tags::DimensionfulSpinVector<Frame::Inertial>>(
      "DimensionfulSpinVector");
  TestHelpers::db::test_simple_tag<
      gr::surfaces::Tags::DimensionlessSpinMagnitude<Frame::Inertial>>(
      "DimensionlessSpinMagnitude");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::EuclideanAreaElementCompute<Frame::Inertial>>(
      "EuclideanAreaElement");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::EuclideanSurfaceIntegralCompute<SomeTag,
                                                        Frame::Inertial>>(
      "EuclideanSurfaceIntegral(SomeTag)");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::EuclideanSurfaceIntegralVectorCompute<SomeTag,
                                                              Frame::Inertial>>(
      "EuclideanSurfaceIntegralVector(SomeTag)");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>>("AreaElement");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::SurfaceIntegralCompute<SomeTag, Frame::Inertial>>(
      "SurfaceIntegral(SomeTag)");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::AreaCompute<Frame::Inertial>>("Area");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::IrreducibleMassCompute<Frame::Inertial>>(
      "IrreducibleMass");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::OneOverOneFormMagnitudeCompute<DataVector, 1,
                                                       Frame::Inertial>>(
      "OneOverOneFormMagnitude");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::OneOverOneFormMagnitudeCompute<DataVector, 2,
                                                       Frame::Inertial>>(
      "OneOverOneFormMagnitude");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::OneOverOneFormMagnitudeCompute<DataVector, 3,
                                                       Frame::Inertial>>(
      "OneOverOneFormMagnitude");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::UnitNormalOneFormCompute<Frame::Inertial>>(
      "UnitNormalOneForm");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::GradUnitNormalOneFormCompute<Frame::Inertial>>(
      "GradUnitNormalOneForm");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::ExtrinsicCurvatureCompute<Frame::Inertial>>(
      "ExtrinsicCurvature");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::UnitNormalVectorCompute<Frame::Inertial>>(
      "UnitNormalVector");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::RicciScalarCompute<Frame::Inertial>>("RicciScalar");
  TestHelpers::db::test_compute_tag<StrahlkorperTags::MaxRicciScalarCompute>(
      "MaxRicciScalar");
  TestHelpers::db::test_compute_tag<StrahlkorperTags::MinRicciScalarCompute>(
      "MinRicciScalar");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::SpinFunctionCompute<Frame::Inertial>>("SpinFunction");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::DimensionfulSpinMagnitudeCompute<Frame::Inertial>>(
      "DimensionfulSpinMagnitude");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::ChristodoulouMassCompute<Frame::Inertial>>(
      "ChristodoulouMass");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::DimensionfulSpinVectorCompute<Frame::Inertial,
                                                        Frame::Inertial>>(
      "DimensionfulSpinVector");
  TestHelpers::db::test_compute_tag<
      gr::surfaces::Tags::DimensionlessSpinMagnitudeCompute<Frame::Inertial>>(
      "DimensionlessSpinMagnitude");
}
