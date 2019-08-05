// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace {
struct ArbitraryFrame;
struct ArbitraryType;
}  // namespace

template <size_t Dim, typename Frame, typename Type>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<gr::Tags::SpacetimeMetric<Dim, Frame, Type>>(
      "SpacetimeMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::InverseSpacetimeMetric<Dim, Frame, Type>>(
      "InverseSpacetimeMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::SpatialMetric<Dim, Frame, Type>>(
      "SpatialMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::InverseSpatialMetric<Dim, Frame, Type>>("InverseSpatialMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::DetSpatialMetric<Type>>(
      "DetSpatialMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::SqrtDetSpatialMetric<Type>>(
      "SqrtDetSpatialMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::Shift<Dim, Frame, Type>>("Shift");
  TestHelpers::db::test_simple_tag<gr::Tags::Lapse<Type>>("Lapse");
  TestHelpers::db::test_simple_tag<
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame, Type>>(
      "DerivativesOfSpacetimeMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame, Type>>(
      "SpacetimeChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame, Type>>(
      "SpacetimeChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpatialChristoffelFirstKind<Dim, Frame, Type>>(
      "SpatialChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpatialChristoffelSecondKind<Dim, Frame, Type>>(
      "SpatialChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeNormalOneForm<Dim, Frame, Type>>(
      "SpacetimeNormalOneForm");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeNormalVector<Dim, Frame, Type>>(
      "SpacetimeNormalVector");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim, Frame, Type>>(
      "TraceSpacetimeChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpatialChristoffelFirstKind<Dim, Frame, Type>>(
      "TraceSpatialChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpatialChristoffelSecondKind<Dim, Frame, Type>>(
      "TraceSpatialChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::ExtrinsicCurvature<Dim, Frame, Type>>("ExtrinsicCurvature");
  TestHelpers::db::test_simple_tag<gr::Tags::TraceExtrinsicCurvature<Type>>(
      "TraceExtrinsicCurvature");
  TestHelpers::db::test_simple_tag<gr::Tags::SpatialRicci<Dim, Frame, Type>>(
      "SpatialRicci");
  TestHelpers::db::test_simple_tag<gr::Tags::EnergyDensity<Type>>(
      "EnergyDensity");
  TestHelpers::db::test_simple_tag<gr::Tags::StressTrace<Type>>("StressTrace");
  TestHelpers::db::test_simple_tag<gr::Tags::WeylElectric<Dim, Frame, Type>>(
      "WeylElectric");
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Tags",
                  "[Unit][PointwiseFunctions]") {
  test_simple_tags<1, ArbitraryFrame, ArbitraryType>();
  test_simple_tags<2, ArbitraryFrame, ArbitraryType>();
  test_simple_tags<3, ArbitraryFrame, ArbitraryType>();
}
