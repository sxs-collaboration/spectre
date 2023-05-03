// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace {
struct ArbitraryFrame;
struct ArbitraryType;

struct Tag : db::SimpleTag {
  using type = int;
};
}  // namespace

template <size_t Dim, typename Frame, typename Type>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<gr::Tags::SpacetimeMetric<Type, Dim, Frame>>(
      "SpacetimeMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::InverseSpacetimeMetric<Type, Dim, Frame>>(
      "InverseSpacetimeMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::SpatialMetric<Type, Dim, Frame>>(
      "SpatialMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::InverseSpatialMetric<Type, Dim, Frame>>("InverseSpatialMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::DetSpatialMetric<Type>>(
      "DetSpatialMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::SqrtDetSpatialMetric<Type>>(
      "SqrtDetSpatialMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::DerivDetSpatialMetric<Type, Dim, Frame>>(
      "DerivDetSpatialMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::DerivInverseSpatialMetric<Type, Dim, Frame>>(
      "DerivInverseSpatialMetric");
  TestHelpers::db::test_simple_tag<gr::Tags::Shift<Type, Dim, Frame>>("Shift");
  TestHelpers::db::test_simple_tag<gr::Tags::Lapse<Type>>("Lapse");
  TestHelpers::db::test_simple_tag<
      gr::Tags::DerivativesOfSpacetimeMetric<Type, Dim, Frame>>(
      "DerivativesOfSpacetimeMetric");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeChristoffelFirstKind<Type, Dim, Frame>>(
      "SpacetimeChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeChristoffelSecondKind<Type, Dim, Frame>>(
      "SpacetimeChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpatialChristoffelFirstKind<Type, Dim, Frame>>(
      "SpatialChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpatialChristoffelSecondKind<Type, Dim, Frame>>(
      "SpatialChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeNormalOneForm<Type, Dim, Frame>>(
      "SpacetimeNormalOneForm");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpacetimeNormalVector<Type, Dim, Frame>>(
      "SpacetimeNormalVector");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpacetimeChristoffelFirstKind<Type, Dim, Frame>>(
      "TraceSpacetimeChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpacetimeChristoffelSecondKind<Type, Dim, Frame>>(
      "TraceSpacetimeChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpatialChristoffelFirstKind<Type, Dim, Frame>>(
      "TraceSpatialChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::TraceSpatialChristoffelSecondKind<Type, Dim, Frame>>(
      "TraceSpatialChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      gr::Tags::SpatialChristoffelSecondKindContracted<Type, Dim, Frame>>(
      "SpatialChristoffelSecondKindContracted");
  TestHelpers::db::test_simple_tag<
      gr::Tags::ExtrinsicCurvature<Type, Dim, Frame>>("ExtrinsicCurvature");
  TestHelpers::db::test_simple_tag<gr::Tags::TraceExtrinsicCurvature<Type>>(
      "TraceExtrinsicCurvature");
  TestHelpers::db::test_simple_tag<gr::Tags::SpatialRicci<Type, Dim, Frame>>(
      "SpatialRicci");
  TestHelpers::db::test_simple_tag<gr::Tags::EnergyDensity<Type>>(
      "EnergyDensity");
  TestHelpers::db::test_simple_tag<gr::Tags::MomentumDensity<Type, Dim, Frame>>(
      "MomentumDensity");
  TestHelpers::db::test_simple_tag<gr::Tags::StressTrace<Type>>("StressTrace");
  TestHelpers::db::test_simple_tag<gr::Tags::HamiltonianConstraint<Type>>(
      "HamiltonianConstraint");
  TestHelpers::db::test_simple_tag<
      gr::Tags::MomentumConstraint<Type, Dim, Frame>>("MomentumConstraint");
  TestHelpers::db::test_simple_tag<gr::Tags::WeylElectric<Type, Dim, Frame>>(
      "WeylElectric");
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Tags",
                  "[Unit][PointwiseFunctions]") {
  test_simple_tags<1, ArbitraryFrame, ArbitraryType>();
  test_simple_tags<2, ArbitraryFrame, ArbitraryType>();
  test_simple_tags<3, ArbitraryFrame, ArbitraryType>();

  TestHelpers::db::test_prefix_tag<gr::Tags::Conformal<Tag, -3>>(
      "Conformal(Tag)");
}
