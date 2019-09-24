// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace {
struct ArbitraryFrame;
struct ArbitraryType;
}  // namespace

template <size_t Dim, typename Frame, typename Type>
void test_simple_tags() {
  CHECK(db::tag_name<gr::Tags::SpacetimeMetric<Dim, Frame, Type>>() ==
        "SpacetimeMetric");
  CHECK(db::tag_name<gr::Tags::InverseSpacetimeMetric<Dim, Frame, Type>>() ==
        "InverseSpacetimeMetric");
  CHECK(db::tag_name<gr::Tags::InverseSpatialMetric<Dim, Frame, Type>>() ==
        "InverseSpatialMetric");
  CHECK(db::tag_name<gr::Tags::DetSpatialMetric<Type>>() == "DetSpatialMetric");
  CHECK(db::tag_name<gr::Tags::SqrtDetSpatialMetric<Type>>() ==
        "SqrtDetSpatialMetric");
  CHECK(db::tag_name<gr::Tags::Shift<Dim, Frame, Type>>() == "Shift");
  CHECK(db::tag_name<gr::Tags::Lapse<Type>>() == "Lapse");
  CHECK(db::tag_name<gr::Tags::DerivSpacetimeMetric<Dim, Frame, Type>>() ==
        "DerivSpacetimeMetric");
  CHECK(db::tag_name<
            gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame, Type>>() ==
        "DerivativesOfSpacetimeMetric");
  CHECK(db::tag_name<
            gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame, Type>>() ==
        "SpacetimeChristoffelFirstKind");
  CHECK(db::tag_name<
            gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame, Type>>() ==
        "SpacetimeChristoffelSecondKind");
  CHECK(
      db::tag_name<gr::Tags::SpatialChristoffelFirstKind<Dim, Frame, Type>>() ==
      "SpatialChristoffelFirstKind");
  CHECK(db::tag_name<
            gr::Tags::SpatialChristoffelSecondKind<Dim, Frame, Type>>() ==
        "SpatialChristoffelSecondKind");
  CHECK(db::tag_name<gr::Tags::SpacetimeNormalOneForm<Dim, Frame, Type>>() ==
        "SpacetimeNormalOneForm");
  CHECK(db::tag_name<gr::Tags::SpacetimeNormalVector<Dim, Frame, Type>>() ==
        "SpacetimeNormalVector");
  CHECK(db::tag_name<
            gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim, Frame, Type>>() ==
        "TraceSpacetimeChristoffelFirstKind");
  CHECK(db::tag_name<
            gr::Tags::TraceSpatialChristoffelFirstKind<Dim, Frame, Type>>() ==
        "TraceSpatialChristoffelFirstKind");
  CHECK(db::tag_name<
            gr::Tags::TraceSpatialChristoffelSecondKind<Dim, Frame, Type>>() ==
        "TraceSpatialChristoffelSecondKind");
  CHECK(db::tag_name<gr::Tags::ExtrinsicCurvature<Dim, Frame, Type>>() ==
        "ExtrinsicCurvature");
  CHECK(db::tag_name<gr::Tags::TraceExtrinsicCurvature<Type>>() ==
        "TraceExtrinsicCurvature");
  CHECK(db::tag_name<gr::Tags::EnergyDensity<Type>>() == "EnergyDensity");
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Tags",
                  "[Unit][PointwiseFunctions]") {
  test_simple_tags<1, ArbitraryFrame, ArbitraryType>();
  test_simple_tags<2, ArbitraryFrame, ArbitraryType>();
  test_simple_tags<3, ArbitraryFrame, ArbitraryType>();
}
