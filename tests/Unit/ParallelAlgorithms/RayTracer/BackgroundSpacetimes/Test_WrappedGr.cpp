// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"

namespace ray_tracing {
namespace {

using DerivSpatialMetric =
    ::Tags::deriv<gr::Tags::SpatialMetric<double, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>;
using DerivInvSpatialMetric =
    ::Tags::deriv<gr::Tags::InverseSpatialMetric<double, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>;

template <typename SolutionType>
void test_wrapped_gr(const std::string& options_string) {
  CAPTURE(pretty_type::name<SolutionType>());
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      BackgroundSpacetime, WrappedGr<SolutionType>>(options_string);
  REQUIRE(dynamic_cast<const WrappedGr<SolutionType>*>(created.get()) !=
          nullptr);
  const auto& background_spacetime =
      dynamic_cast<const WrappedGr<SolutionType>&>(*created);
  {
    INFO("Semantics");
    test_serialization(background_spacetime);
    test_copy_semantics(background_spacetime);
    auto move_background_spacetime = background_spacetime;
    test_move_semantics(std::move(move_background_spacetime),
                        background_spacetime);
    const auto clone = background_spacetime.get_clone();
    REQUIRE(dynamic_cast<const WrappedGr<SolutionType>*>(clone.get()) !=
            nullptr);
    CHECK(dynamic_cast<const WrappedGr<SolutionType>&>(*clone) ==
          background_spacetime);
  }
  {
    INFO("Variables");
    const tnsr::I<double, 3> x{{3.0, 4.0, 5.0}};
    const auto vars = background_spacetime.variables(x, 0.0);
    const auto solution_vars =
        background_spacetime.wrapped_solution().variables(
            x, 0.0,
            tmpl::push_back<
                tmpl::remove<BackgroundSpacetime::tags, DerivInvSpatialMetric>,
                DerivSpatialMetric>{});
    const auto expected_vars = tuples::tagged_tuple_cat(
        solution_vars,
        tuples::TaggedTuple<DerivInvSpatialMetric>{
            gr::deriv_inverse_spatial_metric(
                get<gr::Tags::InverseSpatialMetric<double, 3>>(solution_vars),
                get<DerivSpatialMetric>(solution_vars))});
    tmpl::for_each<BackgroundSpacetime::tags>([&](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(expected_vars));
    });
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RayTracer.BackgroundSpacetimes.WrappedGr",
                  "[Unit][ParallelAlgorithms]") {
  test_wrapped_gr<gr::Solutions::KerrSchild>(
      "KerrSchild:\n"
      "  Mass: 1.0\n"
      "  Spin: [0.0, 0.0, 0.5]\n"
      "  Center: [1.0, 0.0, 0.0]\n"
      "  Velocity: [0.0, 0.0, 0.2]\n");
  test_wrapped_gr<gr::Solutions::Minkowski<3>>("Minkowski");
}

}  // namespace ray_tracing
