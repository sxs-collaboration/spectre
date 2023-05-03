// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/HydroQuantities.hpp"
#include "Elliptic/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {
namespace {

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using analytic_solutions_and_data = tmpl::push_back<
        Xcts::Solutions::all_analytic_solutions,
        Xcts::AnalyticData::Binary<elliptic::analytic_data::AnalyticSolution,
                                   Xcts::Solutions::all_analytic_solutions>>;
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::analytic_data::Background,
                             analytic_solutions_and_data>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.HydroQuantities",
                  "[Unit][Elliptic]") {
  using hydro_tags = AnalyticData::hydro_tags<DataVector>;
  using spatial_metric_tag = gr::Tags::SpatialMetric<DataVector, 3>;
  const Solutions::TovStar tov_star{
      1.e-3, std::make_unique<EquationsOfState::PolytropicFluid<true>>(1., 2.),
      RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
  const double outer_radius = tov_star.radial_solution().outer_radius();
  const tnsr::I<DataVector, 3> x{
      {{{outer_radius / 2., outer_radius * 2.}, {0., 0.}, {0., 0.}}}};
  const auto spatial_metric = get<spatial_metric_tag>(
      tov_star.variables(x, tmpl::list<spatial_metric_tag>{}));
  const auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Coordinates<3, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataVector, 3>,
          elliptic::Tags::Background<elliptic::analytic_data::Background>,
          Parallel::Tags::MetavariablesImpl<Metavariables>>,
      db::AddComputeTags<Tags::HydroQuantitiesCompute<hydro_tags>,
                         Tags::LowerSpatialFourVelocityCompute>>(
      x, spatial_metric,
      std::unique_ptr<elliptic::analytic_data::Background>(
          std::make_unique<Solutions::TovStar>(tov_star)),
      Metavariables{});
  const auto expected_vars = tov_star.variables(x, hydro_tags{});
  tmpl::for_each<hydro_tags>([&box, &expected_vars](const auto tag_v) {
    using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
    CHECK_ITERABLE_APPROX(db::get<tag>(box), get<tag>(expected_vars));
  });
  const auto u_i = db::get<
      hydro::Tags::LowerSpatialFourVelocity<DataVector, 3, Frame::Inertial>>(
      box);
  const auto expected_u_i =
      make_with_value<tnsr::i<DataVector, 3>>(DataVector(2), 0.);
  CHECK_ITERABLE_APPROX(u_i, expected_u_i);
}

}  // namespace Xcts
