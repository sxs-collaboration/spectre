// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeSpacetimeTags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CSW.Worldtube.InitializeSpacetimeTags",
    "[Unit][Evolution]") {
  for (const double orbital_radius :
       make_array<double>(2.5, 3., 5., 6.1234, 15., 173.6)) {
    CAPTURE(orbital_radius);
    static constexpr size_t Dim = 3;
    const double angular_velocity =
        1. / (orbital_radius * sqrt(orbital_radius));
    const tnsr::I<double, Dim, Frame::Grid> wt_center{{orbital_radius, 0., 0.}};
    auto box = db::create<db::AddSimpleTags<
        gr::Tags::InverseSpacetimeMetric<double, Dim, Frame::Grid>,
        gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim, Frame::Grid>,
        Tags::ExcisionSphere<Dim>>>(tnsr::AA<double, Dim, Frame::Grid>{},
                                    tnsr::A<double, Dim, Frame::Grid>{},
                                    ExcisionSphere<Dim>(1., wt_center, {}));

    db::mutate_apply<Initialization::InitializeSpacetimeTags>(
        make_not_null(&box));

    const gr::Solutions::KerrSchild kerr_schild(1., {0., 0., 0.}, {0., 0., 0.});
    using background_tags = tmpl::list<
        gr::Tags::Lapse<double>, ::Tags::dt<gr::Tags::Lapse<double>>,
        ::Tags::deriv<gr::Tags::Lapse<double>, tmpl::size_t<Dim>, Frame::Grid>,
        gr::Tags::Shift<double, Dim, Frame::Grid>,
        ::Tags::dt<gr::Tags::Shift<double, Dim, Frame::Grid>>,
        ::Tags::deriv<gr::Tags::Shift<double, Dim, Frame::Grid>,
                      tmpl::size_t<Dim>, Frame::Grid>,
        gr::Tags::SpatialMetric<double, Dim, Frame::Grid>,
        gr::Tags::InverseSpatialMetric<double, Dim, Frame::Grid>,
        ::Tags::dt<gr::Tags::SpatialMetric<double, Dim, Frame::Grid>>,
        ::Tags::deriv<gr::Tags::SpatialMetric<double, Dim, Frame::Grid>,
                      tmpl::size_t<Dim>, Frame::Grid>>;
    const auto background_vars =
        kerr_schild.variables(wt_center, 0., background_tags{});

    const auto inverse_metric = gr::inverse_spacetime_metric(
        get<gr::Tags::Lapse<double>>(background_vars),
        get<gr::Tags::Shift<double, Dim, Frame::Grid>>(background_vars),
        get<gr::Tags::InverseSpatialMetric<double, Dim, Frame::Grid>>(
            background_vars));

    // jacobian and hessian to transform into a frame co-rotating about the
    // z-axis evaluated at (t,x,y,z) = (0,R,0,0)
    tnsr::Ab<double, Dim, Frame::Grid> jacobian(0.);
    for (size_t i = 0; i < 4; ++i) {
      jacobian.get(i, i) = 1.;
    }
    get<2, 0>(jacobian) = -orbital_radius * angular_velocity;

    tnsr::AA<double, Dim, Frame::Grid> boosted_inverse_metric{};
    tenex::evaluate<ti::A, ti::B>(make_not_null(&boosted_inverse_metric),
                                  jacobian(ti::A, ti::c) *
                                      jacobian(ti::B, ti::d) *
                                      inverse_metric(ti::C, ti::D));
    const auto& inverse_spacetime_metric =
        get<gr::Tags::InverseSpacetimeMetric<double, Dim, Frame::Grid>>(box);
    CHECK_ITERABLE_APPROX(boosted_inverse_metric, inverse_spacetime_metric);

    tnsr::Abb<double, Dim, Frame::Grid> inverse_hessian(0.);
    get<1, 0, 0>(inverse_hessian) =
        -angular_velocity * angular_velocity * orbital_radius;
    get<1, 0, 2>(inverse_hessian) = -angular_velocity;
    get<2, 0, 1>(inverse_hessian) = angular_velocity;

    const auto d_spacetime_metric = gr::derivatives_of_spacetime_metric(
        get<gr::Tags::Lapse<double>>(background_vars),
        get<::Tags::dt<gr::Tags::Lapse<double>>>(background_vars),
        get<::Tags::deriv<gr::Tags::Lapse<double>, tmpl::size_t<Dim>,
                          Frame::Grid>>(background_vars),
        get<gr::Tags::Shift<double, Dim, Frame::Grid>>(background_vars),
        get<::Tags::dt<gr::Tags::Shift<double, Dim, Frame::Grid>>>(
            background_vars),
        get<::Tags::deriv<gr::Tags::Shift<double, Dim, Frame::Grid>,
                          tmpl::size_t<Dim>, Frame::Grid>>(background_vars),
        get<gr::Tags::SpatialMetric<double, Dim, Frame::Grid>>(background_vars),
        get<::Tags::dt<gr::Tags::SpatialMetric<double, Dim, Frame::Grid>>>(
            background_vars),
        get<::Tags::deriv<gr::Tags::SpatialMetric<double, Dim, Frame::Grid>,
                          tmpl::size_t<Dim>, Frame::Grid>>(background_vars));

    const auto christoffel =
        gr::christoffel_second_kind(d_spacetime_metric, inverse_metric);

    const auto inverse_jacobian = determinant_and_inverse(jacobian).second;

    // christoffel symbol needs Hessian to transform
    tnsr::Abb<double, Dim, Frame::Grid> boosted_christoffel{};
    tenex::evaluate<ti::A, ti::b, ti::c>(
        make_not_null(&boosted_christoffel),
        jacobian(ti::A, ti::d) * inverse_jacobian(ti::E, ti::b) *
                inverse_jacobian(ti::F, ti::c) *
                christoffel(ti::D, ti::e, ti::f) +
            inverse_hessian(ti::D, ti::b, ti::c) * jacobian(ti::A, ti::d));
    const auto contracted_boosted_christoffel =
        tenex::evaluate<ti::A>(boosted_inverse_metric(ti::B, ti::C) *
                               boosted_christoffel(ti::A, ti::b, ti::c));
    const auto& contracted_christoffel =
        get<gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim,
                                                          Frame::Grid>>(box);
    CHECK_ITERABLE_APPROX(contracted_christoffel,
                          contracted_boosted_christoffel);
  }
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
