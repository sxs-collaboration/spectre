// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Ccz4WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::PowX<1, Frame::Inertial>>>>;
  };
};

template <typename SolutionType>
void test_copy_and_move(const SolutionType& solution) {
  test_copy_semantics(solution);
  auto solution_copy2 = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution_copy2), solution);  // NOLINT
}

template <typename SolutionType>
void test_ccz4_solution(
    const SolutionType& solution,
    const Ccz4::Solutions::Ccz4WrappedGr<SolutionType>& wrapped_solution) {
  const DataVector data_vector{3.0, 4.0};
  const tnsr::I<DataVector, SolutionType::volume_dim, Frame::Inertial> x{
      data_vector};
  // Don't set time to signaling NaN, since not all solutions tested here
  // are static
  const double t = 44.44;

  // Check that the wrapped solution returns the same variables as
  // the solution
  const auto vars = solution.variables(
      x, t, typename SolutionType::template tags<DataVector>{});
  const auto wrapped_vars = wrapped_solution.variables(
      x, t, typename SolutionType::template tags<DataVector>{});

  tmpl::for_each<typename SolutionType::template tags<DataVector>>(
      [&vars, &wrapped_vars](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        CHECK(get<tag>(vars) == get<tag>(wrapped_vars));
      });

  // Check that the wrapped solution returns the correct extra Ccz4 tags
  const auto wrapped_Ccz4_vars = wrapped_solution.variables(
      x, t,
      tmpl::list<
          Ccz4::Tags::ConformalMetric<DataVector, SolutionType::volume_dim>,
          Ccz4::Tags::ConformalFactor<DataVector>,
          Ccz4::Tags::ATilde<DataVector, SolutionType::volume_dim>,
          gr::Tags::TraceExtrinsicCurvature<DataVector>,
          Ccz4::Tags::Theta<DataVector>,
          Ccz4::Tags::GammaHat<DataVector, SolutionType::volume_dim>,
          Ccz4::Tags::AuxiliaryShiftB<DataVector, SolutionType::volume_dim>>{});

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SolutionType::volume_dim>>(vars);
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);
  Scalar<DataVector> conformal_factor;
  get(conformal_factor) = pow(get(sqrt_det_spatial_metric), -1. / 3.);
  CHECK(conformal_factor ==
        get<Ccz4::Tags::ConformalFactor<DataVector>>(wrapped_Ccz4_vars));

  tnsr::ii<DataVector, SolutionType::volume_dim> conformal_spatial_metric;
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&conformal_spatial_metric),
      conformal_factor() * conformal_factor() * spatial_metric(ti::i, ti::j));
  CHECK(conformal_spatial_metric ==
        get<Ccz4::Tags::ConformalMetric<DataVector, SolutionType::volume_dim>>(
            wrapped_Ccz4_vars));

  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, SolutionType::volume_dim>>(
          vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SolutionType::volume_dim>>(
          vars);
  const auto trace_extrinsic_curvature =
      trace(extrinsic_curvature, inverse_spatial_metric);
  CHECK(trace_extrinsic_curvature ==
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(wrapped_Ccz4_vars));

  Scalar<DataVector> conformal_factor_squared;
  get(conformal_factor_squared) = pow(get(sqrt_det_spatial_metric), -2. / 3.);
  const auto a_tilde =
      ::Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                      extrinsic_curvature, trace_extrinsic_curvature);
  CHECK(a_tilde ==
        get<Ccz4::Tags::ATilde<DataVector, SolutionType::volume_dim>>(
            wrapped_Ccz4_vars));

  const auto theta =
      make_with_value<Scalar<DataVector>>(conformal_factor_squared, 0.0);
  CHECK(theta == get<Ccz4::Tags::Theta<DataVector>>(wrapped_Ccz4_vars));

  tnsr::II<DataVector, SolutionType::volume_dim>
      inverse_conformal_spatial_metric;
  ::tenex::evaluate<ti::I, ti::J>(
      make_not_null(&inverse_conformal_spatial_metric),
      inverse_spatial_metric(ti::I, ti::J) / conformal_factor_squared());
  const auto& d_spatial_metric = get<
      Tags::deriv<gr::Tags::SpatialMetric<DataVector, SolutionType::volume_dim>,
                  tmpl::size_t<SolutionType::volume_dim>, Frame::Inertial>>(
      vars);
  tnsr::i<DataVector, SolutionType::volume_dim> d_det_spatial_metric;
  ::tenex::evaluate<ti::i>(make_not_null(&d_det_spatial_metric),
                           sqrt_det_spatial_metric() *
                               sqrt_det_spatial_metric() *
                               inverse_spatial_metric(ti::J, ti::K) *
                               d_spatial_metric(ti::i, ti::j, ti::k));
  Scalar<DataVector> det_spatial_metric_to_minus_four_thirds;
  get(det_spatial_metric_to_minus_four_thirds) =
      pow(get(sqrt_det_spatial_metric), -8. / 3.);
  tnsr::ijj<DataVector, SolutionType::volume_dim> field_d;
  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      make_not_null(&field_d),
      0.5 * conformal_factor_squared() * d_spatial_metric(ti::i, ti::j, ti::k) -
          0.5 * spatial_metric(ti::j, ti::k) * d_det_spatial_metric(ti::i) *
              det_spatial_metric_to_minus_four_thirds() / 3.);
  const auto conformal_christoffel_second_kind =
      ::Ccz4::conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d);
  const auto contracted_conformal_christoffel_second_kind =
      ::Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  // we don't use CHECK here for we switch to tenex to evaluate field_d
  // from the for loops in Ccz4WrappedGr.cpp
  CHECK_ITERABLE_APPROX(
      contracted_conformal_christoffel_second_kind,
      (get<Ccz4::Tags::GammaHat<DataVector, SolutionType::volume_dim>>(
          wrapped_Ccz4_vars)));

  const double one_over_f = 1. / ::Ccz4::fd::System::f;
  const bool shifting_shift = ::Ccz4::fd::System::shifting_shift;
  tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>
      auxiliary_shift_b;
  const auto& shift =
      get<gr::Tags::Shift<DataVector, SolutionType::volume_dim>>(vars);
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<DataVector, SolutionType::volume_dim>,
                    tmpl::size_t<SolutionType::volume_dim>, Frame::Inertial>;
  const auto& d_shift = get<DerivShift>(vars);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, SolutionType::volume_dim>>>(
          vars);
  if (shifting_shift) {
    ::tenex::evaluate<ti::I>(
        make_not_null(&auxiliary_shift_b),
        one_over_f * dt_shift(ti::I) -
            one_over_f * shift(ti::K) * d_shift(ti::k, ti::I));
  } else {
    ::tenex::evaluate<ti::I>(make_not_null(&auxiliary_shift_b),
                             one_over_f * dt_shift(ti::I));
  }
  CHECK(auxiliary_shift_b ==
        get<Ccz4::Tags::AuxiliaryShiftB<DataVector, SolutionType::volume_dim>>(
            wrapped_Ccz4_vars));

  // Weak test of operators == and !=
  CHECK(wrapped_solution == wrapped_solution);
  CHECK_FALSE(wrapped_solution != wrapped_solution);

  if constexpr (std::is_same_v<SolutionType,
                               gr::Solutions::GaugePlaneWave<3>>) {
    register_factory_classes_with_charm<Metavariables>();
  }

  test_serialization(wrapped_solution);
  test_copy_and_move(wrapped_solution);
}

template <typename SolutionType>
void test_construct_from_options(
    const Ccz4::Solutions::Ccz4WrappedGr<SolutionType>& wrapped_solution) {
  const auto created =
      TestHelpers::test_creation<Ccz4::Solutions::Ccz4WrappedGr<SolutionType>>(
          "Mass: 0.5\n"
          "N: 2.5");

  CHECK(created == wrapped_solution);
}

void test_gauge_plane_wave() {
  const auto wave_vector = std::array<double, 3>{{0.1, 0.2, 0.3}};
  const gr::Solutions::GaugePlaneWave<3> gauge_plane_wave_solution{
      wave_vector,
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2)};
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::GaugePlaneWave<3>>
      wrapped_gauge_plane_wave_solution{
          wave_vector,
          std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2)};
  test_ccz4_solution<gr::Solutions::GaugePlaneWave<3>>(
      gauge_plane_wave_solution, wrapped_gauge_plane_wave_solution);
}
}  // namespace

// [[TimeOut, 40]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.Ccz4WrappedGr",
                  "[PointwiseFunctions][Unit]") {
  const double amplitude = 0.24;
  const double wavelength = 4.4;
  const gr::Solutions::GaugeWave<3> gauge_wave_solution{amplitude, wavelength};
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::GaugeWave<3>>
      wrapped_gauge_wave_solution{amplitude, wavelength};
  test_ccz4_solution<gr::Solutions::GaugeWave<3>>(gauge_wave_solution,
                                                  wrapped_gauge_wave_solution);

  test_gauge_plane_wave();

  const gr::Solutions::Minkowski<3> minkowski_solution{};
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>
      wrapped_minkowski_solution{};
  test_ccz4_solution<gr::Solutions::Minkowski<3>>(minkowski_solution,
                                                  wrapped_minkowski_solution);

  const double mass = 0.5;
  const double n = 2.5;
  const gr::Solutions::TrumpetSchwarzschild trumpet_schwarzschild_solution{mass,
                                                                           n};
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::TrumpetSchwarzschild>
      wrapped_trumpet_schwarzschild_solution{mass, n};
  test_ccz4_solution<gr::Solutions::TrumpetSchwarzschild>(
      trumpet_schwarzschild_solution, wrapped_trumpet_schwarzschild_solution);

  test_construct_from_options<gr::Solutions::TrumpetSchwarzschild>(
      wrapped_trumpet_schwarzschild_solution);
}
