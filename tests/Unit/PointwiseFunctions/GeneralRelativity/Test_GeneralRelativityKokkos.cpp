// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Expansion1D.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SecondTimeDerivOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfDetSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

// The inputs are positive, so the functions don't divide by zero or take square
// roots of negative numbers.

namespace {
void test_spacetime_decomposition(
    const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto lapse = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<double, 3>>(
          generator, make_not_null(&distribution));
  // `gr::lapse` takes a square root that is only positive for a spacetime
  // metric that is consistent with a lapse
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto dt_lapse = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto deriv_lapse = make_with_random_values<tnsr::i<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto deriv_shift = make_with_random_values<tnsr::iJ<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto deriv_spatial_metric =
      make_with_random_values<tnsr::ijj<double, 3>>(
          generator, make_not_null(&distribution));
  const auto extrinsic_curvature = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(
        gr::lapse(shift, spacetime_metric),
        gr::shift(spacetime_metric, inverse_spatial_metric),
        gr::spatial_metric(spacetime_metric),
        gr::spacetime_metric(lapse, shift, spatial_metric),
        gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric),
        gr::spacetime_normal_vector(lapse, shift),
        gr::spacetime_normal_one_form<double, 3, Frame::Inertial>(lapse),
        gr::time_derivative_of_spacetime_metric(lapse, dt_lapse, shift,
                                                dt_shift, spatial_metric,
                                                dt_spatial_metric),
        gr::time_derivative_of_spatial_metric(
            lapse, shift, deriv_shift, spatial_metric, deriv_spatial_metric,
            extrinsic_curvature),
        gr::derivatives_of_spacetime_metric(
            lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
            spatial_metric, dt_spatial_metric, deriv_spatial_metric),
        gr::deriv_inverse_spatial_metric(inverse_spatial_metric,
                                         deriv_spatial_metric));
  });
}

void test_curvature(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<double, 3>>(
          generator, make_not_null(&distribution));
  const auto inverse_spacetime_metric =
      make_with_random_values<tnsr::AA<double, 3>>(
          generator, make_not_null(&distribution));
  const auto d_spacetime_metric = make_with_random_values<tnsr::abb<double, 3>>(
      generator, make_not_null(&distribution));
  const auto d_christoffel = make_with_random_values<tnsr::aBcc<double, 3>>(
      generator, make_not_null(&distribution));
  const auto spatial_ricci = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto extrinsic_curvature = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto cov_deriv_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<double, 3>>(
          generator, make_not_null(&distribution));
  const auto interface_normal_vector =
      make_with_random_values<tnsr::I<double, 3>>(generator,
                                                  make_not_null(&distribution));
  const auto interface_normal_one_form =
      make_with_random_values<tnsr::i<double, 3>>(generator,
                                                  make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    const auto christoffel = gr::christoffel_second_kind(
        d_spacetime_metric, inverse_spacetime_metric);
    const auto ricci = gr::ricci_tensor(christoffel, d_christoffel);
    const auto weyl_electric = gr::weyl_electric(
        spatial_ricci, extrinsic_curvature, inverse_spatial_metric);
    const auto weyl_type_d1 =
        gr::weyl_type_D1(weyl_electric, spatial_metric, inverse_spatial_metric);
    return TestHelpers::flatten(
        gr::christoffel_first_kind(d_spacetime_metric), christoffel, ricci,
        gr::ricci_scalar(ricci, inverse_spacetime_metric), weyl_electric,
        gr::weyl_electric_scalar(weyl_electric, inverse_spatial_metric),
        weyl_type_d1,
        gr::weyl_type_D1_scalar(weyl_type_d1, inverse_spatial_metric),
        gr::weyl_propagating(
            spatial_ricci, extrinsic_curvature, inverse_spatial_metric,
            cov_deriv_extrinsic_curvature, interface_normal_vector,
            gr::transverse_projection_operator(inverse_spatial_metric,
                                               interface_normal_vector),
            gr::transverse_projection_operator(spatial_metric,
                                               interface_normal_one_form),
            gr::transverse_projection_operator(interface_normal_vector,
                                               interface_normal_one_form),
            1.0));
  });
}

void test_projections(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<double, 3>>(
          generator, make_not_null(&distribution));
  const auto spacetime_metric = make_with_random_values<tnsr::aa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_spacetime_metric =
      make_with_random_values<tnsr::AA<double, 3>>(
          generator, make_not_null(&distribution));
  const auto normal_vector = make_with_random_values<tnsr::A<double, 3>>(
      generator, make_not_null(&distribution));
  const auto normal_one_form = make_with_random_values<tnsr::a<double, 3>>(
      generator, make_not_null(&distribution));
  const auto interface_normal_vector =
      make_with_random_values<tnsr::I<double, 3>>(generator,
                                                  make_not_null(&distribution));
  const auto interface_normal_one_form =
      make_with_random_values<tnsr::i<double, 3>>(generator,
                                                  make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(
        gr::interface_null_normal(normal_one_form, interface_normal_one_form,
                                  shift, 1.0),
        gr::interface_null_normal(normal_vector, interface_normal_vector, -1.0),
        gr::transverse_projection_operator(inverse_spatial_metric,
                                           interface_normal_vector),
        gr::transverse_projection_operator(spatial_metric,
                                           interface_normal_one_form),
        gr::transverse_projection_operator(interface_normal_vector,
                                           interface_normal_one_form),
        gr::transverse_projection_operator(spacetime_metric, normal_one_form,
                                           interface_normal_one_form, shift),
        gr::transverse_projection_operator(
            inverse_spacetime_metric, normal_vector, interface_normal_vector),
        gr::transverse_projection_operator(normal_vector, normal_one_form,
                                           interface_normal_vector,
                                           interface_normal_one_form, shift));
  });
}

void test_gh_derivatives(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto lapse = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto sqrt_det_spatial_metric = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<double, 3>>(
          generator, make_not_null(&distribution));
  const auto inverse_spacetime_metric =
      make_with_random_values<tnsr::AA<double, 3>>(
          generator, make_not_null(&distribution));
  const auto normal_vector = make_with_random_values<tnsr::A<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_lapse = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto deriv_lapse = make_with_random_values<tnsr::i<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto deriv_shift = make_with_random_values<tnsr::iJ<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto deriv_spatial_metric =
      make_with_random_values<tnsr::ijj<double, 3>>(
          generator, make_not_null(&distribution));
  const auto phi = make_with_random_values<tnsr::iaa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto pi = make_with_random_values<tnsr::aa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_phi = make_with_random_values<tnsr::iaa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto dt_pi = make_with_random_values<tnsr::aa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<double>>(generator,
                                              make_not_null(&distribution));
  const auto trace_christoffel_last_indices =
      make_with_random_values<tnsr::i<double, 3>>(generator,
                                                  make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    // This function has no return-by-value overload
    tnsr::abb<double, 3> d_spacetime_metric{};
    gh::spacetime_derivative_of_spacetime_metric(
        make_not_null(&d_spacetime_metric), lapse, shift, pi, phi);
    return TestHelpers::flatten(
        d_spacetime_metric, gh::deriv_spatial_metric(phi),
        gh::spacetime_deriv_of_det_spatial_metric(sqrt_det_spatial_metric,
                                                  inverse_spatial_metric,
                                                  dt_spatial_metric, phi),
        gh::spatial_deriv_of_lapse(lapse, normal_vector, phi),
        gh::spatial_deriv_of_shift(lapse, inverse_spacetime_metric,
                                   normal_vector, phi),
        gh::time_deriv_of_lapse(lapse, shift, normal_vector, phi, pi),
        gh::time_deriv_of_shift(lapse, shift, inverse_spatial_metric,
                                normal_vector, phi, pi),
        gh::time_deriv_of_spatial_metric(lapse, shift, phi, pi),
        gh::phi(lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
                deriv_spatial_metric),
        gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
               dt_spatial_metric, phi),
        gh::gauge_source(lapse, dt_lapse, deriv_lapse, shift, dt_shift,
                         deriv_shift, spatial_metric, trace_extrinsic_curvature,
                         trace_christoffel_last_indices),
        gh::second_time_deriv_of_spacetime_metric(
            lapse, dt_lapse, shift, dt_shift, phi, dt_phi, pi, dt_pi),
        gh::time_derivative_of_spacetime_metric(lapse, shift, pi, phi));
  });
}

void test_gh_curvature(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto spatial_metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<double, 3>>(
          generator, make_not_null(&distribution));
  const auto inverse_spacetime_metric =
      make_with_random_values<tnsr::AA<double, 3>>(
          generator, make_not_null(&distribution));
  const auto normal_vector = make_with_random_values<tnsr::A<double, 3>>(
      generator, make_not_null(&distribution));
  const auto normal_one_form = make_with_random_values<tnsr::a<double, 3>>(
      generator, make_not_null(&distribution));
  const auto deriv_spatial_metric =
      make_with_random_values<tnsr::ijj<double, 3>>(
          generator, make_not_null(&distribution));
  const auto phi = make_with_random_values<tnsr::iaa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto pi = make_with_random_values<tnsr::aa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto d_phi = make_with_random_values<tnsr::ijaa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto d_pi = make_with_random_values<tnsr::iaa<double, 3>>(
      generator, make_not_null(&distribution));
  const auto extrinsic_curvature = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto spatial_christoffel =
      make_with_random_values<tnsr::Ijj<double, 3>>(
          generator, make_not_null(&distribution));
  const auto coords = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(
        gh::christoffel_second_kind(phi, inverse_spatial_metric),
        gh::trace_christoffel(normal_one_form, normal_vector,
                              inverse_spatial_metric, inverse_spacetime_metric,
                              pi, phi),
        gh::extrinsic_curvature(normal_vector, pi, phi),
        gh::covariant_deriv_of_extrinsic_curvature(
            extrinsic_curvature, normal_vector, spatial_christoffel,
            inverse_spacetime_metric, phi, d_pi, d_phi),
        gh::spatial_ricci_tensor(phi, d_phi, inverse_spatial_metric),
        gh::expansion1D(spatial_metric, deriv_spatial_metric,
                        extrinsic_curvature, coords));
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Kokkos",
                  "[Unit][PointwiseFunctions]") {
  MAKE_GENERATOR(generator);
  test_spacetime_decomposition(make_not_null(&generator));
  test_curvature(make_not_null(&generator));
  test_projections(make_not_null(&generator));
  test_gh_derivatives(make_not_null(&generator));
  test_gh_curvature(make_not_null(&generator));
}
