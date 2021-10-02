// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/AnalyticSolutions/AnalyticDataHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::Solutions {

namespace {
template <int Spin>
void check_22_and_33_modes(
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
    const std::complex<double>& expected_22_mode,
    const std::complex<double>& expected_33_mode, const size_t l_max,
    Approx bondi_approx) {
  CHECK_COMPLEX_CUSTOM_APPROX(
      goldberg_modes.data()[Spectral::Swsh::goldberg_mode_index(l_max, 2, 2)],
      expected_22_mode, bondi_approx);
  CHECK_COMPLEX_CUSTOM_APPROX(
      goldberg_modes.data()[Spectral::Swsh::goldberg_mode_index(l_max, 2, -2)],
      expected_22_mode, bondi_approx);
  CHECK_COMPLEX_CUSTOM_APPROX(
      goldberg_modes.data()[Spectral::Swsh::goldberg_mode_index(l_max, 3, 3)],
      expected_33_mode, bondi_approx);
  CHECK_COMPLEX_CUSTOM_APPROX(
      goldberg_modes.data()[Spectral::Swsh::goldberg_mode_index(l_max, 3, -3)],
      -expected_33_mode, bondi_approx);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.LinearizedBondiSachs",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> parameter_dist{0.1, 1.0};
  const double extraction_radius = 5.0;
  const size_t l_max = 24;
  const double frequency = parameter_dist(gen);
  const std::complex<double> c_2a =
      0.01 * make_with_random_values<std::complex<double>>(
                 make_not_null(&gen), make_not_null(&parameter_dist));
  const std::complex<double> c_3a =
      0.01 * make_with_random_values<std::complex<double>>(
                 make_not_null(&gen), make_not_null(&parameter_dist));
  const std::complex<double> c_2b = 3.0 * c_2a / square(frequency);
  const std::complex<double> c_3b =
      -3.0 * std::complex<double>(0.0, 1.0) * c_3a / pow<3>(frequency);

  const LinearizedBondiSachs boundary_solution{
      {c_2a, c_3a}, extraction_radius, frequency};
  const double time = 10.0 * parameter_dist(gen);
  const auto boundary_data = boundary_solution.variables(
      l_max, time, Solutions::LinearizedBondiSachs::tags{});
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          boundary_data);
  // check the serialization
  const auto serialized_and_deserialized_analytic_solution =
      serialize_and_deserialize(boundary_solution);
  const auto boundary_tuple_from_serialized =
      serialized_and_deserialized_analytic_solution.variables(
          l_max, time,
          tmpl::list<
              gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>{});
  const auto& spacetime_metric_from_serialized =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          boundary_tuple_from_serialized);
  CHECK_ITERABLE_APPROX(spacetime_metric_from_serialized, spacetime_metric);

  CartesianiSphericalJ inverse_jacobian{get<0, 0>(spacetime_metric).size()};
  boundary_solution.inverse_jacobian(make_not_null(&inverse_jacobian), l_max);
  const auto bondi_quantities =
      TestHelpers::extract_bondi_scalars_from_cartesian_metric(
          spacetime_metric, inverse_jacobian, extraction_radius);

  const auto& dt_spacetime_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      boundary_data);
  const auto dt_bondi_quantities =
      TestHelpers::extract_dt_bondi_scalars_from_cartesian_metric(
          dt_spacetime_metric, spacetime_metric, inverse_jacobian,
          extraction_radius);

  CartesianiSphericalJ dr_inverse_jacobian{get<0, 0>(spacetime_metric).size()};
  boundary_solution.dr_inverse_jacobian(make_not_null(&dr_inverse_jacobian),
                                        l_max);
  const auto& d_spacetime_metric =
      get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(boundary_data);
  const auto& dr_cartesian_coordinates =
      get<Tags::Dr<Tags::CauchyCartesianCoords>>(boundary_data);
  tnsr::aa<DataVector, 3> dr_spacetime_metric{
      get<0, 0>(spacetime_metric).size(), 0.0};
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      for (size_t i = 0; i < 3; ++i) {
        dr_spacetime_metric.get(a, b) +=
            dr_cartesian_coordinates.get(i) * d_spacetime_metric.get(i, a, b);
      }
    }
  }
  const auto dr_bondi_quantities =
      TestHelpers::extract_dr_bondi_scalars_from_cartesian_metric(
          dr_spacetime_metric, spacetime_metric, inverse_jacobian,
          dr_inverse_jacobian, extraction_radius);

  // Note for checking the bondi quantities, the best we can do is extract its
  // mode content and compare it to the formula implemented in the analytic
  // solution. The test that the formula in the analytic solution is correct, we
  // have to actually run the analytic solution in the evolution system.

  Approx bondi_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  // check that beta and its derivatives are as close to zero as we can
  // expect given the reconstruction of the quantity from the metric.
  const auto& beta = get<Tags::BondiBeta>(bondi_quantities);
  const auto& dt_beta = get<::Tags::dt<Tags::BondiBeta>>(dt_bondi_quantities);
  const auto& dr_beta = get<Tags::Dr<Tags::BondiBeta>>(dr_bondi_quantities);
  for (size_t i = 0; i < get(beta).size(); ++i) {
    CHECK(abs(get(beta).data()[i]) < 1e-10);
    CHECK(abs(get(dt_beta).data()[i]) < 1e-10);
    CHECK(abs(get(dr_beta).data()[i]) < 1e-10);
  }
  // check u and its derivatives
  const auto u_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(l_max, 1,
                                     get(get<Tags::BondiU>(bondi_quantities))),
      l_max);
  const auto dt_u_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(
          l_max, 1, get(get<::Tags::dt<Tags::BondiU>>(dt_bondi_quantities))),
      l_max);
  const auto dr_u_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(
          l_max, 1, get(get<Tags::Dr<Tags::BondiU>>(dr_bondi_quantities))),
      l_max);
  const std::complex<double> expected_time_factor =
      cos(frequency * time) +
      std::complex<double>(0.0, 1.0) * sin(frequency * time);
  const std::complex<double> expected_dt_time_factor =
      frequency * (std::complex<double>(0.0, 1.0) * cos(frequency * time) -
                   sin(frequency * time));

  const std::complex<double> expected_bondi_u_22_mode =
      sqrt(3.0) * real(expected_time_factor *
                       (0.5 * c_2a / square(extraction_radius) +
                        0.25 * c_2b / pow<4>(extraction_radius) +
                        std::complex<double>(0.0, 1.0) * frequency *
                            (c_2b / (3.0 * pow<3>(extraction_radius)))));
  const std::complex<double> expected_bondi_u_33_mode =
      sqrt(6.0) *
      real(expected_time_factor *
           (0.5 * c_3a / square(extraction_radius) -
            2.0 * square(frequency) * c_3b / (3.0 * pow<3>(extraction_radius)) +
            c_3b / pow<5>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
                (1.25 * c_3b / pow<4>(extraction_radius))));
  const std::complex<double> expected_dt_bondi_u_22_mode =
      sqrt(3.0) * real(expected_dt_time_factor *
                       (0.5 * c_2a / square(extraction_radius) +
                        0.25 * c_2b / pow<4>(extraction_radius) +
                        std::complex<double>(0.0, 1.0) * frequency *
                        (c_2b / (3.0 * pow<3>(extraction_radius)))));
  const std::complex<double> expected_dt_bondi_u_33_mode =
      sqrt(6.0) *
      real(expected_dt_time_factor *
           (0.5 * c_3a / square(extraction_radius) -
            2.0 * square(frequency) * c_3b / (3.0 * pow<3>(extraction_radius)) +
            c_3b / pow<5>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
            (1.25 * c_3b / pow<4>(extraction_radius))));
  const std::complex<double> expected_dr_bondi_u_22_mode =
      -expected_dt_bondi_u_22_mode +
      sqrt(3.0) * real(expected_time_factor *
                       (-c_2a / pow<3>(extraction_radius) -
                        c_2b / pow<5>(extraction_radius) +
                        std::complex<double>(0.0, 1.0) * frequency *
                            (-c_2b / (pow<4>(extraction_radius)))));
  const std::complex<double> expected_dr_bondi_u_33_mode =
      -expected_dt_bondi_u_33_mode +
      sqrt(6.0) *
      real(expected_time_factor *
           (-c_3a / pow<3>(extraction_radius) +
            2.0 * square(frequency) * c_3b / (pow<4>(extraction_radius)) -
            5.0 * c_3b / pow<6>(extraction_radius) -
            4.0 * std::complex<double>(0.0, 1.0) * frequency *
                (1.25 * c_3b / pow<5>(extraction_radius))));
  {
    INFO("Bondi U modes");
    check_22_and_33_modes(u_goldberg_modes, expected_bondi_u_22_mode,
                          expected_bondi_u_33_mode, l_max, bondi_approx);
  }
  {
    INFO("dr Bondi U modes");
    check_22_and_33_modes(dr_u_goldberg_modes, expected_dr_bondi_u_22_mode,
                          expected_dr_bondi_u_33_mode, l_max, bondi_approx);
  }
  {
    INFO("dt Bondi U modes");
    check_22_and_33_modes(dt_u_goldberg_modes, expected_dt_bondi_u_22_mode,
                          expected_dt_bondi_u_33_mode, l_max, bondi_approx);
  }

  // check w and its derivatives
  const auto w_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(l_max, 1,
                                     get(get<Tags::BondiW>(bondi_quantities))),
      l_max);
  const auto dt_w_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(
          l_max, 1, get(get<::Tags::dt<Tags::BondiW>>(dt_bondi_quantities))),
      l_max);
  const auto dr_w_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(
          l_max, 1, get(get<Tags::Dr<Tags::BondiW>>(dr_bondi_quantities))),
      l_max);
  const std::complex<double> expected_bondi_w_22_mode =
      real(expected_time_factor *
           (-square(frequency) * c_2b / square(extraction_radius) +
            0.5 * c_2b / pow<4>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
                (c_2b / pow<3>(extraction_radius)))) /
      sqrt(2.0);
  const std::complex<double> expected_bondi_w_33_mode =
      real(expected_time_factor *
           (2.5 * frequency * c_3b / pow<4>(extraction_radius) +
            3.0 * c_3b / pow<5>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
                (-2.0 * square(frequency) * c_3b / square(extraction_radius) -
                 4.0 * frequency * c_3b / pow<3>(extraction_radius)))) /
      sqrt(2.0);
  const std::complex<double> expected_dt_bondi_w_22_mode =
      real(expected_dt_time_factor *
           (-square(frequency) * c_2b / square(extraction_radius) +
            0.5 * c_2b / pow<4>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
            (c_2b / pow<3>(extraction_radius)))) /
      sqrt(2.0);
  const std::complex<double> expected_dt_bondi_w_33_mode =
      real(expected_dt_time_factor *
           (2.5 * frequency * c_3b / pow<4>(extraction_radius) +
            3.0 * c_3b / pow<5>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
            (-2.0 * square(frequency) * c_3b / square(extraction_radius) -
             4.0 * frequency * c_3b / pow<3>(extraction_radius)))) /
      sqrt(2.0);
  const std::complex<double> expected_dr_bondi_w_22_mode =
      -expected_dt_bondi_w_22_mode +
      real(expected_time_factor *
           (2.0 * square(frequency) * c_2b / pow<3>(extraction_radius) -
            2.0 * c_2b / pow<5>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
                (-3.0 * c_2b / pow<4>(extraction_radius)))) /
          sqrt(2.0);
  const std::complex<double> expected_dr_bondi_w_33_mode =
      -expected_dt_bondi_w_33_mode +
      real(expected_time_factor *
           (-10.0 * frequency * c_3b / pow<5>(extraction_radius) -
            15.0 * c_3b / pow<6>(extraction_radius) +
            std::complex<double>(0.0, 1.0) * frequency *
                (4.0 * square(frequency) * c_3b / pow<3>(extraction_radius) +
                 12.0 * frequency * c_3b / pow<4>(extraction_radius)))) /
          sqrt(2.0);
  {
    INFO("Bondi W modes");
    check_22_and_33_modes(w_goldberg_modes, expected_bondi_w_22_mode,
                          expected_bondi_w_33_mode, l_max, bondi_approx);
  }
  {
    INFO("dr Bondi W modes");
    check_22_and_33_modes(dr_w_goldberg_modes, expected_dr_bondi_w_22_mode,
                          expected_dr_bondi_w_33_mode, l_max, bondi_approx);
  }
  {
    INFO("dt Bondi W modes");
    check_22_and_33_modes(dt_w_goldberg_modes, expected_dt_bondi_w_22_mode,
                          expected_dt_bondi_w_33_mode, l_max, bondi_approx);
  }
  // check j and its derivatives
  const auto j_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(l_max, 1,
                                     get(get<Tags::BondiJ>(bondi_quantities))),
      l_max);
  const auto dt_j_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(
          l_max, 1, get(get<::Tags::dt<Tags::BondiJ>>(dt_bondi_quantities))),
      l_max);
  const auto dr_j_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(
          l_max, 1, get(get<Tags::Dr<Tags::BondiJ>>(dr_bondi_quantities))),
      l_max);
  const std::complex<double> expected_bondi_j_22_mode =
      sqrt(12.0) *
      real(expected_time_factor * (0.25 * c_2a / extraction_radius -
                                   c_2b / (12.0 * pow<3>(extraction_radius))));
  const std::complex<double> expected_bondi_j_33_mode =
      sqrt(60.0) * real(expected_time_factor *
                        (0.1 * c_3a / extraction_radius -
                         0.25 * c_3b / pow<4>(extraction_radius) +
                         std::complex<double>(0.0, 1.0) * frequency *
                             (-c_3b / (6.0 * pow<3>(extraction_radius)))));
  const std::complex<double> expected_dt_bondi_j_22_mode =
      sqrt(12.0) * real(expected_dt_time_factor *
                        (0.25 * c_2a / extraction_radius -
                         c_2b / (12.0 * pow<3>(extraction_radius))));
  const std::complex<double> expected_dt_bondi_j_33_mode =
      sqrt(60.0) * real(expected_dt_time_factor *
                        (0.1 * c_3a / extraction_radius -
                         0.25 * c_3b / pow<4>(extraction_radius) +
                         std::complex<double>(0.0, 1.0) * frequency *
                         (-c_3b / (6.0 * pow<3>(extraction_radius)))));
  const std::complex<double> expected_dr_bondi_j_22_mode =
      -expected_dt_bondi_j_22_mode +
      sqrt(12.0) * real(expected_time_factor *
                        (-0.25 * c_2a / square(extraction_radius) +
                         c_2b / (4.0 * pow<4>(extraction_radius))));
  const std::complex<double> expected_dr_bondi_j_33_mode =
      -expected_dt_bondi_j_33_mode +
      sqrt(60.0) * real(expected_time_factor *
                        (-0.1 * c_3a / square(extraction_radius) +
                         c_3b / pow<5>(extraction_radius) +
                         std::complex<double>(0.0, 1.0) * frequency *
                             (0.5 * c_3b / (pow<4>(extraction_radius)))));
  {
    INFO("Bondi J modes");
    check_22_and_33_modes(j_goldberg_modes, expected_bondi_j_22_mode,
                          expected_bondi_j_33_mode, l_max, bondi_approx);
  }
  {
    INFO("dr Bondi J modes");

    check_22_and_33_modes(dr_j_goldberg_modes, expected_dr_bondi_j_22_mode,
                          expected_dr_bondi_j_33_mode, l_max, bondi_approx);
  }
  {
    INFO("dt Bondi J modes");
    check_22_and_33_modes(dt_j_goldberg_modes, expected_dt_bondi_j_22_mode,
                          expected_dt_bondi_j_33_mode, l_max, bondi_approx);
  }
  const std::complex<double> expected_news_22_mode =
      real(std::complex<double>(0.0, 1.0) * pow<3>(frequency) * c_2b *
           expected_time_factor / sqrt(12.0));
  const std::complex<double> expected_news_33_mode =
      real(-pow<4>(frequency) * c_3b * expected_time_factor / sqrt(15.0));
  const auto news_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(l_max, 1,
                                     get(get<Tags::News>(boundary_data))),
      l_max);
  {
    INFO("Bondi News modes");
    check_22_and_33_modes(news_goldberg_modes, expected_news_22_mode,
                          expected_news_33_mode, l_max, bondi_approx);
  }
  Solutions::TestHelpers::check_adm_metric_quantities(
      boundary_data, spacetime_metric, dt_spacetime_metric, d_spacetime_metric);
  Solutions::TestHelpers::test_initialize_j(
      l_max, 5_st, extraction_radius, time,
      std::make_unique<
          LinearizedBondiSachs_detail::InitializeJ::LinearizedBondiSachs>(
          time, frequency, c_2a, c_2b, c_3a, c_3b),
      boundary_solution.get_clone());
}
}  // namespace
}  // namespace Cce::Solutions
