// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"

#include <complex>
#include <cstddef>
#include <memory>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

/// \cond
TeukolskyWave::TeukolskyWave(const double extraction_radius,
                             const double amplitude,
                             const double duration) noexcept
    : SphericalMetricData{extraction_radius},
      amplitude_{amplitude},
      duration_{duration} {}

std::unique_ptr<WorldtubeData> TeukolskyWave::get_clone() const noexcept {
  return std::make_unique<TeukolskyWave>(*this);
}

double TeukolskyWave::pulse_profile_coefficient_a(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return 3.0 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
         (pow<4>(duration_) * pow<5>(extraction_radius_)) *
         (3.0 * pow<4>(duration_) +
          4.0 * square(extraction_radius_) * square(retarded_time) -
          2.0 * square(duration_) * extraction_radius_ *
              (extraction_radius_ + 3.0 * retarded_time));
}

double TeukolskyWave::pulse_profile_coefficient_b(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return 2.0 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
         (pow<6>(duration_) * pow<5>(extraction_radius_)) *
         (-3.0 * pow<6>(duration_) +
          4.0 * pow<3>(extraction_radius_) * pow<3>(retarded_time) -
          6.0 * square(duration_) * pow<2>(extraction_radius_) * retarded_time *
              (extraction_radius_ + retarded_time) +
          3.0 * pow<4>(duration_) * extraction_radius_ *
              (extraction_radius_ + 2.0 * retarded_time));
}

double TeukolskyWave::pulse_profile_coefficient_c(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return 0.25 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
         (pow<8>(duration_) * pow<5>(extraction_radius_)) *
         (21.0 * pow<8>(duration_) +
          16.0 * pow<4>(extraction_radius_) * pow<4>(retarded_time) -
          16.0 * square(duration_) * pow<3>(extraction_radius_) *
              square(retarded_time) *
              (3.0 * extraction_radius_ + retarded_time) -
          6.0 * pow<6>(duration_) * extraction_radius_ *
              (3.0 * extraction_radius_ + 7.0 * retarded_time) +
          12.0 * pow<4>(duration_) * square(extraction_radius_) *
              (square(extraction_radius_) +
               2.0 * extraction_radius_ * retarded_time +
               3.0 * square(retarded_time)));
}

double TeukolskyWave::dr_pulse_profile_coefficient_a(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return -dt_pulse_profile_coefficient_a(time) -
         9.0 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
             (pow<4>(duration_) * pow<6>(extraction_radius_)) *
             (5.0 * pow<4>(duration_) +
              4.0 * square(extraction_radius_) * square(retarded_time) -
              2.0 * square(duration_) * extraction_radius_ *
                  (extraction_radius_ + 4.0 * retarded_time));
}

double TeukolskyWave::dr_pulse_profile_coefficient_b(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return -dt_pulse_profile_coefficient_b(time) +
         2.0 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
             (pow<6>(duration_ * extraction_radius_)) *
             (15.0 * pow<6>(duration_) -
              8.0 * pow<3>(extraction_radius_) * pow<3>(retarded_time) +
              6.0 * square(duration_) * square(extraction_radius_) *
                  retarded_time *
                  (2.0 * extraction_radius_ + 3.0 * retarded_time) -
              3.0 * pow<4>(duration_) * extraction_radius_ *
                  (3.0 * extraction_radius_ + 8.0 * retarded_time));
}

double TeukolskyWave::dr_pulse_profile_coefficient_c(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return -dt_pulse_profile_coefficient_c(time) -
         0.25 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
             (pow<8>(duration_) * pow<6>(extraction_radius_)) *
             (105.0 * pow<8>(duration_) +
              16.0 * pow<4>(extraction_radius_ * retarded_time) -
              16.0 * square(duration_) * pow<3>(extraction_radius_) *
                  square(retarded_time) *
                  (3.0 * extraction_radius_ + 2.0 * retarded_time) -
              6.0 * pow<6>(duration_) * extraction_radius_ *
                  (9.0 * extraction_radius_ + 28.0 * retarded_time) +
              12.0 * pow<4>(duration_) * square(extraction_radius_) *
                  (square(extraction_radius_) +
                   4.0 * extraction_radius_ * retarded_time +
                   9.0 * square(retarded_time)));
}

double TeukolskyWave::dt_pulse_profile_coefficient_a(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return -2.0 * retarded_time / square(duration_) *
             pulse_profile_coefficient_a(time) +
         3.0 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
             (pow<4>(duration_) * pow<5>(extraction_radius_)) *
             (8.0 * square(extraction_radius_) * retarded_time -
              6.0 * square(duration_) * extraction_radius_);
}

double TeukolskyWave::dt_pulse_profile_coefficient_b(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return -2.0 * retarded_time / square(duration_) *
             pulse_profile_coefficient_b(time) +
         2.0 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
             (pow<6>(duration_) * pow<5>(extraction_radius_)) *
             (12.0 * pow<3>(extraction_radius_) * square(retarded_time) -
              6.0 * square(duration_) * square(extraction_radius_) *
                  (extraction_radius_ + 2.0 * retarded_time) +
              6.0 * pow<4>(duration_) * extraction_radius_);
}

double TeukolskyWave::dt_pulse_profile_coefficient_c(const double time) const
    noexcept {
  const double retarded_time = time - extraction_radius_;
  return -2.0 * retarded_time / square(duration_) *
             pulse_profile_coefficient_c(time) +
         0.25 * amplitude_ * exp(-square(retarded_time) / square(duration_)) /
             (pow<8>(duration_) * pow<5>(extraction_radius_)) *
             (64.0 * pow<4>(extraction_radius_) * pow<3>(retarded_time) -
              16.0 * square(duration_) * pow<3>(extraction_radius_) *
                  retarded_time *
                  (6.0 * extraction_radius_ + 3.0 * retarded_time) -
              42.0 * pow<6>(duration_) * extraction_radius_ +
              12.0 * pow<4>(duration_) * square(extraction_radius_) *
                  (2.0 * extraction_radius_ + 6.0 * retarded_time));
}

DataVector TeukolskyWave::sin_theta(const size_t l_max) noexcept {
  DataVector sin_theta_result{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    sin_theta_result[collocation_point.offset] = sin(collocation_point.theta);
  }
  return sin_theta_result;
}

DataVector TeukolskyWave::cos_theta(const size_t l_max) noexcept {
  DataVector cos_theta_result{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    cos_theta_result[collocation_point.offset] = cos(collocation_point.theta);
  }
  return cos_theta_result;
}

void TeukolskyWave::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t l_max, const double time) const noexcept {
  const auto coefficient_a = pulse_profile_coefficient_a(time);
  const auto coefficient_b = pulse_profile_coefficient_b(time);
  const auto coefficient_c = pulse_profile_coefficient_c(time);

  const auto local_sin_theta = sin_theta(l_max);
  const auto local_cos_theta = cos_theta(l_max);

  get<0, 0>(*spherical_metric) = -1.0;
  get<0, 1>(*spherical_metric) = 0.0;
  get<0, 2>(*spherical_metric) = 0.0;
  get<0, 3>(*spherical_metric) = 0.0;

  get<1, 3>(*spherical_metric) = 0.0;
  get<2, 3>(*spherical_metric) = 0.0;
  get<1, 1>(*spherical_metric) =
      1.0 + coefficient_a * (2.0 - 3.0 * square(local_sin_theta));
  get<1, 2>(*spherical_metric) = -3.0 * local_sin_theta * local_cos_theta *
                                 extraction_radius_ * coefficient_b;
  get<2, 2>(*spherical_metric) =
      (1.0 + 3.0 * square(local_sin_theta) * coefficient_c - coefficient_a) *
      square(extraction_radius_);
  // note: omitted sin^2(theta) to optimize the precision in common case where
  // it would cancel with derivative or Jacobian factors.
  get<3, 3>(*spherical_metric) =
      (1.0 - 3.0 * square(local_sin_theta) * coefficient_c +
       (3.0 * square(local_sin_theta) - 1.0) * coefficient_a) *
      square(extraction_radius_);
}

void TeukolskyWave::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  const auto coefficient_a = pulse_profile_coefficient_a(time);
  const auto coefficient_b = pulse_profile_coefficient_b(time);
  const auto coefficient_c = pulse_profile_coefficient_c(time);

  const auto dr_coefficient_a = dr_pulse_profile_coefficient_a(time);
  const auto dr_coefficient_b = dr_pulse_profile_coefficient_b(time);
  const auto dr_coefficient_c = dr_pulse_profile_coefficient_c(time);

  const auto local_sin_theta = sin_theta(l_max);
  const auto local_cos_theta = cos_theta(l_max);

  for (size_t a = 0; a < 4; ++a) {
    dr_spherical_metric->get(0, a) = 0.0;
  }

  get<1, 1>(*dr_spherical_metric) =
      dr_coefficient_a * (2.0 - 3.0 * square(local_sin_theta));
  get<1, 3>(*dr_spherical_metric) = 0.0;
  get<2, 3>(*dr_spherical_metric) = 0.0;

  get<1, 2>(*dr_spherical_metric) =
      -3.0 * local_sin_theta * local_cos_theta *
      (coefficient_b + extraction_radius_ * dr_coefficient_b);
  get<2, 2>(*dr_spherical_metric) =
      (2.0 *
           (1.0 - coefficient_a +
            3.0 * square(local_sin_theta) * coefficient_c) *
           extraction_radius_ +
       (-dr_coefficient_a + 3.0 * square(local_sin_theta) * dr_coefficient_c) *
           square(extraction_radius_));
  // note: omitted sin^2(theta) to optimize the precision in common case where
  // it would cancel with derivative or Jacobian factors.
  get<3, 3>(*dr_spherical_metric) =
      (2.0 *
           (1.0 + (3.0 * square(local_sin_theta) - 1.0) * coefficient_a -
            3.0 * square(local_sin_theta) * coefficient_c) *
           extraction_radius_ +
       ((3.0 * square(local_sin_theta) - 1.0) * dr_coefficient_a -
        3.0 * square(local_sin_theta) * dr_coefficient_c) *
           square(extraction_radius_));
}

void TeukolskyWave::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  const auto dt_coefficient_a = dt_pulse_profile_coefficient_a(time);
  const auto dt_coefficient_b = dt_pulse_profile_coefficient_b(time);
  const auto dt_coefficient_c = dt_pulse_profile_coefficient_c(time);

  const auto local_sin_theta = sin_theta(l_max);
  const auto local_cos_theta = cos_theta(l_max);

  for (size_t a = 0; a < 4; ++a) {
    dt_spherical_metric->get(0, a) = 0.0;
  }

  get<1, 1>(*dt_spherical_metric) =
      dt_coefficient_a * (2.0 - 3.0 * square(local_sin_theta));
  get<1, 3>(*dt_spherical_metric) = 0.0;
  get<2, 3>(*dt_spherical_metric) = 0.0;
  get<1, 2>(*dt_spherical_metric) = -3.0 * local_sin_theta * local_cos_theta *
                                    extraction_radius_ * dt_coefficient_b;
  get<2, 2>(*dt_spherical_metric) =
      ((-dt_coefficient_a + 3.0 * square(local_sin_theta) * dt_coefficient_c) *
       square(extraction_radius_));
  // note: omitted sin^2(theta) to optimize the precision in common case where
  // it would cancel with derivative or Jacobian factors.
  get<3, 3>(*dt_spherical_metric) =
      (((3.0 * square(local_sin_theta) - 1.0) * dt_coefficient_a -
        3.0 * square(local_sin_theta) * dt_coefficient_c) *
       square(extraction_radius_));
}

void TeukolskyWave::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t l_max, const double time,
    tmpl::type_<Tags::News> /*meta*/) const noexcept {
  const auto local_sin_theta = sin_theta(l_max);
  get(*news).data() =
      -std::complex<double>{6.0, 0.0} * square(local_sin_theta) * amplitude_ *
      exp(-square(time - extraction_radius_) / square(duration_)) *
      (time - extraction_radius_) / pow<10>(duration_) *
      (15.0 * pow<4>(duration_) -
       20.0 * square(duration_ * (time - extraction_radius_)) +
       4.0 * pow<4>(time - extraction_radius_));
}

void TeukolskyWave::pup(PUP::er& p) noexcept {
  SphericalMetricData::pup(p);
  p | amplitude_;
  p | duration_;
}

PUP::able::PUP_ID TeukolskyWave::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::Solutions
