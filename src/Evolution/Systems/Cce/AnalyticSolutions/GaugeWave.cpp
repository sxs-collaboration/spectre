// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

/// \cond
GaugeWave::GaugeWave(const double extraction_radius, const double mass,
                     const double frequency, const double amplitude,
                     const double peak_time, const double duration) noexcept
    : SphericalMetricData{extraction_radius},
      mass_{mass},
      frequency_{frequency},
      amplitude_{amplitude},
      peak_time_{peak_time},
      duration_{duration} {}

std::unique_ptr<WorldtubeData> GaugeWave::get_clone() const
    noexcept {
  return std::make_unique<GaugeWave>(*this);
}

double GaugeWave::coordinate_wave_function(const double time) const noexcept {
  const auto retarded_time = time - extraction_radius_;
  return amplitude_ * sin(frequency_ * retarded_time) *
         exp(-square(retarded_time - peak_time_) / square(duration_));
}

double GaugeWave::du_coordinate_wave_function(const double time) const
    noexcept {
  const auto retarded_time = time - extraction_radius_;
  return amplitude_ *
         (-2.0 * (retarded_time - peak_time_) / square(duration_) *
              sin(frequency_ * retarded_time) +
          frequency_ * cos(frequency_ * retarded_time)) *
         exp(-square(retarded_time - peak_time_) / square(duration_));
}

double GaugeWave::du_du_coordinate_wave_function(const double time) const
    noexcept {
  const auto retarded_time = time - extraction_radius_;
  return amplitude_ *
         (-4.0 * square(duration_) * (retarded_time - peak_time_) * frequency_ *
              cos(frequency_ * retarded_time) +
          (-2.0 * square(duration_) + 4.0 * square(retarded_time - peak_time_) -
           pow<4>(duration_) * square(frequency_)) *
              sin(frequency_ * retarded_time)) /
         pow<4>(duration_) *
         exp(-square(retarded_time - peak_time_) / square(duration_));
}

void GaugeWave::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t /*l_max*/, const double time) const noexcept {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  get<0, 0>(*spherical_metric) = -(extraction_radius_ - 2.0 * mass_) *
                                 square(extraction_radius_ + du_wave_f) /
                                 pow<3>(extraction_radius_);
  get<0, 1>(*spherical_metric) =
      (extraction_radius_ + du_wave_f) *
      (2.0 * mass_ * square(extraction_radius_) +
       (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_wave_f + wave_f)) /
      pow<4>(extraction_radius_);
  get<0, 2>(*spherical_metric) = 0.0;
  get<0, 3>(*spherical_metric) = 0.0;

  get<1, 1>(*spherical_metric) =
      (square(extraction_radius_) - extraction_radius_ * du_wave_f - wave_f) *
      (pow<3>(extraction_radius_) + 2.0 * mass_ * square(extraction_radius_) +
       (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_wave_f + wave_f)) /
      pow<5>(extraction_radius_);
  get<1, 2>(*spherical_metric) = 0.0;
  get<1, 3>(*spherical_metric) = 0.0;
  get<2, 2>(*spherical_metric) = square(extraction_radius_);
  get<2, 3>(*spherical_metric) = 0.0;
  get<3, 3>(*spherical_metric) = square(extraction_radius_);
}

void GaugeWave::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  // for simpler expressions, we take advantage of the F derivatives evaluated
  // in the dt function (because the F function depends only on retarded time
  // t - r)
  dt_spherical_metric(dr_spherical_metric, l_max, time);

  get<0, 0>(*dr_spherical_metric) =
      -get<0, 0>(*dr_spherical_metric) +
      2.0 / pow<4>(extraction_radius_) * (extraction_radius_ + du_wave_f) *
          (-mass_ * extraction_radius_ +
           (extraction_radius_ - 3.0 * mass_) * du_wave_f);

  get<0, 1>(*dr_spherical_metric) =
      -get<0, 1>(*dr_spherical_metric) -
      (2.0 * mass_ * pow<3>(extraction_radius_) +
       2.0 * extraction_radius_ * wave_f * (extraction_radius_ - 3.0 * mass_) +
       du_wave_f * (pow<3>(extraction_radius_) +
                    wave_f * (3.0 * extraction_radius_ - 8.0 * mass_)) +
       2.0 * extraction_radius_ * square(du_wave_f) *
           (extraction_radius_ - 3.0 * mass_)) /
          pow<5>(extraction_radius_);

  get<0, 2>(*dr_spherical_metric) = 0.0;
  get<0, 3>(*dr_spherical_metric) = 0.0;

  get<1, 1>(*dr_spherical_metric) =
      -get<1, 1>(*dr_spherical_metric) +
      2.0 *
          (-mass_ * pow<4>(extraction_radius_) +
           square(wave_f) * (2.0 * extraction_radius_ - 5.0 * mass_) +
           du_wave_f * square(extraction_radius_) *
               (4.0 * mass_ * extraction_radius_ +
                du_wave_f * (extraction_radius_ - 3.0 * mass_)) +
           wave_f * extraction_radius_ *
               (6.0 * mass_ * extraction_radius_ +
                du_wave_f * (3.0 * extraction_radius_ - 8.0 * mass_))) /
          pow<6>(extraction_radius_);

  get<1, 2>(*dr_spherical_metric) = 0.0;
  get<1, 3>(*dr_spherical_metric) = 0.0;
  get<2, 2>(*dr_spherical_metric) = 2.0 * extraction_radius_;
  get<2, 3>(*dr_spherical_metric) = 0.0;
  get<3, 3>(*dr_spherical_metric) = 2.0 * extraction_radius_;
}

void GaugeWave::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    const size_t /*l_max*/, const double time) const noexcept {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  const auto du_du_wave_f = du_du_coordinate_wave_function(time);

  get<0, 0>(*dt_spherical_metric) =
      -2.0 * du_du_wave_f / pow<3>(extraction_radius_) *
      (extraction_radius_ - 2.0 * mass_) * (extraction_radius_ + du_wave_f);
  get<0, 1>(*dt_spherical_metric) =
      (du_du_wave_f * (2.0 * mass_ * square(extraction_radius_) +
                       (extraction_radius_ - 2.0 * mass_) *
                           (extraction_radius_ * du_wave_f + wave_f)) +
       (extraction_radius_ + du_wave_f) * (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_du_wave_f + du_wave_f)) /
      pow<4>(extraction_radius_);

  get<0, 2>(*dt_spherical_metric) = 0.0;
  get<0, 3>(*dt_spherical_metric) = 0.0;

  get<1, 1>(*dt_spherical_metric) =
      (-(extraction_radius_ * du_du_wave_f + du_wave_f) *
           (pow<3>(extraction_radius_) +
            2.0 * mass_ * square(extraction_radius_) +
            (extraction_radius_ - 2.0 * mass_) *
                (extraction_radius_ * du_wave_f + wave_f)) +
       (square(extraction_radius_) - extraction_radius_ * du_wave_f - wave_f) *
           (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_du_wave_f + du_wave_f)) /
      pow<5>(extraction_radius_);

  get<1, 2>(*dt_spherical_metric) = 0.0;
  get<1, 3>(*dt_spherical_metric) = 0.0;
  get<2, 2>(*dt_spherical_metric) = 0.0;
  get<2, 3>(*dt_spherical_metric) = 0.0;
  get<3, 3>(*dt_spherical_metric) = 0.0;
}

void GaugeWave::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t /*l_max*/, const double /*time*/,
    tmpl::type_<Tags::News> /*meta*/) const noexcept {
  get(*news).data() = 0.0;
}

void GaugeWave::pup(PUP::er& p) noexcept {
  SphericalMetricData::pup(p);
  p | mass_;
  p | frequency_;
  p | amplitude_;
  p | peak_time_;
  p | duration_;
}

PUP::able::PUP_ID GaugeWave::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::Solutions
