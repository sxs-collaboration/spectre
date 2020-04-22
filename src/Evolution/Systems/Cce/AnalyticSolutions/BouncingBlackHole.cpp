// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Solutions {

/// \cond

BouncingBlackHole::BouncingBlackHole(const double amplitude,
                                     const double extraction_radius,
                                     const double mass,
                                     const double period) noexcept
    : WorldtubeData(extraction_radius),
      amplitude_{amplitude},
      mass_{mass},
      frequency_{2.0 * M_PI / period} {}

std::unique_ptr<WorldtubeData> BouncingBlackHole::get_clone() const noexcept {
  return std::make_unique<BouncingBlackHole>(*this);
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial,
                                          DataVector>> /*meta*/) const
    noexcept {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(l_max, time);

  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));

  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);

  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));

  const DataVector inverse_r_cubed = 1.0 / pow<3>(r);

  get<0, 0>(*spacetime_metric) =
      -1.0 + 2.0 * mass_ / r +
      square(dt_adjusted_x_coordinate) *
          (1.0 +
           2.0 * mass_ * square(adjusted_x_coordinate) * inverse_r_cubed) +
      4.0 * mass_ * dt_adjusted_x_coordinate * adjusted_x_coordinate /
          square(r);

  get<0, 1>(*spacetime_metric) =
      dt_adjusted_x_coordinate +
      2.0 * mass_ * adjusted_x_coordinate / square(r) +
      2.0 * mass_ * dt_adjusted_x_coordinate * square(adjusted_x_coordinate) *
          inverse_r_cubed;

  for (size_t i = 1; i < 3; ++i) {
    spacetime_metric->get(0, i + 1) =
        2.0 * mass_ * (dt_adjusted_x_coordinate * adjusted_x_coordinate + r) *
        cartesian_coordinates.get(i) * inverse_r_cubed;

    spacetime_metric->get(i + 1, i + 1) =
        1.0 +
        2.0 * mass_ * square(cartesian_coordinates.get(i)) * inverse_r_cubed;

    spacetime_metric->get(1, i + 1) = 2.0 * mass_ * adjusted_x_coordinate *
                                      cartesian_coordinates.get(i) *
                                      inverse_r_cubed;
  }

  get<1, 1>(*spacetime_metric) =
      1.0 + 2.0 * mass_ * square(adjusted_x_coordinate) * inverse_r_cubed;

  get<2, 3>(*spacetime_metric) = 2.0 * mass_ * get<1>(cartesian_coordinates) *
                                 get<2>(cartesian_coordinates) *
                                 inverse_r_cubed;
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<::Tags::dt<
        gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>> /*meta*/)
    const noexcept {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(l_max, time);

  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));
  const double dt_dt_adjusted_x_coordinate =
      4.0 * amplitude_ * square(frequency_) *
      (3.0 * square(cos(frequency_ * time)) * square(sin(frequency_ * time)) -
       pow<4>(sin(frequency_ * time)));

  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);
  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));
  const DataVector inverse_r_cubed = 1.0 / pow<3>(r);

  get<0, 0>(*dt_spacetime_metric) =
      2.0 * dt_dt_adjusted_x_coordinate * dt_adjusted_x_coordinate *
          (2.0 * mass_ * square(adjusted_x_coordinate) * inverse_r_cubed +
           1.0) +
      2.0 * dt_dt_adjusted_x_coordinate * 2.0 * mass_ * adjusted_x_coordinate /
          square(r) +
      pow<3>(dt_adjusted_x_coordinate) *
          (4.0 * mass_ * adjusted_x_coordinate * inverse_r_cubed -
           6.0 * mass_ * pow<3>(adjusted_x_coordinate) / pow<5>(r)) +
      2.0 * square(dt_adjusted_x_coordinate) *
          (2.0 * mass_ / square(r) -
           4.0 * mass_ * square(adjusted_x_coordinate) / pow<4>(r)) -
      dt_adjusted_x_coordinate * 2.0 * mass_ * adjusted_x_coordinate *
          inverse_r_cubed;

  get<0, 1>(*dt_spacetime_metric) =
      dt_dt_adjusted_x_coordinate +
      2.0 * mass_ / square(r) *
          (dt_adjusted_x_coordinate -
           2.0 * square(adjusted_x_coordinate) * dt_adjusted_x_coordinate /
               square(r) +
           square(adjusted_x_coordinate) * dt_dt_adjusted_x_coordinate / r +
           2.0 * adjusted_x_coordinate * square(dt_adjusted_x_coordinate) / r -
           3.0 * pow<3>(adjusted_x_coordinate) *
               square(dt_adjusted_x_coordinate) * inverse_r_cubed);

  for (size_t i = 1; i < 3; ++i) {
    dt_spacetime_metric->get(0, i + 1) =
        2.0 * mass_ * cartesian_coordinates.get(i) * inverse_r_cubed *
        (dt_dt_adjusted_x_coordinate * adjusted_x_coordinate +
         square(dt_adjusted_x_coordinate) -
         2.0 * adjusted_x_coordinate * dt_adjusted_x_coordinate / r -
         3.0 * square(adjusted_x_coordinate) *
             square(dt_adjusted_x_coordinate) / square(r));

    dt_spacetime_metric->get(1, i + 1) =
        2.0 * mass_ * dt_adjusted_x_coordinate * cartesian_coordinates.get(i) *
        inverse_r_cubed *
        (1.0 - 3.0 * square(adjusted_x_coordinate) / square(r));

    dt_spacetime_metric->get(i + 1, i + 1) =
        -6.0 * mass_ * adjusted_x_coordinate *
        square(cartesian_coordinates.get(i)) * dt_adjusted_x_coordinate /
        pow<5>(r);
  }
  get<1, 1>(*dt_spacetime_metric) =
      2.0 * mass_ * adjusted_x_coordinate * dt_adjusted_x_coordinate *
      inverse_r_cubed * (2.0 - 3.0 * square(adjusted_x_coordinate) / square(r));

  get<2, 3>(*dt_spacetime_metric) =
      -6.0 * mass_ * adjusted_x_coordinate * get<1>(cartesian_coordinates) *
      get<2>(cartesian_coordinates) * dt_adjusted_x_coordinate / pow<5>(r);
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>> /*meta*/)
    const noexcept {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(l_max, time);

  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));

  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);
  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));
  const DataVector inverse_r_cubed = 1.0 / pow<3>(r);

  get<0, 0, 0>(*d_spacetime_metric) =
      2.0 * mass_ / square(r) *
      (2.0 * dt_adjusted_x_coordinate +
       (2.0 * square(dt_adjusted_x_coordinate) * adjusted_x_coordinate -
        adjusted_x_coordinate) /
           r -
       4.0 * dt_adjusted_x_coordinate * square(adjusted_x_coordinate) /
           square(r) -
       3.0 * square(dt_adjusted_x_coordinate * adjusted_x_coordinate) *
           adjusted_x_coordinate * inverse_r_cubed);

  get<0, 0, 1>(*d_spacetime_metric) =
      2.0 * mass_ / square(r) *
      (1.0 + 2.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r -
       2.0 * square(adjusted_x_coordinate) / square(r) -
       3.0 * dt_adjusted_x_coordinate * pow<3>(adjusted_x_coordinate) *
           inverse_r_cubed);

  get<0, 1, 1>(*d_spacetime_metric) =
      2.0 * mass_ *
      (2.0 * adjusted_x_coordinate -
       3.0 * pow<3>(adjusted_x_coordinate) / square(r)) *
      inverse_r_cubed;

  for (size_t i = 1; i < 3; ++i) {
    d_spacetime_metric->get(i, 0, 0) =
        -2.0 * mass_ * cartesian_coordinates.get(i) * inverse_r_cubed *
        (1.0 + 4.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r +
         3.0 * square(dt_adjusted_x_coordinate * adjusted_x_coordinate) /
             square(r));

    d_spacetime_metric->get(i, 0, 1) =
        -2.0 * mass_ * adjusted_x_coordinate * cartesian_coordinates.get(i) /
        pow<4>(r) *
        (2.0 + 3.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r);

    d_spacetime_metric->get(i, 1, 1) = -6.0 * mass_ *
                                       square(adjusted_x_coordinate) *
                                       cartesian_coordinates.get(i) / pow<5>(r);

    d_spacetime_metric->get(0, 1, i + 1) =
        2.0 * mass_ * cartesian_coordinates.get(i) *
        (1.0 - 3.0 * square(adjusted_x_coordinate) / square(r)) *
        inverse_r_cubed;

    d_spacetime_metric->get(0, 0, i + 1) =
        2.0 * mass_ * cartesian_coordinates.get(i) * inverse_r_cubed *
        (dt_adjusted_x_coordinate - 2.0 * adjusted_x_coordinate / r -
         3.0 * dt_adjusted_x_coordinate * square(adjusted_x_coordinate) /
             square(r));

    for (size_t j = 1; j < 3; ++j) {
      if (i == j) {
        d_spacetime_metric->get(i, 0, j + 1) =
            2.0 * mass_ / square(r) *
            (1.0 + dt_adjusted_x_coordinate * adjusted_x_coordinate / r -
             2.0 * square(cartesian_coordinates.get(i)) / square(r) -
             3.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate *
                 square(cartesian_coordinates.get(i)) * inverse_r_cubed);

        d_spacetime_metric->get(i, 1, j + 1) =
            2.0 * mass_ * adjusted_x_coordinate *
            (1.0 - 3.0 * square(cartesian_coordinates.get(i)) / square(r)) *
            inverse_r_cubed;

        d_spacetime_metric->get(0, i + 1, j + 1) =
            -6.0 * mass_ * square(cartesian_coordinates.get(i)) *
            adjusted_x_coordinate / pow<5>(r);
      } else {
        d_spacetime_metric->get(i, 0, j + 1) =
            -2.0 * mass_ * cartesian_coordinates.get(i) *
            cartesian_coordinates.get(j) / pow<4>(r) *
            (3.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r + 2.0);

        d_spacetime_metric->get(i, 1, j + 1) =
            -6.0 * mass_ * adjusted_x_coordinate *
            cartesian_coordinates.get(i) * cartesian_coordinates.get(j) /
            pow<5>(r);

        d_spacetime_metric->get(0, i + 1, j + 1) =
            -6.0 * mass_ * cartesian_coordinates.get(i) *
            cartesian_coordinates.get(j) * adjusted_x_coordinate / pow<5>(r);
      }
      for (size_t k = 1; k < 3; ++k) {
        if (i == j and j == k) {
          d_spacetime_metric->get(k, i + 1, j + 1) =
              2.0 * mass_ *
              (2.0 * cartesian_coordinates.get(i) -
               3.0 * pow<3>(cartesian_coordinates.get(i)) / square(r)) *
              inverse_r_cubed;

        } else if (i == j) {
          d_spacetime_metric->get(i, j + 1, k + 1) =
              2.0 * mass_ *
              (cartesian_coordinates.get(k) -
               3.0 * square(cartesian_coordinates.get(i)) *
                   cartesian_coordinates.get(k) / square(r)) *
              inverse_r_cubed;
          d_spacetime_metric->get(k, i + 1, j + 1) =
              -6.0 * mass_ * square(cartesian_coordinates.get(i)) *
              cartesian_coordinates.get(k) / pow<5>(r);
        }
      }
    }
  }
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t /*output_l_max*/, const double /*time*/,
    tmpl::type_<Tags::News> /*meta*/) const noexcept {
  get(*news).data() = 0.0;
}

void BouncingBlackHole::pup(PUP::er& p) noexcept {
  WorldtubeData::pup(p);
  p | amplitude_;
  p | mass_;
  p | frequency_;
}

PUP::able::PUP_ID BouncingBlackHole::my_PUP_ID = 0;
/// \endcond
}  // namespace Solutions
}  // namespace Cce
