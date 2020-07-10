// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"

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
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

/// \cond
LinearizedBondiSachs::LinearizedBondiSachs(
    const std::vector<std::complex<double>>& mode_constants,
    const double extraction_radius, const double frequency) noexcept
    : SphericalMetricData{extraction_radius}, frequency_{frequency} {
  auto mode_constant_copy = mode_constants;
  mode_constant_copy.resize(2, std::complex<double>(0.0, 0.0));
  c_2a_ = gsl::at(mode_constant_copy, 0_st);
  c_3a_ = gsl::at(mode_constant_copy, 1_st);
  c_2b_ = 3.0 * c_2a_ / square(frequency_);
  c_3b_ = std::complex<double>(0.0, -3.0) * c_3a_ / pow<3>(frequency_);
}

std::unique_ptr<WorldtubeData> LinearizedBondiSachs::get_clone()
    const noexcept {
  return std::make_unique<LinearizedBondiSachs>(*this);
}

template <int Spin>
void LinearizedBondiSachs::assign_components_from_l_factors(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> bondi_quantity,
    const std::complex<double>& l_2_factor,
    const std::complex<double>& l_3_factor, const size_t l_max,
    const double time) const noexcept {
  const std::complex<double> time_factor =
      cos(frequency_ * time) -
      std::complex<double>(0.0, 1.0) * sin(frequency_ * time);

  // mode constants for the computation of the linearized solution.
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_22{Spin, 2_st, 2};
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_2m2{Spin, 2_st, -2};
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_33{Spin, 3_st, 3};
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_3m3{Spin, 3_st, -3};

  const auto& collocation_metadata =
      Spectral::Swsh::cached_collocation_metadata<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation_metadata) {
    // assign to collocation points
    const std::complex<double> z_22_factor =
        (y_22.evaluate(collocation_point.theta, collocation_point.phi) +
         y_2m2.evaluate(collocation_point.theta, collocation_point.phi));
    const std::complex<double> z_33_factor =
        (y_33.evaluate(collocation_point.theta, collocation_point.phi) -
         y_3m3.evaluate(collocation_point.theta, collocation_point.phi));
    bondi_quantity->data()[collocation_point.offset] =
        z_22_factor * real(time_factor * l_2_factor) +
        z_33_factor * real(time_factor * l_3_factor);
  }
}

template <int Spin>
void LinearizedBondiSachs::assign_du_components_from_l_factors(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        du_bondi_quantity,
    const std::complex<double>& l_2_factor,
    const std::complex<double>& l_3_factor, const size_t l_max,
    const double time) const noexcept {
  const std::complex<double> time_factor =
      frequency_ * (std::complex<double>(0.0, 1.0) * cos(frequency_ * time) +
                    sin(frequency_ * time));

  // mode constants for the computation of the linearized solution.
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_22{Spin, 2_st, 2};
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_2m2{Spin, 2_st, -2};
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_33{Spin, 3_st, 3};
  Spectral::Swsh::SpinWeightedSphericalHarmonic y_3m3{Spin, 3_st, -3};

  const auto& collocation_metadata =
      Spectral::Swsh::cached_collocation_metadata<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation_metadata) {
    // assign to collocation points
    const std::complex<double> z_22_factor =
        (y_22.evaluate(collocation_point.theta, collocation_point.phi) +
         y_2m2.evaluate(collocation_point.theta, collocation_point.phi));
    const std::complex<double> z_33_factor =
        (y_33.evaluate(collocation_point.theta, collocation_point.phi) -
         y_3m3.evaluate(collocation_point.theta, collocation_point.phi));
    du_bondi_quantity->data()[collocation_point.offset] =
        z_22_factor * real(time_factor * l_2_factor) +
        z_33_factor * real(time_factor * l_3_factor);
  }
}

void LinearizedBondiSachs::linearized_bondi_j(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> bondi_j,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> j_2_coefficient =
      sqrt(0.75) *
      (c_2a_ / extraction_radius_ - c_2b_ / (3.0 * pow<3>(extraction_radius_)));

  const std::complex<double> j_3_coefficient =
      sqrt(30.0) * (0.1 * c_3a_ / extraction_radius_ -
                    0.25 * c_3b_ / pow<4>(extraction_radius_) -
                    std::complex<double>(0.0, 2.0) * frequency_ *
                        (c_3b_ / (12.0 * pow<3>(extraction_radius_))));

  assign_components_from_l_factors(bondi_j, j_2_coefficient, j_3_coefficient,
                                   l_max, time);
}

void LinearizedBondiSachs::linearized_bondi_u(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> bondi_u,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> u_2_coefficient =
      sqrt(3.0) * (0.5 * c_2a_ / square(extraction_radius_) +
                   0.25 * c_2b_ / pow<4>(extraction_radius_) +
                   std::complex<double>(0.0, 1.0) * frequency_ *
                       (c_2b_ / (3.0 * pow<3>(extraction_radius_))));

  const std::complex<double> u_3_coefficient =
      sqrt(6.0) *
      (0.5 * c_3a_ / square(extraction_radius_) -
       2.0 * square(frequency_) * c_3b_ / (3.0 * pow<3>(extraction_radius_)) +
       c_3b_ / pow<5>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (1.25 * c_3b_ / pow<4>(extraction_radius_)));

  assign_components_from_l_factors(bondi_u, u_2_coefficient, u_3_coefficient,
                                   l_max, time);
}

void LinearizedBondiSachs::linearized_bondi_w(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> bondi_w,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> w_2_coefficient =
      (-square(frequency_) * c_2b_ / square(extraction_radius_) +
       0.5 * c_2b_ / pow<4>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (c_2b_ / pow<3>(extraction_radius_))) /
      sqrt(2.0);

  const std::complex<double> w_3_coefficient =
      (2.5 * frequency_ * c_3b_ / pow<4>(extraction_radius_) +
       3.0 * c_3b_ / pow<5>(extraction_radius_) -
       std::complex<double>(0.0, 2.0) * frequency_ *
           (square(frequency_) * c_3b_ / square(extraction_radius_) +
            2.0 * frequency_ * c_3b_ / pow<3>(extraction_radius_))) /
      sqrt(2.0);

  assign_components_from_l_factors(bondi_w, w_2_coefficient, w_3_coefficient,
                                   l_max, time);
}

void LinearizedBondiSachs::linearized_dr_bondi_j(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dr_bondi_j,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> dr_j_2_coefficient =
      -sqrt(0.75) *
      (c_2a_ / square(extraction_radius_) - c_2b_ / pow<4>(extraction_radius_));

  const std::complex<double> dr_j_3_coefficient =
      sqrt(30.0) * (-0.1 * c_3a_ / square(extraction_radius_) +
                    c_3b_ / pow<5>(extraction_radius_) +
                    std::complex<double>(0.0, 0.5) * frequency_ * c_3b_ /
                        (pow<4>(extraction_radius_)));

  assign_components_from_l_factors(dr_bondi_j, dr_j_2_coefficient,
                                   dr_j_3_coefficient, l_max, time);
}

void LinearizedBondiSachs::linearized_dr_bondi_u(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> dr_bondi_u,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> dr_u_2_coefficient =
      sqrt(3.0) * (-c_2a_ / pow<3>(extraction_radius_) -
                   c_2b_ / pow<5>(extraction_radius_) +
                   std::complex<double>(0.0, 1.0) * frequency_ *
                       (-c_2b_ / (pow<4>(extraction_radius_))));

  const std::complex<double> dr_u_3_coefficient =
      sqrt(6.0) *
      (-c_3a_ / pow<3>(extraction_radius_) +
       2.0 * square(frequency_) * c_3b_ / (pow<4>(extraction_radius_)) -
       5.0 * c_3b_ / pow<6>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (-5.0 * c_3b_ / pow<5>(extraction_radius_)));

  assign_components_from_l_factors(dr_bondi_u, dr_u_2_coefficient,
                                   dr_u_3_coefficient, l_max, time);
}

void LinearizedBondiSachs::linearized_dr_bondi_w(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> dr_bondi_w,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> dr_w_2_coefficient =
      (2.0 * square(frequency_) * c_2b_ / pow<3>(extraction_radius_) -
       2.0 * c_2b_ / pow<5>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (-3.0 * c_2b_ / pow<4>(extraction_radius_))) /
      sqrt(2.0);

  const std::complex<double> dr_w_3_coefficient =
      (-10.0 * frequency_ * c_3b_ / pow<5>(extraction_radius_) -
       15.0 * c_3b_ / pow<6>(extraction_radius_) +
       std::complex<double>(0.0, 4.0) * square(frequency_) * c_3b_ *
           (frequency_ / pow<3>(extraction_radius_) +
            3.0 / pow<4>(extraction_radius_))) /
      sqrt(2.0);

  assign_components_from_l_factors(dr_bondi_w, dr_w_2_coefficient,
                                   dr_w_3_coefficient, l_max, time);
}

void LinearizedBondiSachs::linearized_du_bondi_j(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> du_bondi_j,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> j_2_coefficient =
      sqrt(0.75) *
      (c_2a_ / extraction_radius_ - c_2b_ / (3.0 * pow<3>(extraction_radius_)));

  const std::complex<double> j_3_coefficient =
      sqrt(30.0) * (0.1 * c_3a_ / extraction_radius_ -
                    0.25 * c_3b_ / pow<4>(extraction_radius_) +
                    std::complex<double>(0.0, 1.0) * frequency_ *
                        (-c_3b_ / (6.0 * pow<3>(extraction_radius_))));

  assign_du_components_from_l_factors(du_bondi_j, j_2_coefficient,
                                      j_3_coefficient, l_max, time);
}

void LinearizedBondiSachs::linearized_du_bondi_u(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> du_bondi_u,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> u_2_coefficient =
      sqrt(3.0) * (0.5 * c_2a_ / square(extraction_radius_) +
                   0.25 * c_2b_ / pow<4>(extraction_radius_) +
                   std::complex<double>(0.0, 1.0) * frequency_ *
                       (c_2b_ / (3.0 * pow<3>(extraction_radius_))));

  const std::complex<double> u_3_coefficient =
      sqrt(6.0) *
      (0.5 * c_3a_ / square(extraction_radius_) -
       2.0 * square(frequency_) * c_3b_ / (3.0 * pow<3>(extraction_radius_)) +
       c_3b_ / pow<5>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (1.25 * c_3b_ / pow<4>(extraction_radius_)));

  assign_du_components_from_l_factors(du_bondi_u, u_2_coefficient,
                                      u_3_coefficient, l_max, time);
}

void LinearizedBondiSachs::linearized_du_bondi_w(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> du_bondi_w,
    const size_t l_max, const double time) const noexcept {
  const std::complex<double> w_2_coefficient =
      (-square(frequency_) * c_2b_ / square(extraction_radius_) +
       0.5 * c_2b_ / pow<4>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (c_2b_ / pow<3>(extraction_radius_))) /
      sqrt(2.0);

  const std::complex<double> w_3_coefficient =
      (2.5 * frequency_ * c_3b_ / pow<4>(extraction_radius_) +
       3.0 * c_3b_ / pow<5>(extraction_radius_) +
       std::complex<double>(0.0, 1.0) * frequency_ *
           (-2.0 * square(frequency_) * c_3b_ / square(extraction_radius_) -
            4.0 * frequency_ * c_3b_ / pow<3>(extraction_radius_))) /
      sqrt(2.0);

  assign_du_components_from_l_factors(du_bondi_w, w_2_coefficient,
                                      w_3_coefficient, l_max, time);
}

void LinearizedBondiSachs::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t l_max, const double time) const noexcept {
  Variables<tmpl::list<Tags::BondiJ, Tags::BondiU, Tags::BondiK, Tags::BondiW>>
      temporary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  auto& bondi_j = get(get<Tags::BondiJ>(temporary_variables));
  auto& bondi_u = get(get<Tags::BondiU>(temporary_variables));
  auto& bondi_k = get(get<Tags::BondiK>(temporary_variables));
  auto& bondi_w = get(get<Tags::BondiW>(temporary_variables));

  linearized_bondi_j(make_not_null(&bondi_j), l_max, time);
  linearized_bondi_u(make_not_null(&bondi_u), l_max, time);
  linearized_bondi_w(make_not_null(&bondi_w), l_max, time);
  bondi_k = sqrt(1.0 + bondi_j * conj(bondi_j));

  get<0, 0>(*spherical_metric) =
      -real(1.0 + extraction_radius_ * bondi_w.data() -
            square(extraction_radius_) *
                (conj(bondi_j.data()) * square(bondi_u.data()) +
                 bondi_k.data() * bondi_u.data() * conj(bondi_u.data())));
  get<0, 1>(*spherical_metric) = -1.0 - get<0, 0>(*spherical_metric);
  get<0, 2>(*spherical_metric) =
      square(extraction_radius_) * real(bondi_j.data() * conj(bondi_u.data()) +
                                        bondi_k.data() * bondi_u.data());
  get<0, 3>(*spherical_metric) =
      square(extraction_radius_) * imag(bondi_j.data() * conj(bondi_u.data()) +
                                        bondi_k.data() * bondi_u.data());
  get<1, 1>(*spherical_metric) = get<0, 0>(*spherical_metric) + 2.0;
  get<1, 2>(*spherical_metric) = -get<0, 2>(*spherical_metric);
  get<1, 3>(*spherical_metric) = -get<0, 3>(*spherical_metric);
  get<2, 2>(*spherical_metric) =
      square(extraction_radius_) * real(bondi_j.data() + bondi_k.data());
  get<2, 3>(*spherical_metric) =
      imag(bondi_j.data() * square(extraction_radius_));
  get<3, 3>(*spherical_metric) =
      square(extraction_radius_) * real(bondi_k.data() - bondi_j.data());
}

void LinearizedBondiSachs::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  Variables<tmpl::list<Tags::BondiJ, Tags::BondiU, Tags::BondiK, Tags::BondiW,
                       Tags::Dr<Tags::BondiJ>, Tags::Dr<Tags::BondiU>,
                       Tags::Dr<Tags::BondiK>, Tags::Dr<Tags::BondiW>>>
      temporary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  auto& bondi_j = get(get<Tags::BondiJ>(temporary_variables));
  auto& bondi_u = get(get<Tags::BondiU>(temporary_variables));
  auto& bondi_k = get(get<Tags::BondiK>(temporary_variables));
  auto& bondi_w = get(get<Tags::BondiW>(temporary_variables));
  auto& dr_bondi_j = get(get<Tags::Dr<Tags::BondiJ>>(temporary_variables));
  auto& dr_bondi_u = get(get<Tags::Dr<Tags::BondiU>>(temporary_variables));
  auto& dr_bondi_k = get(get<Tags::Dr<Tags::BondiK>>(temporary_variables));
  auto& dr_bondi_w = get(get<Tags::Dr<Tags::BondiW>>(temporary_variables));

  dt_spherical_metric(dr_spherical_metric, l_max, time);

  linearized_bondi_j(make_not_null(&bondi_j), l_max, time);
  linearized_bondi_u(make_not_null(&bondi_u), l_max, time);
  linearized_bondi_w(make_not_null(&bondi_w), l_max, time);
  bondi_k = sqrt(1.0 + bondi_j * conj(bondi_j));

  linearized_dr_bondi_j(make_not_null(&dr_bondi_j), l_max, time);
  linearized_dr_bondi_u(make_not_null(&dr_bondi_u), l_max, time);
  linearized_dr_bondi_w(make_not_null(&dr_bondi_w), l_max, time);
  dr_bondi_k =
      0.5 * (dr_bondi_j * conj(bondi_j) + conj(dr_bondi_j) * bondi_j) / bondi_k;

  // we introduce expression templates for clarity - each of these is only used
  // once.
  const auto dr_r_squared_jbar_u_squared =
      conj(dr_bondi_j.data()) * square(extraction_radius_) *
          square(bondi_u.data()) +
      2.0 * conj(bondi_j.data()) * extraction_radius_ * square(bondi_u.data()) +
      2.0 * conj(bondi_j.data()) * square(extraction_radius_) * bondi_u.data() *
          dr_bondi_u.data();
  const auto dr_r_squared_k_u_ubar =
      dr_bondi_k.data() * square(extraction_radius_) * bondi_u.data() *
          conj(bondi_u.data()) +
      2.0 * bondi_k.data() * extraction_radius_ * bondi_u.data() *
          conj(bondi_u.data()) +
      bondi_k.data() * square(extraction_radius_) *
          (dr_bondi_u.data() * conj(bondi_u.data()) +
           bondi_u.data() * conj(dr_bondi_u.data()));
  get<0, 0>(*dr_spherical_metric) =
      -get<0, 0>(*dr_spherical_metric) +
      real(dr_r_squared_jbar_u_squared + dr_r_squared_k_u_ubar -
           bondi_w.data() - extraction_radius_ * dr_bondi_w.data());

  get<0, 1>(*dr_spherical_metric) = -get<0, 0>(*dr_spherical_metric);

  const ComplexDataVector dr_r_squared_j_ubar =
      2.0 * bondi_j.data() * extraction_radius_ * conj(bondi_u.data()) +
      square(extraction_radius_) * (dr_bondi_j.data() * conj(bondi_u.data()) +
                                    bondi_j.data() * conj(dr_bondi_u.data()));
  const ComplexDataVector dr_r_squared_k_u =
      2.0 * bondi_k.data() * extraction_radius_ * bondi_u.data() +
      square(extraction_radius_) * (dr_bondi_k.data() * bondi_u.data() +
                                    bondi_k.data() * dr_bondi_u.data());
  get<0, 2>(*dr_spherical_metric) = -get<0, 2>(*dr_spherical_metric) +  real(
      dr_r_squared_j_ubar + dr_r_squared_k_u);

  get<0, 3>(*dr_spherical_metric) =
      -get<0, 3>(*dr_spherical_metric) +
      imag(dr_r_squared_j_ubar + dr_r_squared_k_u);

  get<1, 1>(*dr_spherical_metric) = get<0, 0>(*dr_spherical_metric) ;
  get<1, 2>(*dr_spherical_metric) = -get<0, 2>(*dr_spherical_metric);
  get<1, 3>(*dr_spherical_metric) = -get<0, 3>(*dr_spherical_metric);
  get<2, 2>(*dr_spherical_metric) =
      -get<2, 2>(*dr_spherical_metric) +
      2.0 * extraction_radius_ * real(bondi_j.data() + bondi_k.data()) +
      square(extraction_radius_) * real(dr_bondi_j.data() + dr_bondi_k.data());
  get<2, 3>(*dr_spherical_metric) =
      -get<2, 3>(*dr_spherical_metric) +
      imag(2.0 * bondi_j.data() * extraction_radius_ +
           dr_bondi_j.data() * square(extraction_radius_));
  get<3, 3>(*dr_spherical_metric) =
      -get<3, 3>(*dr_spherical_metric) +
      2.0 * extraction_radius_ * real(bondi_k.data() - bondi_j.data()) +
      square(extraction_radius_) * real(dr_bondi_k.data() - dr_bondi_j.data());
}

void LinearizedBondiSachs::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  Variables<tmpl::list<Tags::BondiJ, Tags::BondiU, Tags::BondiK, Tags::BondiW,
                       Tags::Du<Tags::BondiJ>, Tags::Du<Tags::BondiU>,
                       Tags::Du<Tags::BondiK>, Tags::Du<Tags::BondiW>>>
      temporary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  auto& bondi_j = get(get<Tags::BondiJ>(temporary_variables));
  auto& bondi_u = get(get<Tags::BondiU>(temporary_variables));
  auto& bondi_k = get(get<Tags::BondiK>(temporary_variables));
  auto& bondi_w = get(get<Tags::BondiW>(temporary_variables));
  auto& du_bondi_j = get(get<Tags::Du<Tags::BondiJ>>(temporary_variables));
  auto& du_bondi_u = get(get<Tags::Du<Tags::BondiU>>(temporary_variables));
  auto& du_bondi_k = get(get<Tags::Du<Tags::BondiK>>(temporary_variables));
  auto& du_bondi_w = get(get<Tags::Du<Tags::BondiW>>(temporary_variables));

  linearized_bondi_j(make_not_null(&bondi_j), l_max, time);
  linearized_bondi_u(make_not_null(&bondi_u), l_max, time);
  linearized_bondi_w(make_not_null(&bondi_w), l_max, time);
  bondi_k = sqrt(1.0 + bondi_j * conj(bondi_j));

  linearized_du_bondi_j(make_not_null(&du_bondi_j), l_max, time);
  linearized_du_bondi_u(make_not_null(&du_bondi_u), l_max, time);
  linearized_du_bondi_w(make_not_null(&du_bondi_w), l_max, time);
  du_bondi_k =
      0.5 * (du_bondi_j * conj(bondi_j) + conj(du_bondi_j) * bondi_j) / bondi_k;

  // we introduce expression templates for clarity - each of these is only used
  // once.
  const auto du_r_squared_jbar_u_squared =
      conj(du_bondi_j.data()) * square(extraction_radius_) *
          square(bondi_u.data()) +
      2.0 * conj(bondi_j.data()) * square(extraction_radius_) * bondi_u.data() *
          du_bondi_u.data();
  const auto du_r_squared_k_u_ubar =
      du_bondi_k.data() * square(extraction_radius_) * bondi_u.data() *
          conj(bondi_u.data()) +
      bondi_k.data() * square(extraction_radius_) *
          (du_bondi_u.data() * conj(bondi_u.data()) +
           bondi_u.data() * conj(du_bondi_u.data()));
  get<0, 0>(*dt_spherical_metric) =
      real(du_r_squared_jbar_u_squared + du_r_squared_k_u_ubar -
           extraction_radius_ * du_bondi_w.data());

  get<0, 1>(*dt_spherical_metric) = -get<0, 0>(*dt_spherical_metric);

  const ComplexDataVector du_r_squared_j_ubar =
      square(extraction_radius_) * (du_bondi_j.data() * conj(bondi_u.data()) +
                                    bondi_j.data() * conj(du_bondi_u.data()));
  const ComplexDataVector du_r_squared_k_u =
      square(extraction_radius_) *
      (du_bondi_k.data() * bondi_u.data() + bondi_k.data() * du_bondi_u.data());
  get<0, 2>(*dt_spherical_metric) =
      real(du_r_squared_j_ubar + du_r_squared_k_u);
  get<0, 3>(*dt_spherical_metric) =
      imag(du_r_squared_j_ubar + du_r_squared_k_u);
  get<1, 1>(*dt_spherical_metric) = get<0, 0>(*dt_spherical_metric);
  get<1, 2>(*dt_spherical_metric) = -get<0, 2>(*dt_spherical_metric);
  get<1, 3>(*dt_spherical_metric) = -get<0, 3>(*dt_spherical_metric);
  get<2, 2>(*dt_spherical_metric) =
      square(extraction_radius_) * real(du_bondi_j.data() + du_bondi_k.data());
  get<2, 3>(*dt_spherical_metric) =
      imag(du_bondi_j.data() * square(extraction_radius_));
  get<3, 3>(*dt_spherical_metric) =
      square(extraction_radius_) * real(du_bondi_k.data() - du_bondi_j.data());
}

void LinearizedBondiSachs::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t l_max, const double time,
    tmpl::type_<Tags::News> /*meta*/) const noexcept {
  const std::complex<double> news_2_coefficient =
      std::complex<double>(0.0, 0.25) * pow<3>(frequency_) * c_2b_ / sqrt(3.0);

  const std::complex<double> news_3_coefficient =
      -0.5 * pow<4>(frequency_) * c_3b_ / sqrt(15.0);

  assign_components_from_l_factors(make_not_null(&get(*news)),
                                   news_2_coefficient, news_3_coefficient,
                                   l_max, time);
}

void LinearizedBondiSachs::pup(PUP::er& p) noexcept {
  SphericalMetricData::pup(p);
  p | c_2a_;
  p | c_3a_;
  p | c_2b_;
  p | c_3b_;
  p | frequency_;
}

PUP::able::PUP_ID LinearizedBondiSachs::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::Solutions
