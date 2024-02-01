// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts::AnalyticData {

namespace detail {

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> distance_left,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::DistanceLeft<DataType> /*meta*/) const {
  tnsr::I<DataType, 3> v(x);
  v.get(0) -= xcoord_left;
  *distance_left = magnitude(v);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> distance_right,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::DistanceRight<DataType> /*meta*/) const {
  tnsr::I<DataType, 3> v(x);
  v.get(0) -= xcoord_right;
  *distance_right = magnitude(v);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_one_over_distance_left,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& normal_left =
      cache->get_var(*this, detail::Tags::NormalLeft<DataType>{});
  for (size_t i = 0; i < 3; ++i) {
    deriv_one_over_distance_left->get(i) =
        -normal_left.get(i) / square(distance_left);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_one_over_distance_right,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto& normal_right =
      cache->get_var(*this, detail::Tags::NormalRight<DataType>{});
  for (size_t i = 0; i < 3; ++i) {
    deriv_one_over_distance_right->get(i) =
        -normal_right.get(i) / square(distance_right);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    gsl::not_null<tnsr::ijk<DataType, 3>*> deriv_3_distance_left,
    gsl::not_null<Cache*> cache,
    ::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceLeft<DataType>,
                                    tmpl::size_t<Dim>, Frame::Inertial>,
                      tmpl::size_t<Dim>, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& normal_left =
      cache->get_var(*this, detail::Tags::NormalLeft<DataType>{});
  std::array<std::array<double, 3>, 3> delta{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_3_distance_left->get(i, j, k) =
            (-normal_left.get(i) * delta.at(j).at(k) -
             normal_left.get(j) * delta.at(i).at(k) -
             normal_left.get(k) * delta.at(i).at(j) +
             3 * normal_left.get(i) * normal_left.get(j) * normal_left.get(k)) /
            square(distance_left);
      }
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    gsl::not_null<tnsr::ijk<DataType, 3>*> deriv_3_distance_right,
    gsl::not_null<Cache*> cache,
    ::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceRight<DataType>,
                                    tmpl::size_t<Dim>, Frame::Inertial>,
                      tmpl::size_t<Dim>, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto& normal_right =
      cache->get_var(*this, detail::Tags::NormalRight<DataType>{});
  std::array<std::array<double, 3>, 3> delta{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_3_distance_right->get(i, j, k) =
            (-normal_right.get(i) * delta.at(j).at(k) -
             normal_right.get(j) * delta.at(i).at(k) -
             normal_right.get(k) * delta.at(i).at(j) +
             3 * normal_right.get(i) * normal_right.get(j) *
                 normal_right.get(k)) /
            square(distance_right);
      }
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> normal_left,
    const gsl::not_null<Cache*> cache,
    detail::Tags::NormalLeft<DataType> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  get<0>(*normal_left) = (get<0>(x) - xcoord_left) / distance_left;
  get<1>(*normal_left) = get<1>(x) / distance_left;
  get<2>(*normal_left) = get<2>(x) / distance_left;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> normal_right,
    const gsl::not_null<Cache*> cache,
    detail::Tags::NormalRight<DataType> /*meta*/) const {
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  get<0>(*normal_right) = (get<0>(x) - xcoord_right) / distance_right;
  get<1>(*normal_right) = get<1>(x) / distance_right;
  get<2>(*normal_right) = get<2>(x) / distance_right;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> radiative_term,
    const gsl::not_null<Cache*> cache,
    detail::Tags::RadiativeTerm<DataType> /*meta*/) const {
  get<0, 0>(*radiative_term) = 0.;
  get<0, 1>(*radiative_term) = 0.;
  get<0, 2>(*radiative_term) = 0.;
  get<1, 1>(*radiative_term) = 0.;
  get<1, 2>(*radiative_term) = 0.;
  get<2, 2>(*radiative_term) = 0.;
  add_near_zone_term_to_radiative(radiative_term, cache);
  add_present_term_to_radiative(radiative_term, cache);
  add_past_term_to_radiative(radiative_term, cache);
  add_integral_term_to_radiative(radiative_term, cache);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> near_zone_term,
    const gsl::not_null<Cache*> cache,
    detail::Tags::NearZoneTerm<DataType> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto& normal_left =
      cache->get_var(*this, detail::Tags::NormalLeft<DataType>{});
  const auto& normal_right =
      cache->get_var(*this, detail::Tags::NormalRight<DataType>{});
  const auto s = distance_left + distance_right + separation;
  get<0, 0>(*near_zone_term) = 0.;
  get<0, 1>(*near_zone_term) = 0.;
  get<0, 2>(*near_zone_term) = 0.;
  get<1, 1>(*near_zone_term) = 0.;
  get<1, 2>(*near_zone_term) = 0.;
  get<2, 2>(*near_zone_term) = 0.;
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      near_zone_term->get(i, j) +=
          0.25 / (mass_left * distance_left) *
              (2. * momentum_left.get(i) * momentum_left.get(j) +
               (3. * get(dot_product(normal_left, momentum_left)) *
                    get(dot_product(normal_left, momentum_left)) -
                5. * get(dot_product(momentum_left, momentum_left))) *
                   normal_left.get(i) * normal_left.get(j) +
               6. * get(dot_product(normal_left, momentum_left)) *
                   (normal_left.get(i) * momentum_left.get(j) +
                    normal_left.get(j) * momentum_left.get(i))) +
          0.25 / (mass_right * distance_right) *
              (2. * momentum_right.get(i) * momentum_right.get(j) +
               (3. * get(dot_product(normal_right, momentum_right)) *
                    get(dot_product(normal_right, momentum_right)) -
                5. * get(dot_product(momentum_right, momentum_right))) *
                   normal_right.get(i) * normal_right.get(j) +
               6. * get(dot_product(normal_right, momentum_right)) *
                   (normal_right.get(i) * momentum_right.get(j) +
                    normal_right.get(j) * momentum_right.get(i))) +
          0.125 * (mass_left * mass_right) *
              (-32. / s * (1. / separation + 1. / s) * normal_lr.at(i) *
                   normal_lr.at(j) +
               2. *
                   ((distance_left + distance_right) / cube(separation) +
                    12. / square(s)) *
                   normal_left.get(i) * normal_right.get(j) +
               16. * (2. / square(s) - 1. / square(separation)) *
                   (normal_left.get(i) * normal_lr.at(j) +
                    normal_left.get(j) * normal_lr.at(i)) +
               (5. / (separation * distance_left) -
                1. / cube(separation) *
                    (square(distance_right) / distance_left +
                     3. * distance_left) -
                8. / s * (1. / distance_left + 1. / s)) *
                   normal_left.get(i) * normal_left.get(j) -
               32. / s * (1. / separation + 1. / s) * normal_lr.at(i) *
                   normal_lr.at(j) +
               2. *
                   ((distance_left + distance_right) / cube(separation) +
                    12. / square(s)) *
                   normal_right.get(i) * normal_left.get(j) -
               16. * (2. / square(s) - 1. / square(separation)) *
                   (normal_right.get(i) * normal_lr.at(j) +
                    normal_right.get(j) * normal_lr.at(i)) +
               (5. / (separation * distance_right) -
                1. / cube(separation) *
                    (square(distance_left) / distance_right +
                     3. * distance_right) -
                8. / s * (1. / distance_right + 1. / s)) *
                   normal_right.get(i) * normal_right.get(j));
    }
    near_zone_term->get(i, i) +=
        0.25 / (mass_left * distance_left) *
            (get(dot_product(momentum_left, momentum_left)) -
             5. * get(dot_product(normal_left, momentum_left)) *
                 get(dot_product(normal_left, momentum_left))) +
        0.25 / (mass_right * distance_right) *
            (get(dot_product(momentum_right, momentum_right)) -
             5. * get(dot_product(normal_right, momentum_right)) *
                 get(dot_product(normal_right, momentum_right))) +
        0.125 * (mass_left * mass_right) *
            (5. * distance_left / cube(separation) *
                 (distance_left / distance_right - 1.) -
             17. / (separation * distance_left) +
             4. / (distance_left * distance_right) +
             8. / s * (1. / distance_left + 4. / separation) +
             5. * distance_right / cube(separation) *
                 (distance_right / distance_left - 1.) -
             17. / (separation * distance_right) +
             4. / (distance_left * distance_right) +
             8. / s * (1. / distance_right + 4. / separation));
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> present_term,
    const gsl::not_null<Cache*> cache,
    detail::Tags::PresentTerm<DataType> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto& normal_left =
      cache->get_var(*this, detail::Tags::NormalLeft<DataType>{});
  const auto& normal_right =
      cache->get_var(*this, detail::Tags::NormalRight<DataType>{});
  tnsr::I<DataType, 3> u1_1(x);
  tnsr::I<DataType, 3> u1_2(x);
  tnsr::I<DataType, 3> u2(x);
  for (size_t i = 0; i < 3; ++i) {
    u1_1.get(i) = momentum_left.get(i) / sqrt(mass_left);
    u1_2.get(i) = momentum_right.get(i) / sqrt(mass_right);
    u2.get(i) =
        sqrt(mass_left * mass_right / (2. * separation)) * normal_lr.at(i);
  }
  get<0, 0>(*present_term) = 0.;
  get<0, 1>(*present_term) = 0.;
  get<0, 2>(*present_term) = 0.;
  get<1, 1>(*present_term) = 0.;
  get<1, 2>(*present_term) = 0.;
  get<2, 2>(*present_term) = 0.;
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      present_term->get(i, j) +=
          -0.25 / distance_left *
              (2. * u1_1.get(i) * u1_1.get(j) +
               (3. * get(dot_product(u1_1, normal_left)) *
                    get(dot_product(u1_1, normal_left)) -
                5. * get(dot_product(u1_1, u1_1))) *
                   normal_left.get(i) * normal_left.get(j) +
               6. * get(dot_product(u1_1, normal_left)) *
                   (normal_left.get(i) * u1_1.get(j) +
                    normal_left.get(j) * u1_1.get(i)) +
               2. * u2.get(i) * u2.get(j) +
               (3. * get(dot_product(u2, normal_left)) *
                    get(dot_product(u2, normal_left)) -
                5. * get(dot_product(u2, u2))) *
                   normal_left.get(i) * normal_left.get(j) +
               6. * get(dot_product(u2, normal_left)) *
                   (normal_left.get(i) * u2.get(j) +
                    normal_left.get(j) * u2.get(i))) -
          0.25 / distance_right *
              (2. * u1_2.get(i) * u1_2.get(j) +
               (3. * get(dot_product(u1_2, normal_right)) *
                    get(dot_product(u1_2, normal_right)) -
                5. * get(dot_product(u1_2, u1_2))) *
                   normal_right.get(i) * normal_right.get(j) +
               6. * get(dot_product(u1_2, normal_right)) *
                   (normal_right.get(i) * u1_2.get(j) +
                    normal_right.get(j) * u1_2.get(i)) +
               2. * u2.get(i) * u2.get(j) +
               (3. * get(dot_product(u2, normal_right)) *
                    get(dot_product(u2, normal_right)) -
                5. * get(dot_product(u2, u2))) *
                   normal_right.get(i) * normal_right.get(j) +
               6. * get(dot_product(u2, normal_right)) *
                   (normal_right.get(i) * u2.get(j) +
                    normal_right.get(j) * u2.get(i)));
    }
    present_term->get(i, i) += -0.25 / distance_left *
                                   (get(dot_product(u1_1, u1_1)) -
                                    5. * get(dot_product(u1_1, normal_left)) *
                                        get(dot_product(u1_1, normal_left)) +
                                    get(dot_product(u2, u2)) -
                                    5. * get(dot_product(u2, normal_left)) *
                                        get(dot_product(u2, normal_left))) -
                               0.25 / distance_right *
                                   (get(dot_product(u1_2, u1_2)) -
                                    5. * get(dot_product(u1_2, normal_right)) *
                                        get(dot_product(u1_2, normal_right)) +
                                    get(dot_product(u2, u2)) -
                                    5. * get(dot_product(u2, normal_right)) *
                                        get(dot_product(u2, normal_right)));
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> past_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::PastTerm<DataType> /*meta*/) const {
  get<0, 0>(*past_term) = 0.;
  get<0, 1>(*past_term) = 0.;
  get<0, 2>(*past_term) = 0.;
  get<1, 1>(*past_term) = 0.;
  get<1, 2>(*past_term) = 0.;
  get<2, 2>(*past_term) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> integral_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::IntegralTerm<DataType> /*meta*/) const {
  get<0, 0>(*integral_term) = 0.;
  get<0, 1>(*integral_term) = 0.;
  get<0, 2>(*integral_term) = 0.;
  get<1, 1>(*integral_term) = 0.;
  get<1, 2>(*integral_term) = 0.;
  get<2, 2>(*integral_term) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> pn_conjugate_momentum3,
    const gsl::not_null<Cache*> cache,
    detail::Tags::PostNewtonianConjugateMomentum3<DataType> /*meta*/) const {
  const auto& deriv_one_over_distance_left = cache->get_var(
      *this, ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& deriv_one_over_distance_right = cache->get_var(
      *this, ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& deriv_3_distance_left = cache->get_var(
      *this,
      ::Tags::deriv<
          ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceLeft<DataType>,
                                      tmpl::size_t<Dim>, Frame::Inertial>,
                        tmpl::size_t<Dim>, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& deriv_3_distance_right = cache->get_var(
      *this,
      ::Tags::deriv<
          ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceRight<DataType>,
                                      tmpl::size_t<Dim>, Frame::Inertial>,
                        tmpl::size_t<Dim>, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial>{});
  std::array<std::array<double, 3>, 3> delta{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      pn_conjugate_momentum3->get(i, j) = 0.;
      for (size_t k = 0; k < 3; ++k) {
        pn_conjugate_momentum3->get(i, j) +=
            momentum_left.get(k) *
                (2 * (delta.at(i).at(k) * deriv_one_over_distance_left.get(j) +
                      delta.at(j).at(k) * deriv_one_over_distance_left.get(i)) -
                 delta.at(i).at(j) * deriv_one_over_distance_left.get(k) -
                 0.5 * deriv_3_distance_left.get(i, j, k)) +
            momentum_right.get(k) *
                (2 * (delta.at(i).at(k) * deriv_one_over_distance_right.get(j) +
                      delta.at(j).at(k) *
                          deriv_one_over_distance_right.get(i)) -
                 delta.at(i).at(j) * deriv_one_over_distance_right.get(k) -
                 0.5 * deriv_3_distance_right.get(i, j, k));
      }
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> pn_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    detail::Tags::PostNewtonianExtrinsicCurvature<DataType> /*meta*/) const {
  const auto& pn_conjugate_momentum3 = cache->get_var(
      *this, detail::Tags::PostNewtonianConjugateMomentum3<DataType>{});
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto E_left =
      mass_left +
      get(dot_product(momentum_left, momentum_left)) / (2. * mass_left) -
      mass_left * mass_right / (2. * separation);
  const auto E_right =
      mass_right +
      get(dot_product(momentum_right, momentum_right)) / (2. * mass_right) -
      mass_left * mass_right / (2. * separation);
  const auto pn_comformal_factor =
      1. + E_left / (2. * distance_left) + E_right / (2. * distance_right);
  const auto one_over_pn_comformal_factor_to_ten =
      1. / pow(pn_comformal_factor, 10);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      pn_extrinsic_curvature->get(i, j) = -one_over_pn_comformal_factor_to_ten *
                                          pn_conjugate_momentum3.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto E_left =
      mass_left +
      get(dot_product(momentum_left, momentum_left)) / (2. * mass_left) -
      mass_left * mass_right / (2. * separation);
  const auto E_right =
      mass_right +
      get(dot_product(momentum_right, momentum_right)) / (2. * mass_right) -
      mass_left * mass_right / (2. * separation);
  const auto pn_conformal_factor =
      1. + E_left / (2. * distance_left) + E_right / (2. * distance_right);
  get<0, 0>(*conformal_metric) = pow(pn_conformal_factor, 4);
  get<1, 1>(*conformal_metric) = pow(pn_conformal_factor, 4);
  get<2, 2>(*conformal_metric) = pow(pn_conformal_factor, 4);
  get<0, 1>(*conformal_metric) = 0.;
  get<0, 2>(*conformal_metric) = 0.;
  get<1, 2>(*conformal_metric) = 0.;
  add_radiative_term_PN_of_conformal_metric(conformal_metric, cache);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_conformal_metric->begin(), deriv_conformal_metric->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  const auto& pn_extrinsic_curvature = cache->get_var(
      *this, detail::Tags::PostNewtonianExtrinsicCurvature<DataType>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this,
      ::Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  trace(trace_extrinsic_curvature, pn_extrinsic_curvature,
        inv_conformal_metric);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
    gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*conformal_energy_density) = 0;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*conformal_stress_trace) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
    const {
  std::fill(conformal_momentum_density->begin(),
            conformal_momentum_density->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*lapse_times_conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rest_mass_density,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::RestMassDensity<DataType> /*meta*/) const {
  get(*rest_mass_density) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const {
  get(*specific_enthalpy) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> pressure,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::Pressure<DataType> /*meta*/) const {
  get(*pressure) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const {
  std::fill(spatial_velocity->begin(), spatial_velocity->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lorentz_factor,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
  get(*lorentz_factor) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  std::fill(magnetic_field->begin(), magnetic_field->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::
    add_radiative_term_PN_of_conformal_metric(
        const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
        const gsl::not_null<Cache*> cache) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const auto& radiative_term =
      cache->get_var(*this, detail::Tags::RadiativeTerm<DataType>{});
  const auto Fat =
      1. / ((1. + attenuation_parameter * attenuation_parameter * mass_left *
                      mass_left / (distance_left * distance_left)) *
            (1. + attenuation_parameter * attenuation_parameter * mass_right *
                      mass_right / (distance_right * distance_right)));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric->get(i, j) += Fat * radiative_term.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::
    add_near_zone_term_to_radiative(
        const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
        const gsl::not_null<Cache*> cache) const {
  const auto& near_zone_term =
      cache->get_var(*this, detail::Tags::NearZoneTerm<DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term->get(i, j) += near_zone_term.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::
    add_present_term_to_radiative(
        const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
        const gsl::not_null<Cache*> cache) const {
  const auto& present_term =
      cache->get_var(*this, detail::Tags::PresentTerm<DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term->get(i, j) += present_term.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::
    add_past_term_to_radiative(
        const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
        const gsl::not_null<Cache*> cache) const {
  const auto& past_term =
      cache->get_var(*this, detail::Tags::PastTerm<DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term->get(i, j) += past_term.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::
    add_integral_term_to_radiative(
        const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
        const gsl::not_null<Cache*> cache) const {
  const auto& integral_term =
      cache->get_var(*this, detail::Tags::IntegralTerm<DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term->get(i, j) += integral_term.get(i, j);
    }
  }
}

template class BinaryWithGravitationalWavesVariables<DataVector>;

}  // namespace detail

void BinaryWithGravitationalWaves::initialize() {
  double separation = xcoord_right() - xcoord_left();

  total_mass = mass_left() + mass_right();
  reduced_mass = mass_left() * mass_right() / total_mass;
  reduced_mass_over_total_mass = reduced_mass / total_mass;

  double p_circular_squared =
      reduced_mass * reduced_mass * total_mass / separation +
      4. * reduced_mass * reduced_mass * total_mass * total_mass /
          (separation * separation) +
      (74. - 43. * reduced_mass / total_mass) * reduced_mass * reduced_mass *
          total_mass * total_mass * total_mass /
          (8. * separation * separation * separation);
  ymomentum_left_ = sqrt(p_circular_squared);
  ymomentum_right_ = -sqrt(p_circular_squared);
}

PUP::able::PUP_ID BinaryWithGravitationalWaves::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    BinaryWithGravitationalWavesVariables<DataVector>::Cache>;
