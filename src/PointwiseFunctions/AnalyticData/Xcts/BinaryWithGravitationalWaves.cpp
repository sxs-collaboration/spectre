// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"

#include <boost/math/interpolators/cubic_hermite.hpp>
#include <cstddef>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// Boost MultiArray is used internally in odeint, so odeint must be included
// later
#include <boost/numeric/odeint.hpp>

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
  std::fill(radiative_term->begin(), radiative_term->end(), 0.);
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
  const DataType s = distance_left + distance_right + separation;
  std::fill(near_zone_term->begin(), near_zone_term->end(), 0.);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      near_zone_term->get(i, j) +=
          0.25 / (mass_left * distance_left) *
              (2. * momentum_left.at(i) * momentum_left.at(j) +
               (3. * get(this_dot_product(normal_left, momentum_left)) *
                    get(this_dot_product(normal_left, momentum_left)) -
                5. * dot(momentum_left, momentum_left)) *
                   normal_left.get(i) * normal_left.get(j) +
               6. * get(this_dot_product(normal_left, momentum_left)) *
                   (normal_left.get(i) * momentum_left.at(j) +
                    normal_left.get(j) * momentum_left.at(i))) +
          0.25 / (mass_right * distance_right) *
              (2. * momentum_right.at(i) * momentum_right.at(j) +
               (3. * get(this_dot_product(normal_right, momentum_right)) *
                    get(this_dot_product(normal_right, momentum_right)) -
                5. * dot(momentum_right, momentum_right)) *
                   normal_right.get(i) * normal_right.get(j) +
               6. * get(this_dot_product(normal_right, momentum_right)) *
                   (normal_right.get(i) * momentum_right.at(j) +
                    normal_right.get(j) * momentum_right.at(i))) +
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
            (dot(momentum_left, momentum_left) -
             5. * get(this_dot_product(normal_left, momentum_left)) *
                 get(this_dot_product(normal_left, momentum_left))) +
        0.25 / (mass_right * distance_right) *
            (dot(momentum_right, momentum_right) -
             5. * get(this_dot_product(normal_right, momentum_right)) *
                 get(this_dot_product(normal_right, momentum_right))) +
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
    u1_1.get(i) = momentum_left.at(i) / sqrt(mass_left);
    u1_2.get(i) = momentum_right.at(i) / sqrt(mass_right);
    u2.get(i) =
        sqrt(mass_left * mass_right / (2. * separation)) * normal_lr.at(i);
  }
  std::fill(present_term->begin(), present_term->end(), 0.);
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
    const gsl::not_null<Cache*> cache,
    detail::Tags::PastTerm<DataType> /*meta*/) const {
  const auto& retarded_time_left =
      get(cache->get_var(*this, detail::Tags::RetardedTimeLeft<DataType>{}));
  const auto& retarded_time_right =
      get(cache->get_var(*this, detail::Tags::RetardedTimeRight<DataType>{}));
  DataType distance_left_at_retarded_time_left =
      get(get_past_distance_left(retarded_time_left));
  DataType distance_right_at_retarded_time_right =
      get(get_past_distance_right(retarded_time_right));
  DataType separation_at_retarded_time_left =
      get(get_past_separation(retarded_time_left));
  DataType separation_at_retarded_time_right =
      get(get_past_separation(retarded_time_right));
  tnsr::I<DataType, 3> momentum_left_at_retarded_time_left =
      get_past_momentum_left(retarded_time_left);
  tnsr::I<DataType, 3> momentum_right_at_retarded_time_right =
      get_past_momentum_right(retarded_time_right);
  tnsr::I<DataType, 3> normal_left_at_retarded_time_left =
      get_past_normal_left(retarded_time_left);
  tnsr::I<DataType, 3> normal_right_at_retarded_time_right =
      get_past_normal_right(retarded_time_right);
  tnsr::I<DataType, 3> normal_lr_at_retarded_time_left =
      get_past_normal_lr(retarded_time_left);
  tnsr::I<DataType, 3> normal_lr_at_retarded_time_right =
      get_past_normal_lr(retarded_time_right);
  tnsr::I<DataType, 3> u1_1;
  tnsr::I<DataType, 3> u1_2;
  tnsr::I<DataType, 3> u2_1;
  tnsr::I<DataType, 3> u2_2;
  for (size_t i = 0; i < 3; ++i) {
    u1_1.get(i) =
        momentum_left_at_retarded_time_left.get(i) / std::sqrt(mass_left);
    u2_1.get(i) =
        sqrt(mass_left * mass_right / (2. * separation_at_retarded_time_left)) *
        normal_lr_at_retarded_time_left.get(i);

    u1_2.get(i) =
        momentum_right_at_retarded_time_right.get(i) / std::sqrt(mass_right);
    u2_2.get(i) = sqrt(mass_left * mass_right /
                       (2. * separation_at_retarded_time_right)) *
                  normal_lr_at_retarded_time_right.get(i);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      past_term->get(i, j) =
          -1. / (distance_left_at_retarded_time_left) *
              (4. * u1_1.get(i) * u1_1.get(j) +
               (2. * get(dot_product(u1_1, u1_1)) +
                2. * get(dot_product(u1_1, normal_left_at_retarded_time_left)) *
                    get(dot_product(u1_1, normal_left_at_retarded_time_left))) *
                   normal_left_at_retarded_time_left.get(i) *
                   normal_left_at_retarded_time_left.get(j) -
               4. * get(dot_product(u1_1, normal_left_at_retarded_time_left)) *
                   (normal_left_at_retarded_time_left.get(i) * u1_1.get(j) +
                    normal_left_at_retarded_time_left.get(j) * u1_1.get(i))) -
          1. / (distance_right_at_retarded_time_right) *
              (4. * u1_2.get(i) * u1_2.get(j) +
               (2. * get(dot_product(u1_2, u1_2)) +
                2. *
                    get(dot_product(u1_2,
                                    normal_right_at_retarded_time_right)) *
                    get(dot_product(u1_2,
                                    normal_right_at_retarded_time_right))) *
                   normal_right_at_retarded_time_right.get(i) *
                   normal_right_at_retarded_time_right.get(j) -
               4. *
                   get(dot_product(u1_2, normal_right_at_retarded_time_right)) *
                   (normal_right_at_retarded_time_right.get(i) * u1_2.get(j) +
                    normal_right_at_retarded_time_right.get(j) * u1_2.get(i))) -
          1. / (distance_left_at_retarded_time_left) *
              (4. * u2_1.get(i) * u2_1.get(j) +
               (2. * get(dot_product(u2_1, u2_1)) +
                2. * get(dot_product(u2_1, normal_left_at_retarded_time_left)) *
                    get(dot_product(u2_1, normal_left_at_retarded_time_left))) *
                   normal_left_at_retarded_time_left.get(i) *
                   normal_left_at_retarded_time_left.get(j) -
               4. * get(dot_product(u2_1, normal_left_at_retarded_time_left)) *
                   (normal_left_at_retarded_time_left.get(i) * u2_1.get(j) +
                    normal_left_at_retarded_time_left.get(j) * u2_1.get(i))) -
          1. / (distance_right_at_retarded_time_right) *
              (4. * u2_2.get(i) * u2_2.get(j) +
               (2. * get(dot_product(u2_2, u2_2)) +
                2. *
                    get(dot_product(u2_2,
                                    normal_right_at_retarded_time_right)) *
                    get(dot_product(u2_2,
                                    normal_right_at_retarded_time_right))) *
                   normal_right_at_retarded_time_right.get(i) *
                   normal_right_at_retarded_time_right.get(j) -
               4. *
                   get(dot_product(u2_2, normal_right_at_retarded_time_right)) *
                   (normal_right_at_retarded_time_right.get(i) * u2_2.get(j) +
                    normal_right_at_retarded_time_right.get(j) * u2_2.get(i)));
    }
    past_term->get(i, i) +=
        -1. / (distance_left_at_retarded_time_left) *
            (-2. * get(dot_product(u1_1, u1_1)) +
             2. * get(dot_product(u1_1, normal_left_at_retarded_time_left)) *
                 get(dot_product(u1_1, normal_left_at_retarded_time_left))) -
        1. / (distance_right_at_retarded_time_right) *
            (-2. * get(dot_product(u1_2, u1_2)) +
             2. * get(dot_product(u1_2, normal_right_at_retarded_time_right)) *
                 get(dot_product(u1_2, normal_right_at_retarded_time_right))) -
        1. / (distance_left_at_retarded_time_left) *
            (-2. * get(dot_product(u2_1, u2_1)) +
             2. * get(dot_product(u2_1, normal_left_at_retarded_time_left)) *
                 get(dot_product(u2_1, normal_left_at_retarded_time_left))) -
        1. / (distance_right_at_retarded_time_right) *
            (-2. * get(dot_product(u2_2, u2_2)) +
             2. * get(dot_product(u2_2, normal_right_at_retarded_time_right)) *
                 get(dot_product(u2_2, normal_right_at_retarded_time_right)));
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> integral_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::IntegralTerm<DataType> /*meta*/) const {
  std::fill(integral_term->begin(), integral_term->end(), 0.);
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
            momentum_left.at(k) *
                (2 * (delta.at(i).at(k) * deriv_one_over_distance_left.get(j) +
                      delta.at(j).at(k) * deriv_one_over_distance_left.get(i)) -
                 delta.at(i).at(j) * deriv_one_over_distance_left.get(k) -
                 0.5 * deriv_3_distance_left.get(i, j, k)) +
            momentum_right.at(k) *
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
  const double E_left = mass_left +
                        dot(momentum_left, momentum_left) / (2. * mass_left) -
                        mass_left * mass_right / (2. * separation);
  const double E_right =
      mass_right + dot(momentum_right, momentum_right) / (2. * mass_right) -
      mass_left * mass_right / (2. * separation);
  const DataType pn_comformal_factor =
      1. + E_left / (2. * distance_left) + E_right / (2. * distance_right);
  const DataType one_over_pn_comformal_factor_to_ten =
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
    const gsl::not_null<Scalar<DataType>*> retarded_time_left,
    const gsl::not_null<Cache*> cache,
    detail::Tags::RetardedTimeLeft<DataType> /*meta*/) const {
  get(*retarded_time_left) = find_retarded_time_left(cache);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> retarded_time_right,
    const gsl::not_null<Cache*> cache,
    detail::Tags::RetardedTimeRight<DataType> /*meta*/) const {
  get(*retarded_time_right) = find_retarded_time_right(cache);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rootfinder_bracket_time_lower,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RootFinderBracketTimeLower<DataType> /*meta*/) const {
  get(*rootfinder_bracket_time_lower) = past_time.front();
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rootfinder_bracket_time_upper,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RootFinderBracketTimeUpper<DataType> /*meta*/) const {
  get(*rootfinder_bracket_time_upper) = past_time.back();

}  // namespace detail

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  const auto& distance_left =
      get(cache->get_var(*this, detail::Tags::DistanceLeft<DataType>{}));
  const auto& distance_right =
      get(cache->get_var(*this, detail::Tags::DistanceRight<DataType>{}));
  const double E_left = mass_left +
                        dot(momentum_left, momentum_left) / (2. * mass_left) -
                        mass_left * mass_right / (2. * separation);
  const double E_right =
      mass_right + dot(momentum_right, momentum_right) / (2. * mass_right) -
      mass_left * mass_right / (2. * separation);
  const DataType pn_conformal_factor =
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
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    const auto& conformal_metric = cache->get_var(
        *this, Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
    partial_derivative(deriv_conformal_metric, conformal_metric, mesh->get(),
                       inv_jacobian->get());
  } else {
    (void)deriv_conformal_metric;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
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
  const DataType Fat =
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

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::this_dot_product(
    const tnsr::I<DataType, 3>& a, const std::array<double, 3>& b) const {
  Scalar<DataType> result{get_size(get<0>(a))};
  get(result) = a.get(0) * b.at(0) + a.get(1) * b.at(1) + a.get(2) * b.at(2);
  return result;
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::this_dot_product(
    const std::array<double, 3>& a, const tnsr::I<DataType, 3>& b) const {
  return this_dot_product(b, a);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<
    DataType>::interpolate_past_history() {
  // Now interpolate the past history
  using boost::math::interpolators::cardinal_cubic_hermite;

  for (size_t i = 0; i < 3; ++i) {
    // static_cast being used because boost requires non-const arrays
    interpolation_position_left.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_position_left.at(i)),
        static_cast<std::vector<double>>(past_dt_position_left.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
    interpolation_position_right.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_position_right.at(i)),
        static_cast<std::vector<double>>(past_dt_position_right.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
    interpolation_momentum_left.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_momentum_left.at(i)),
        static_cast<std::vector<double>>(past_dt_momentum_left.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
    interpolation_momentum_right.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_momentum_right.at(i)),
        static_cast<std::vector<double>>(past_dt_momentum_right.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
  }
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::find_retarded_time_left(
    const gsl::not_null<Cache*> cache) const {
  const auto& rootfinder_bracket_time_lower = cache->get_var(
      *this, detail::Tags::RootFinderBracketTimeLower<DataType>{});
  const auto& rootfinder_bracket_time_upper = cache->get_var(
      *this, detail::Tags::RootFinderBracketTimeUpper<DataType>{});
  return RootFinder::toms748<true>(
      [this](const auto time, const size_t i) {
        tnsr::I<double, 3> v;
        for (size_t j = 0; j < 3; ++j) {
          v.get(j) =
              this->x.get(j)[i] - this->interpolation_position_left.at(j)(time);
        }

        return get(magnitude(v)) + time;
      },
      get(rootfinder_bracket_time_lower), get(rootfinder_bracket_time_upper),
      1e-8, 1e-10);
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::find_retarded_time_right(
    const gsl::not_null<Cache*> cache) const {
  const auto& rootfinder_bracket_time_lower = cache->get_var(
      *this, detail::Tags::RootFinderBracketTimeLower<DataType>{});
  const auto& rootfinder_bracket_time_upper = cache->get_var(
      *this, detail::Tags::RootFinderBracketTimeUpper<DataType>{});
  return RootFinder::toms748<true>(
      [this](const auto time, const size_t i) {
        tnsr::I<double, 3> v;
        for (size_t j = 0; j < 3; ++j) {
          v.get(j) = this->x.get(j)[i] -
                     this->interpolation_position_right.at(j)(time);
        }

        return get(magnitude(v)) + time;
      },
      get(rootfinder_bracket_time_lower), get(rootfinder_bracket_time_upper),
      1e-8, 1e-10);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_past_distance_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - interpolation_position_left.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_past_distance_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - interpolation_position_right.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_past_separation(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_position_right.at(j)(time[i]) -
                    interpolation_position_left.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_past_momentum_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_momentum_left.at(j)(time[i]);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_past_momentum_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_momentum_right.at(j)(time[i]);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_past_normal_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> distance_left = get_past_distance_left(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = (x.get(j)[i] - interpolation_position_left.at(j)(time[i])) /
                    get(distance_left)[i];
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_past_normal_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> distance_right = get_past_distance_right(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] =
          (x.get(j)[i] - interpolation_position_right.at(j)(time[i])) /
          get(distance_right)[i];
    }
  }

  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_past_normal_lr(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> past_separation = get_past_separation(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = (interpolation_position_left.at(j)(time[i]) -
                     interpolation_position_right.at(j)(time[i])) /
                    get(past_separation)[i];
    }
  }
  return v;
}

template class BinaryWithGravitationalWavesVariables<DataVector>;

}  // namespace detail


void BinaryWithGravitationalWaves::reserve_vector_capacity() {
  for (size_t i = 0; i < 3; ++i) {
    past_position_left_.at(i).reserve(number_of_steps);
    past_position_right_.at(i).reserve(number_of_steps);
    past_momentum_left_.at(i).reserve(number_of_steps);
    past_momentum_right_.at(i).reserve(number_of_steps);
    past_dt_position_left_.at(i).reserve(number_of_steps);
    past_dt_position_right_.at(i).reserve(number_of_steps);
    past_dt_momentum_left_.at(i).reserve(number_of_steps);
    past_dt_momentum_right_.at(i).reserve(number_of_steps);
  }
  past_time_.reserve(number_of_steps);
}

void BinaryWithGravitationalWaves::reverse_vector() {
  std::reverse(past_time_.begin(), past_time_.end());
  for (size_t i = 0; i < 3; ++i) {
    reverse(past_position_left_.at(i).begin(), past_position_left_.at(i).end());
    reverse(past_position_right_.at(i).begin(),
            past_position_right_.at(i).end());
    reverse(past_momentum_left_.at(i).begin(), past_momentum_left_.at(i).end());
    reverse(past_momentum_right_.at(i).begin(),
            past_momentum_right_.at(i).end());
    reverse(past_dt_position_left_.at(i).begin(),
            past_dt_position_left_.at(i).end());
    reverse(past_dt_position_right_.at(i).begin(),
            past_dt_position_right_.at(i).end());
    reverse(past_dt_momentum_left_.at(i).begin(),
            past_dt_momentum_left_.at(i).end());
    reverse(past_dt_momentum_right_.at(i).begin(),
            past_dt_momentum_right_.at(i).end());
  }
}

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

  initial_state_position = {{separation / total_mass, 0., 0.}};
  initial_state_momentum = {{0., ymomentum_right_ / reduced_mass, 0.}};

  time_step = .1;
  initial_time = 0.;
  final_time = std::round(-2 * outer_radius() / time_step) * time_step;
  number_of_steps =
      static_cast<size_t>(std::round((initial_time - final_time) / time_step));
}

void BinaryWithGravitationalWaves::hamiltonian_system(
    const BinaryWithGravitationalWaves::state_type& x,
    BinaryWithGravitationalWaves::state_type& dpdt) const {
  // H = H_Newt + H_1PN + H_2PN + H_3PN

  double pdotp = x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
  double qdotq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  double qdotp = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];

  double dH_dp0_Newt = x[3];
  double dH_dp0_1 =
      0.5 * x[3] * pdotp * (-1. + 3. * reduced_mass_over_total_mass) -
      (x[0] * (x[4] * x[1] + x[5] * x[2]) * reduced_mass_over_total_mass +
       x[3] * (x[1] * x[1] + x[2] * x[2]) *
           (3. + reduced_mass_over_total_mass) +
       x[3] * x[0] * x[0] * (3. + 2. * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  double dH_dp0_2 =
      0.125 *
      (3. * x[3] * pdotp * pdotp *
           (1. - 5. * reduced_mass_over_total_mass +
            5. * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8. * (3. * x[0] * qdotp * reduced_mass_over_total_mass +
              x[3] * qdotq * (5. + 8. * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1. / std::sqrt(qdotq)) *
           (-(12. * x[0] * qdotp * qdotp * qdotp *
              reduced_mass_over_total_mass * reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4. * pdotp * x[0] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (4. * x[3] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4. * x[3] * pdotp *
                (-5. + 20. * reduced_mass_over_total_mass +
                 3. * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  double dH_dp0_3 =
      0.0625 *
      (5. * x[3] * pdotp * pdotp * pdotp *
           (-1. + 7. * (-1. + reduced_mass_over_total_mass) *
                      (-1. + reduced_mass_over_total_mass) *
                      reduced_mass_over_total_mass) +
       1.0 / (3. * qdotq * qdotq * qdotq) * 2 *
           (3. * pdotp * x[0] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            3. * x[3] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            8. * x[0] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            6. * x[3] * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) -
       (3. * x[0] * qdotp * reduced_mass_over_total_mass *
            (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
        x[3] * qdotq *
            (600. + reduced_mass_over_total_mass *
                        (1340. - 3. * M_PI * M_PI +
                         552. * reduced_mass_over_total_mass))) /
           (6. * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2. / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[0] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2. * x[3] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6. * pdotp * x[0] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * x[3] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15. * x[0] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3. * x[3] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass)))));

  double dH_dp1_Newt = x[4];
  double dH_dp1_1 =
      0.5 * x[4] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[1] * (x[3] * x[0] + x[5] * x[2]) * reduced_mass_over_total_mass +
       x[4] * (x[0] * x[0] + x[2] * x[2]) * (3 + reduced_mass_over_total_mass) +
       x[4] * x[1] * x[1] * (3 + 2 * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  double dH_dp1_2 =
      0.125 *
      (3. * x[4] * pdotp * pdotp *
           (1. - 5. * reduced_mass_over_total_mass +
            5. * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8. * (3. * x[1] * qdotp * reduced_mass_over_total_mass +
              x[4] * qdotq * (5. + 8. * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1. / std::sqrt(qdotq)) *
           (-(12. * x[1] * qdotp * qdotp * qdotp *
              reduced_mass_over_total_mass * reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4. * pdotp * x[1] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (8. * x[4] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4. * x[4] * pdotp *
                (-5. + 20. * reduced_mass_over_total_mass +
                 3. * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  double dH_dp1_3 =
      0.0625 *
      (5. * x[4] * pdotp * pdotp * pdotp *
           (-1. + 7. * (-1. + reduced_mass_over_total_mass) *
                      (-1. + reduced_mass_over_total_mass) *
                      reduced_mass_over_total_mass) +
       1.0 / (3. * qdotq * qdotq * qdotq) * 2 *
           (3. * pdotp * x[1] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            3. * x[4] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[1] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            6. * x[4] * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) -
       (3. * x[1] * qdotp * reduced_mass_over_total_mass *
            (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
        x[4] * qdotq *
            (600. + reduced_mass_over_total_mass *
                        (1340. - 3. * M_PI * M_PI +
                         552. * reduced_mass_over_total_mass))) /
           (6. * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2. / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[1] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2. * x[4] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6. * pdotp * x[1] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * x[4] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15. * x[1] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3. * x[4] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass)))));

  double dH_dp2_Newt = x[5];
  double dH_dp2_1 =
      0.5 * x[5] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[2] * (x[4] * x[1] + x[3] * x[0]) * reduced_mass_over_total_mass +
       x[5] * (x[0] * x[0] + x[1] * x[1]) *
           (3. + reduced_mass_over_total_mass) +
       x[5] * x[2] * x[2] * (3. + 2. * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  double dH_dp2_2 =
      0.125 *
      (3. * x[5] * pdotp * pdotp *
           (1. - 5. * reduced_mass_over_total_mass +
            5 * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8. * (3. * x[2] * qdotp * reduced_mass_over_total_mass +
              x[5] * qdotq * (5. + 8. * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1. / std::sqrt(qdotq)) *
           (-(12. * x[2] * qdotp * qdotp * qdotp *
              reduced_mass_over_total_mass * reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4. * pdotp * x[2] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (8. * x[5] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4. * x[5] * pdotp *
                (-5. + 20. * reduced_mass_over_total_mass +
                 3. * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  double dH_dp2_3 =
      0.0625 *
      (5. * x[5] * pdotp * pdotp * pdotp *
           (-1. + 7. * (-1. + reduced_mass_over_total_mass) *
                      (-1. + reduced_mass_over_total_mass) *
                      reduced_mass_over_total_mass) +
       1.0 / (3. * qdotq * qdotq * qdotq) * 2. *
           (3. * pdotp * x[2] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            3. * x[5] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[2] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            6. * x[5] * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) -
       (3. * x[2] * qdotp * reduced_mass_over_total_mass *
            (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
        x[5] * qdotq *
            (600. + reduced_mass_over_total_mass *
                        (1340. - 3. * M_PI * M_PI +
                         552. * reduced_mass_over_total_mass))) /
           (6. * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2. / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[2] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2. * x[5] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6. * pdotp * x[2] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * x[5] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15. * x[2] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3. * x[5] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass)))));

  double dH_dq0_Newt = x[0] / std::sqrt(qdotq * qdotq * qdotq);
  double dH_dq0_1 =
      (-2. * x[0] * std::sqrt(qdotq) +
       3. * x[0] * qdotp * qdotp * reduced_mass_over_total_mass -
       2. * x[3] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[0] * qdotq * (3. + reduced_mass_over_total_mass)) /
      (2. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq0_2 =
      (-48. * x[0] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24. * x[3] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15. * x[0] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * pdotp * x[0] * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       12. * x[3] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4. * x[3] * pdotp * qdotp * qdotq * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * x[0] * qdotq * (1. + 3. * reduced_mass_over_total_mass) -
       8. * pdotp * x[0] * std::sqrt(qdotq * qdotq * qdotq) *
           (5. + 8. * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[0] * qdotq * qdotq *
           (-5. + reduced_mass_over_total_mass *
                      (20. + 3. * reduced_mass_over_total_mass))) /
      (8. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq0_3 =
      (3. / 2. * qdotp * qdotq *
           (x[0] * (x[1] * x[4] + x[2] * x[5]) -
            x[3] * (x[1] * x[1] + x[2] * x[2])) *
           reduced_mass_over_total_mass *
           (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
       2. * x[0] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12. + (-872. + 63. * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6. * qdotp * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass *
           (pdotp * pdotp * x[0] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) -
            6. * pdotp * x[0] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) +
            6. * x[3] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1. + reduced_mass_over_total_mass) -
            15. * x[0] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15. * x[3] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[3] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2. + 3. * reduced_mass_over_total_mass)) -
       2. * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3. * pdotp * x[0] * qdotp * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) -
            3. * x[3] * pdotp * qdotq * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[0] * qdotp * qdotp * qdotp *
                (5. + 43. * reduced_mass_over_total_mass) -
            8. * x[3] * qdotp * qdotp * qdotq *
                (5. + 43. * reduced_mass_over_total_mass)) -
       2. * x[0] * std::sqrt(qdotq) *
           (3. * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            4. * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            3. * pdotp * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) +
       0.75 * x[0] * qdotq *
           (3. * qdotp * qdotp * reduced_mass_over_total_mass *
                (340. + 3. * M_PI * M_PI +
                 112. * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600. + reduced_mass_over_total_mass *
                            (1340. - 3. * M_PI * M_PI +
                             552. * reduced_mass_over_total_mass))) -
       3. * x[0] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5. * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass))))) /
      (48. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                       qdotq * qdotq));

  double dH_dq1_Newt = x[1] / std::sqrt(qdotq * qdotq * qdotq);
  double dH_dq1_1 =
      (-2. * x[1] * std::sqrt(qdotq) +
       3. * x[1] * qdotp * qdotp * reduced_mass_over_total_mass -
       2. * x[4] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[1] * qdotq * (3. + reduced_mass_over_total_mass)) /
      (2. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq1_2 =
      (-48. * x[1] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24. * x[4] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15. * x[1] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * pdotp * x[1] * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       12. * x[4] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4. * x[4] * pdotp * qdotp * qdotq * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * x[1] * qdotq * (1. + 3. * reduced_mass_over_total_mass) -
       8. * pdotp * x[1] * std::sqrt(qdotq * qdotq * qdotq) *
           (5. + 8. * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[1] * qdotq * qdotq *
           (-5. + reduced_mass_over_total_mass *
                      (20. + 3. * reduced_mass_over_total_mass))) /
      (8. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq1_3 =
      (3. / 2. * qdotp * qdotq *
           (x[1] * (x[0] * x[3] + x[2] * x[5]) -
            x[4] * (x[0] * x[0] + x[2] * x[2])) *
           reduced_mass_over_total_mass *
           (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
       2. * x[1] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12 + (-872. + 63. * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6. * qdotp * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass *
           (pdotp * pdotp * x[1] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) -
            6. * pdotp * x[1] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) +
            6. * x[4] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1. + reduced_mass_over_total_mass) -
            15. * x[1] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15. * x[4] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[4] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2. + 3. * reduced_mass_over_total_mass)) -
       2. * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3. * pdotp * x[1] * qdotp * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) -
            3. * x[4] * pdotp * qdotq * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[1] * qdotp * qdotp * qdotp *
                (5. + 43. * reduced_mass_over_total_mass) -
            8. * x[4] * qdotp * qdotp * qdotq *
                (5. + 43. * reduced_mass_over_total_mass)) -
       2. * x[1] * std::sqrt(qdotq) *
           (3. * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            4. * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            3. * pdotp * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) +
       0.75 * x[1] * qdotq *
           (3. * qdotp * qdotp * reduced_mass_over_total_mass *
                (340. + 3. * M_PI * M_PI +
                 112. * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600. + reduced_mass_over_total_mass *
                            (1340. - 3. * M_PI * M_PI +
                             552. * reduced_mass_over_total_mass))) -
       3. * x[1] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5. * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass))))) /
      (48. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                       qdotq * qdotq));

  double dH_dq2_Newt = x[2] / std::sqrt(qdotq * qdotq * qdotq);
  double dH_dq2_1 =
      (-2. * x[2] * std::sqrt(qdotq) +
       3. * x[2] * qdotp * qdotp * reduced_mass_over_total_mass -
       2. * x[5] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[2] * qdotq * (3. + reduced_mass_over_total_mass)) /
      (2. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq2_2 =
      (-48. * x[2] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24. * x[5] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15. * x[2] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * pdotp * x[2] * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       12. * x[5] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4. * x[5] * pdotp * qdotp * qdotq * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * x[2] * qdotq * (1. + 3. * reduced_mass_over_total_mass) -
       8. * pdotp * x[2] * std::sqrt(qdotq * qdotq * qdotq) *
           (5. + 8. * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[2] * qdotq * qdotq *
           (-5. + reduced_mass_over_total_mass *
                      (20. + 3. * reduced_mass_over_total_mass))) /
      (8. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq2_3 =
      (3. / 2. * qdotp * qdotq *
           (x[2] * (x[0] * x[3] + x[1] * x[3]) -
            x[5] * (x[0] * x[0] + x[1] * x[1])) *
           reduced_mass_over_total_mass *
           (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
       2. * x[2] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12. + (-872. + 63. * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6. * qdotp * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass *
           (pdotp * pdotp * x[2] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) -
            6. * pdotp * x[2] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) +
            6. * x[5] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1. + reduced_mass_over_total_mass) -
            15. * x[2] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15. * x[5] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[5] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2. + 3. * reduced_mass_over_total_mass)) -
       2. * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3. * pdotp * x[2] * qdotp * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) -
            3. * x[5] * pdotp * qdotq * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[2] * qdotp * qdotp * qdotp *
                (5. + 43. * reduced_mass_over_total_mass) -
            8. * x[5] * qdotp * qdotp * qdotq *
                (5. + 43. * reduced_mass_over_total_mass)) -
       2. * x[2] * std::sqrt(qdotq) *
           (3. * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            4. * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            3. * pdotp * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) +
       0.75 * x[2] * qdotq *
           (3. * qdotp * qdotp * reduced_mass_over_total_mass *
                (340. + 3. * M_PI * M_PI +
                 112. * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600. + reduced_mass_over_total_mass *
                            (1340. - 3. * M_PI * M_PI +
                             552. * reduced_mass_over_total_mass))) -
       3. * x[2] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5. * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass))))) /
      (48. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                       qdotq * qdotq));

  double L = total_mass * reduced_mass *
             sqrt((x[1] * x[5] - x[2] * x[4]) * (x[1] * x[5] - x[2] * x[4]) +
                  (x[2] * x[3] - x[0] * x[5]) * (x[2] * x[3] - x[0] * x[5]) +
                  (x[0] * x[4] - x[1] * x[3]) * (x[0] * x[4] - x[1] * x[3]));
  double w =
      reduced_mass / (total_mass * mass_left()) * sqrt(pdotp) / sqrt(qdotq);
  double vw = std::cbrt(total_mass * w);
  double gamma_Euler = 0.57721566490153286060651209008240243104215933593992;

  double f2 = -(1247. / 336.) - (35. / 12.) * reduced_mass_over_total_mass;
  double f3 = 4 * M_PI;
  double f4 =
      -(44711. / 9072.) + (9271. / 504.) * reduced_mass_over_total_mass +
      (65. / 18.) * reduced_mass_over_total_mass * reduced_mass_over_total_mass;
  double f5 =
      -(8191. / 672. + 583. / 24. * reduced_mass_over_total_mass) * M_PI;
  double f6 = (6643739519. / 69854400.) + (16. / 3.) * M_PI * M_PI -
              (1712. / 105.) * gamma_Euler +
              (-134543. / 7776. + (41. / 48.) * M_PI * M_PI) *
                  reduced_mass_over_total_mass -
              (94403. / 3024.) * reduced_mass_over_total_mass *
                  reduced_mass_over_total_mass -
              (775. / 324.) * reduced_mass_over_total_mass *
                  reduced_mass_over_total_mass * reduced_mass_over_total_mass;
  double fl6 = -1712. / 105.;
  double f7 = (-16285. / 504. + 214745. / 1728. * reduced_mass_over_total_mass +
               193385. / 3024. * reduced_mass_over_total_mass *
                   reduced_mass_over_total_mass) *
              M_PI;

  double dE_dt =
      -(32. / 5.) * reduced_mass_over_total_mass *
      reduced_mass_over_total_mass * vw * vw * vw * vw * vw * vw * vw * vw *
      vw * vw *
      (1. + f2 * vw * vw + f3 * vw * vw * vw + f4 * vw * vw * vw * vw +
       f5 * vw * vw * vw * vw * vw + f6 * vw * vw * vw * vw * vw * vw +
       fl6 * vw * vw * vw * vw * vw * vw * std::log(4. * vw) +
       f7 * vw * vw * vw * vw * vw * vw * vw);

  std::array<double, 3> F{1. / (w * L) * dE_dt * x[3],
                          1. / (w * L) * dE_dt * x[4],
                          1. / (w * L) * dE_dt * x[5]};

  dpdt[0] = (1. / total_mass) *
            (dH_dp0_Newt + dH_dp0_1 + dH_dp0_2 + dH_dp0_3);  // dX0/dt = dH/dP0
  dpdt[1] = (1. / total_mass) *
            (dH_dp1_Newt + dH_dp1_1 + dH_dp1_2 + dH_dp1_3);  // dX1/dt = dH/dP1
  dpdt[2] = (1. / total_mass) *
            (dH_dp2_Newt + dH_dp2_1 + dH_dp2_2 + dH_dp2_3);  // dX2/dt = dH/dP2

  dpdt[3] =
      -(1. / total_mass) * (dH_dq0_Newt + dH_dq0_1 + dH_dq0_2 + dH_dq0_3) +
      F[0];  // dP0/dt = -dH/dX0 + F0
  dpdt[4] =
      -(1. / total_mass) * (dH_dq1_Newt + dH_dq1_1 + dH_dq1_2 + dH_dq1_3) +
      F[1];  // dP1/dt = -dH/dX1 + F1
  dpdt[5] =
      -(1. / total_mass) * (dH_dq2_Newt + dH_dq2_1 + dH_dq2_2 + dH_dq2_3) +
      F[2];  // dP2/dt = -dH/dX2 + F2
}

void BinaryWithGravitationalWaves::observer_vector(
    const BinaryWithGravitationalWaves::state_type& x, const double t) {
  past_time_.push_back(t);

  std::array<double, 3> x_cm = {
      (xcoord_right() * mass_right() + xcoord_left() * mass_left()) /
          total_mass,
      0., 0.};

  for (size_t i = 0; i < 3; ++i) {
    past_position_left_.at(i).push_back(x_cm.at(i) - mass_right() * x.at(i));
    past_position_right_.at(i).push_back(x_cm.at(i) + mass_left() * x.at(i));
  }
  for (size_t i = 3; i < 6; ++i) {
    past_momentum_left_.at(i - 3).push_back(-x.at(i) * reduced_mass);
    past_momentum_right_.at(i - 3).push_back(x.at(i) * reduced_mass);
  }

  state_type dxdt;
  hamiltonian_system(x, dxdt);

  for (size_t i = 0; i < 3; ++i) {
    past_dt_position_left_.at(i).push_back(-dxdt.at(i) * reduced_mass);
    past_dt_position_right_.at(i).push_back(dxdt.at(i) * reduced_mass);
  }
  for (size_t i = 3; i < 6; ++i) {
    past_dt_momentum_left_.at(i - 3).push_back(-dxdt.at(i) * reduced_mass);
    past_dt_momentum_right_.at(i - 3).push_back(dxdt.at(i) * reduced_mass);
  }
}

void BinaryWithGravitationalWaves::integrate_hamiltonian_system() {
  BinaryWithGravitationalWaves::state_type ini = {
      initial_state_position.at(0),
      initial_state_position.at(1),
      initial_state_position.at(2),
      initial_state_momentum.at(0),
      initial_state_momentum.at(1),
      initial_state_momentum.at(2)};  // initial conditions

  auto hamiltonian_system_lambda = [this](auto&& PH1, auto&& PH2,
                                          const double /*t*/) {
    hamiltonian_system(std::forward<decltype(PH1)>(PH1),
                       std::forward<decltype(PH2)>(PH2));
  };

  auto observer = [this](auto&& PH1, auto&& PH2) {
    observer_vector(std::forward<decltype(PH1)>(PH1),
                    std::forward<decltype(PH2)>(PH2));
  };

  // Integrate the Hamiltonian system
  boost::numeric::odeint::integrate_const(
      boost::numeric::odeint::runge_kutta4<
          BinaryWithGravitationalWaves::state_type>(),
      hamiltonian_system_lambda, ini, initial_time, final_time, -time_step,
      observer);
}

void BinaryWithGravitationalWaves::write_evolution_to_file() const {
  if (write_evolution_option()) {
    std::ofstream file;
    file.open("PastHistoryEvolution.txt");
    file << "time, position_left_x, position_left_y, position_left_z, "
            "momentum_left_x, momentum_left_y, momentum_left_z, "
            "position_right_x, position_right_y, position_right_z, "
            "momentum_right_x, momentum_right_y, momentum_right_z, "
            "dt_momentum_left_x, dt_momentum_left_y, dt_momentum_left_z, "
            "dt_momentum_right_x, dt_momentum_right_y, dt_momentum_right_z, "
         << std::endl;
    for (size_t i = 0; i < number_of_steps; i++) {
      file << past_time_.at(i) << ", ";
      for (size_t j = 0; j < 3; ++j) {
        file << past_position_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_momentum_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_position_right_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_momentum_right_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_dt_momentum_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_dt_momentum_right_.at(j).at(i) << ", ";
      }
      file << std::endl;
    }
    file.close();
  }
}

PUP::able::PUP_ID BinaryWithGravitationalWaves::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    BinaryWithGravitationalWavesVariables<DataVector>::Cache>;
