// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts::AnalyticData::detail {

template <typename DataType>
void BinaryVariables<DataType>::add_deriv_of_window_function(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric)
    const {
  if (not falloff_widths.has_value()) {
    return;
  }
  const auto& flat_metric =
      get<Tags::ConformalMetric<DataType, 3, Frame::Inertial>>(flat_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        for (size_t object_index = 0; object_index < 2; ++object_index) {
          deriv_conformal_metric->get(i, j, k) -=
              2. * gsl::at(x_isolated, object_index).get(i) /
              square(gsl::at(*falloff_widths, object_index)) *
              gsl::at(windows, object_index) *
              (get<Tags::ConformalMetric<DataType, 3, Frame::Inertial>>(
                   gsl::at(isolated_vars, object_index))
                   .get(j, k) -
               flat_metric.get(j, k));
        }
      }
    }
  }
}

template <typename DataType>
void BinaryVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/) const {
  get<0>(*shift_background) = -angular_velocity * get<1>(x) +
                              expansion * get<0>(x) + linear_velocity[0];
  get<1>(*shift_background) =
      angular_velocity * get<0>(x) + expansion * get<1>(x) + linear_velocity[1];
  get<2>(*shift_background) = expansion * get<2>(x) + linear_velocity[2];
}

template <typename DataType>
void BinaryVariables<DataType>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
  get<1, 0>(*deriv_shift_background) = -angular_velocity;
  get<0, 1>(*deriv_shift_background) = angular_velocity;
  get<0, 0>(*deriv_shift_background) = expansion;
  get<1, 1>(*deriv_shift_background) = expansion;
  get<2, 2>(*deriv_shift_background) = expansion;
}

template <typename DataType>
void BinaryVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> longitudinal_shift_background,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial> /*meta*/) const {
  const auto& shift_background = cache->get_var(
      *this, Tags::ShiftBackground<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_shift_background = cache->get_var(
      *this,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& conformal_christoffel_second_kind = cache->get_var(
      *this,
      Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>{});
  Xcts::longitudinal_operator(longitudinal_shift_background, shift_background,
                              deriv_shift_background, inv_conformal_metric,
                              conformal_christoffel_second_kind);
}

template class BinaryVariables<DataVector>;

}  // namespace Xcts::AnalyticData::detail

template class Xcts::AnalyticData::CommonVariables<
    DataVector,
    typename Xcts::AnalyticData::detail::BinaryVariables<DataVector>::Cache>;
