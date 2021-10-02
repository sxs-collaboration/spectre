// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
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
  get<0>(*shift_background) = -angular_velocity * get<1>(x);
  get<1>(*shift_background) = angular_velocity * get<0>(x);
  get<2>(*shift_background) = 0.;
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
}

template <typename DataType>
void BinaryVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> longitudinal_shift_background,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial> /*meta*/) const {
  const auto& shift_background =
      cache->get_var(Tags::ShiftBackground<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_shift_background = cache->get_var(
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& conformal_metric =
      cache->get_var(Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_conformal_metric = cache->get_var(
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>{});
  auto shift_background_strain =
      make_with_value<tnsr::ii<DataType, Dim>>(x, 0.);
  Elasticity::strain(make_not_null(&shift_background_strain),
                     deriv_shift_background, conformal_metric,
                     deriv_conformal_metric, conformal_christoffel_first_kind,
                     shift_background);
  Xcts::longitudinal_operator(longitudinal_shift_background,
                              shift_background_strain, inv_conformal_metric);
}

template class BinaryVariables<DataVector>;

}  // namespace Xcts::AnalyticData::detail

template class Xcts::AnalyticData::CommonVariables<
    DataVector,
    typename Xcts::AnalyticData::detail::BinaryVariables<DataVector>::Cache>;
