// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::Solutions::tov_detail {

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> /* cache */,
    Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  if (tov_star.radial_solution().coordinate_system() ==
      TovCoordinates::Isotropic) {
    get<0, 0>(*conformal_metric) = 1.;
    get<1, 1>(*conformal_metric) = 1.;
    get<2, 2>(*conformal_metric) = 1.;
    get<0, 1>(*conformal_metric) = 0.;
    get<0, 2>(*conformal_metric) = 0.;
    get<1, 2>(*conformal_metric) = 0.;
  } else {
    *conformal_metric =
        get_tov_var(gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>{});
  }
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
    const gsl::not_null<Cache*> /* cache */,
    Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  if (tov_star.radial_solution().coordinate_system() ==
      TovCoordinates::Isotropic) {
    get<0, 0>(*inv_conformal_metric) = 1.;
    get<1, 1>(*inv_conformal_metric) = 1.;
    get<2, 2>(*inv_conformal_metric) = 1.;
    get<0, 1>(*inv_conformal_metric) = 0.;
    get<0, 2>(*inv_conformal_metric) = 0.;
    get<1, 2>(*inv_conformal_metric) = 0.;
  } else {
    *inv_conformal_metric = get_tov_var(
        gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>{});
  }
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> /* cache */,
    ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  if (tov_star.radial_solution().coordinate_system() ==
      TovCoordinates::Isotropic) {
    std::fill(deriv_conformal_metric->begin(), deriv_conformal_metric->end(),
              0.);
  } else {
    *deriv_conformal_metric = get_tov_var(
        ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
                      tmpl::size_t<3>, Frame::Inertial>{});
  }
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType> /*meta*/) const {
  std::fill(extrinsic_curvature->begin(), extrinsic_curvature->end(), 0.);
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  get(*trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /* cache */,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /* cache */,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  std::fill(deriv_trace_extrinsic_curvature->begin(),
            deriv_trace_extrinsic_curvature->end(), 0.);
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<Cache*> /* cache */,
    Tags::ConformalFactor<DataType> /*meta*/) const {
  if (tov_star.radial_solution().coordinate_system() ==
      TovCoordinates::Isotropic) {
    *conformal_factor = get_tov_var(
        RelativisticEuler::Solutions::tov_detail::Tags::ConformalFactor<
            DataType>{});
  } else {
    get(*conformal_factor) = 1.;
  }
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_conformal_factor,
    const gsl::not_null<Cache*> /* cache */,
    ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  if (tov_star.radial_solution().coordinate_system() ==
      TovCoordinates::Isotropic) {
    for (size_t i = 0; i < get_size(radius); ++i) {
      if (get_element(radius, i) > 1.e-14) {
        using tag =
            RelativisticEuler::Solutions::tov_detail::Tags::DrConformalFactor<
                double>;
        const tnsr::I<double, 3> x_i{
            {{get_element(get<0>(x), i), get_element(get<1>(x), i),
              get_element(get<2>(x), i)}}};
        const double dr_conformal_factor =
            get(get<tag>(tov_star.variables(x_i, 0., tmpl::list<tag>{})));
        get_element(get<0>(*deriv_conformal_factor), i) =
            dr_conformal_factor / get_element(radius, i);
      } else {
        get_element(get<0>(*deriv_conformal_factor), i) = 0.;
      }
    }
    get<1>(*deriv_conformal_factor) = get<0>(*deriv_conformal_factor);
    get<2>(*deriv_conformal_factor) = get<0>(*deriv_conformal_factor);
    get<0>(*deriv_conformal_factor) *= get<0>(x);
    get<1>(*deriv_conformal_factor) *= get<1>(x);
    get<2>(*deriv_conformal_factor) *= get<2>(x);
  } else {
    get<0>(*deriv_conformal_factor) = 0.;
    get<1>(*deriv_conformal_factor) = 0.;
    get<2>(*deriv_conformal_factor) = 0.;
  }
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  *lapse = get_tov_var(gr::Tags::Lapse<DataType>{});
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
    const gsl::not_null<Cache*> /* cache */,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  *deriv_lapse = get_tov_var(::Tags::deriv<gr::Tags::Lapse<DataType>,
                                           tmpl::size_t<3>, Frame::Inertial>{});
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    Tags::LapseTimesConformalFactor<DataType> /*meta*/) const {
  const auto& conformal_factor =
      get(cache->get_var(*this, Tags::ConformalFactor<DataType>{}));
  const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
  get(*lapse_times_conformal_factor) = lapse * conformal_factor;
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*>
        deriv_lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor =
      get(cache->get_var(*this, Tags::ConformalFactor<DataType>{}));
  const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
  const auto& deriv_conformal_factor =
      cache->get_var(*this, ::Tags::deriv<Tags::ConformalFactor<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& deriv_lapse =
      cache->get_var(*this, ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  get<0>(*deriv_lapse_times_conformal_factor) =
      lapse * get<0>(deriv_conformal_factor) +
      get<0>(deriv_lapse) * conformal_factor;
  get<1>(*deriv_lapse_times_conformal_factor) =
      lapse * get<1>(deriv_conformal_factor) +
      get<1>(deriv_lapse) * conformal_factor;
  get<2>(*deriv_lapse_times_conformal_factor) =
      lapse * get<2>(deriv_conformal_factor) +
      get<2>(deriv_lapse) * conformal_factor;
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /* cache */,
    Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /* cache */,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
    const gsl::not_null<Cache*> /* cache */,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
    const gsl::not_null<Cache*> /* cache */,
    Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(shift_strain->begin(), shift_strain->end(), 0.);
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>,
                        ConformalMatterScale> /*meta*/) const {
  *energy_density = get_tov_var(hydro::Tags::RestMassDensity<DataType>{});
  get(*energy_density) *=
      get(get_tov_var(hydro::Tags::SpecificEnthalpy<DataType>{}));
  get(*energy_density) -= get(get_tov_var(hydro::Tags::Pressure<DataType>{}));
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> stress_trace,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>,
                        ConformalMatterScale> /*meta*/) const {
  get(*stress_trace) = 3. * get(get_tov_var(hydro::Tags::Pressure<DataType>{}));
}

template <typename DataType>
void TovVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
                        ConformalMatterScale> /*meta*/) const {
  std::fill(momentum_density->begin(), momentum_density->end(), 0.);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template class TovVariables<DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts::Solutions::tov_detail

PUP::able::PUP_ID Xcts::Solutions::TovStar::my_PUP_ID = 0;  // NOLINT

// Instantiate implementations for common variables
template class Xcts::Solutions::CommonVariables<
    double, typename Xcts::Solutions::tov_detail::TovVariablesCache<double>>;
template class Xcts::Solutions::CommonVariables<
    DataVector,
    typename Xcts::Solutions::tov_detail::TovVariablesCache<DataVector>>;
template class Xcts::AnalyticData::CommonVariables<
    double, typename Xcts::Solutions::tov_detail::TovVariablesCache<double>>;
template class Xcts::AnalyticData::CommonVariables<
    DataVector,
    typename Xcts::Solutions::tov_detail::TovVariablesCache<DataVector>>;
