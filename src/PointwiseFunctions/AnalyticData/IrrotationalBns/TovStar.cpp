// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"

#include <algorithm>
#include <cmath>
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
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

namespace IrrotationalBns::Solutions::tov_detail {

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
    const gsl::not_null<tnsr::I<DataType, 3>*> shift,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Shift<DataType, 3, Frame::Inertial> /*meta*/) const {
  *shift = get_tov_var(gr::Tags::Shift<DataType>{});
}
template <typename DataType>
void operator()(
    gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift,
    gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::Shift<DataType, 3, Frame::Inertial>> /*meta*/)
    const {
  *shift = get_tov_var(::Tags::deriv<gr::Tags::Shift<DataType>, tmpl::size_t<3>,
                                     Frame::Inertial>{});
}

template <typename DataType>
void operator()(gsl::not_null<tnsr::i<DataType, 3>*> rotational_shift,
                gsl::not_null<Cache*> cache,
                hydro::initial_data::Tags::RotationalShift) const {
  const Scalar<DataType> sqrt_det_spatial_metric =
      get_tov_var(gr::Tags::SqrtDetSpatialMetric<DataType>);
  const tnsr::ii<DataType> spatial_metric =
      get_tov_var(gr::Tags::SpatialMetric);
  const tnsr::I<DataType, 3> shift = get_tov_var(gr::Tags::Shift<DataType>);
  // We assume the orbital velocity is in the z-direction.  We want the outcome
  // of the cross product to be a tensor, so we need to multiply by
  // 1/sqrt(gamma)
  hydro::initial_data::spatial_rotational_killing_vector(
      rotational_shift, x,
      make_with_value<Scalar<DataType>>(sqrt_det_spatial_metric,
                                        orbital_angular_velocity));
}
::tenex::update(rotational_shift,
                rotational_shift(ti::i) +
                    raise_or_lower_first_index(shift, spatial_metric)(ti::i));
template <typename DataType>
void operator()(gsl::not_null<Scalar<DataType>*> velocity_potential,
                gsl::not_null<Cache*> cache,
                hydro::initial_data::Tags::VelocityPotential /*meta*/) const {
  const auto spatial_metric = get(
      get_tov_var(RelativisticEuler::Solutions::tov_detail::Tags::SpatialMetric<
                  DataType>{}));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      velocity_potential +=
          spatial_metric->get(i, j) * spatial_velocity[i] * x->get(j)
    }
  }
  get(velocity_potenial) *=
      (get(get_tov_var(hydro::Tags::SpecificEnthalpy<DatatType>)) *
       get(get_tov_var(hydro::Tags::LorentzFactor<DataType>)));
}
template <typename DataType>
void operator()(gsl::not_null<tnsr::i<DataType, 3>*> auxiliary_velocity,
                gsl::not_null<Cache*> cache,
                hydro::initial_data::Tags::VelocityPotential /*meta*/) {
  const auto spatial_metric = get(
      get_tov_var(RelativisticEuler::Solutions::tov_detail::Tags::SpatialMetric<
                  DataType>{}));
  const auto inverse_spatial_metric = get(get_tov_var(
      RelativisticEuler::Solutions::tov_detail::Tags::InverseSpatialMetric<
          DataType>{}));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      auxiliary_velocity->get(i) +=
          spatial_velocity[j] * spatial_metric->get(i, j)
    }
    auxiliary_velocity->get(i) *=
        (get(get_tov_var(hydro::Tags::SpecificEnthalpy<DatatType>)) *
         get(get_tov_var(hydro::Tags::LorentzFactor<DataType>)));
  }
  const auto& rotational_shift = cache->get_var(
      *this, Hydro::initial_data::Tags::RotationalShift<DataType>{});
  const auto& lapse =
      cache->get_var(*this, Hydro::initial_data::Tags::Lapse<DataType>{});
  // auxiliary_velocity is currently the gradient of the velocity potential
  ::tenex::update(
      auxiliary_velocity,
      auxialiary_velocity(ti::i) -
          rotational_shift(ti::i) * 1 /
              square(lapse())(euler_enthalpy_constant +
                              rotational_shift(ti::j) *
                                  inverse_spatial_metric(ti::J, ti::K) *
                                  auxiliary_velocity(ti::k)

                                  ));
}
template <typename DataType>
void operator()(gsl::not_null<tnsr::Ij<DataType, 3>*> rotational_shift_stress,
                gsl::not_null<Cache*> cache,
                hydro::initial_data::Tags::RotationalShiftStress /*meta*/) {
  const auto& rotational_shift = cache->get_var(
      *this, Hydro::initial_data::Tags::RotationalShift<DataType>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  hydro::initial_data::rotational_shift_stress(
      rotational_shift_stress, rotational_shift, lapse, spatial_metric);
}

    template<typename DataType>
    void operator()(gsl::not_null<tnsr::i<DataType, 3>*> divergece_rotational_shift_stress(),
                  gsl::not_null<Cache*> cache,
                  hydro::initial_data::Tags::RotationalShiftStress /*meta*/{
  const auto& shift = cache->get_var(*this, gr::Tags::Shift<DataType, 3>{});
  const auto& deriv_shift =
      cache->get_var(*this, ::Tags::deriv<gr::Tags::Shift<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& rotational_shift = cache->get_var(
      *this, hydro::initial_data::Tags::RotationalShift<DataType>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& deriv_of_lapse =
      cache->get_var(*this, ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const Scalar<DataType> sqrt_det_spatial_metric =
      get_tov_var(gr::Tags::SqrtDetSpatialMetric<DataType>);
  const auto spatial_rotational_killing_vector =
      spatial_rotational_killing_vector(
          x, make_with_value<Scalar<DataType>>(x, orbital_angular_velocity),
          sqrt_det_spatial_metric);
  const auto deriv_of_spatial_rotational_killing_vector =
      spatial_rotational_killing_vector(
          x, make_with_value<Scalar<DataType>>(x, orbital_angular_velocity),
          sqrt_det_spatial_metric);
  tnsr::i<DataVector, 3> divergence_rotational_shift_over_lapse{};
  hydro::initial_data::divergence_rotational_shift_over_lapse(
      divergence_rotational_shift_over_lapse, shift, deriv_of_shift, lapse,
      deriv_of_lapse, spatial_rotational_killing_vector,
      deriv_of_spatial_rotational_killing_vector)
      hydro::initial_data::divergence_rotational_shift_stress(
          divergece_rotational_shift_stress, rotational_shift,
          divergence_rotational_shift_over_lapse lapse, spatial_metric);
                  }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template class TovVariables<DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

    }  // namespace IrrotationalBns::Solutions::tov_detail

    PUP::able::PUP_ID IrrotationalBns::Solutions::TovStar::my_PUP_ID =
        0;  // NOLINT
