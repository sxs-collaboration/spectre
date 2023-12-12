// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/IrrotationalBns/TovStar.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/IrrotationalBns/Tags.hpp"
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

namespace IrrotationalBns::InitialData::tov_detail {

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  *lapse = get_tov_var(gr::Tags::Lapse<DataType>{});
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
    const gsl::not_null<Cache*> /* cache */,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  *deriv_lapse = get_tov_var(::Tags::deriv<gr::Tags::Lapse<DataType>,
                                           tmpl::size_t<3>, Frame::Inertial>{});
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
    gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<DataType, 3> /*meta*/) const {
  *spatial_metric = get_tov_var(gr::Tags::SpatialMetric<DataType, 3>{});
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift,
    const gsl::not_null<Cache*> /* cache */,
    gr::Tags::Shift<DataType, 3, Frame::Inertial> /*meta*/) const {
  *shift = get_tov_var(gr::Tags::Shift<DataType, 3>{});
}
template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift,
    gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  *deriv_shift = get_tov_var(::Tags::deriv<gr::Tags::Shift<DataType, 3>,
                                           tmpl::size_t<3>, Frame::Inertial>{});
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<tnsr::I<DataType, 3>*> rotational_shift,
    gsl::not_null<Cache*> /*cache*/,
    IrrotationalBns::Tags::RotationalShift<DataType> /*meta*/) const {
  const Scalar<DataType> sqrt_det_spatial_metric =
      get_tov_var(gr::Tags::SqrtDetSpatialMetric<DataType>{});
  const tnsr::I<DataType, 3> shift =
      get_tov_var(gr::Tags::Shift<DataType, 3>{});
  // We assume the orbital velocity is in the z-direction.  We want the outcome
  // of the cross product to be a tensor, so we need to multiply by
  // 1/sqrt(gamma)
  tnsr::I<DataType, 3> rotational_killing_vector =
      hydro::initial_data::spatial_rotational_killing_vector(
          x,
          make_with_value<Scalar<DataType>>(sqrt_det_spatial_metric,
                                            orbital_angular_velocity),
          sqrt_det_spatial_metric);

  ::tenex::evaluate<ti::I>(rotational_shift,
                           (rotational_killing_vector(ti::I) + shift(ti::I)));
}
template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<Scalar<DataType>*> velocity_potential,
    gsl::not_null<Cache*> /*cache*/,
    IrrotationalBns::Tags::VelocityPotential<DataType> /*meta*/) const {
  const auto spatial_metric =
      get_tov_var(gr::Tags::SpatialMetric<DataType, 3>{});
  std::array<double, 3> star_velocity{
      -star_center[1] * orbital_angular_velocity,
      star_center[0] * orbital_angular_velocity, 0.0};
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      // The velocity of the star is taken to be a constant as a guess
      velocity_potential->get() +=
          (spatial_metric.get(i, j)) * gsl::at(star_velocity, i) * x.get(j);
    }
  }
  velocity_potential->get() *=
      (get(get_tov_var(hydro::Tags::SpecificEnthalpy<DataType>{})) *
       get(get_tov_var(hydro::Tags::LorentzFactor<DataType>{})));
}
template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<tnsr::i<DataType, 3>*> auxiliary_velocity,
    gsl::not_null<Cache*> cache,
    IrrotationalBns::Tags::AuxiliaryVelocity<DataType> /*meta*/) const {
  const auto spatial_metric =
      get_tov_var(gr::Tags::SpatialMetric<DataType, 3>{});
  std::array<double, 3> star_velocity{
      -star_center[1] * orbital_angular_velocity,
      star_center[0] * orbital_angular_velocity, 0.0};
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      auxiliary_velocity->get(i) +=
          gsl::at(star_velocity, j) * spatial_metric.get(i, j);
    }

    auxiliary_velocity->get(i) *=
        (get(get_tov_var(hydro::Tags::SpecificEnthalpy<DataType>{})) *
         get(get_tov_var(hydro::Tags::LorentzFactor<DataType>{})));
  }
  const auto& rotational_shift =
      cache->get_var(*this, IrrotationalBns::Tags::RotationalShift<DataType>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  // auxiliary_velocity is currently the gradient of the velocity potential
  // This equation is missing a factor of lapse^2
  ::tenex::evaluate<ti::i>(
      auxiliary_velocity,
      (*auxiliary_velocity)(ti::i)-spatial_metric(ti::i, ti::j) *
          rotational_shift(ti::J) *
          (euler_enthalpy_constant +
           rotational_shift(ti::K) * (*auxiliary_velocity)(ti::k)));
}
template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<tnsr::Ij<DataType, 3>*> rotational_shift_stress,
    gsl::not_null<Cache*> cache,
    IrrotationalBns::Tags::RotationalShiftStress<DataType> /*meta*/) const {
  const auto& rotational_shift =
      cache->get_var(*this, IrrotationalBns::Tags::RotationalShift<DataType>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  hydro::initial_data::rotational_shift_stress(
      rotational_shift_stress, rotational_shift, lapse, spatial_metric);
}

template <typename DataType, StarRegion Region>
void TovVariables<DataType, Region>::operator()(
    gsl::not_null<tnsr::i<DataType, 3>*> divergence_rotational_shift_stress,
    gsl::not_null<Cache*> cache,
    IrrotationalBns::Tags::DivergenceRotationalShiftStress<DataType> /*meta*/)
    const {
  const auto& deriv_shift =
      cache->get_var(*this, ::Tags::deriv<gr::Tags::Shift<DataType, 3>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& rotational_shift =
      cache->get_var(*this, IrrotationalBns::Tags::RotationalShift<DataType>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& deriv_of_lapse =
      cache->get_var(*this, ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const Scalar<DataType> sqrt_det_spatial_metric =
      get_tov_var(gr::Tags::SqrtDetSpatialMetric<DataType>{});
  const auto deriv_of_spatial_rotational_killing_vector =
      hydro::initial_data::derivative_spatial_rotational_killing_vector(
          x, make_with_value<Scalar<DataType>>(x, orbital_angular_velocity),
          sqrt_det_spatial_metric);
  tnsr::iJ<DataType, 3> derivative_rotational_shift_over_lapse{};
  hydro::initial_data::derivative_rotational_shift_over_lapse(
      make_not_null(&derivative_rotational_shift_over_lapse), rotational_shift,
      deriv_shift, lapse, deriv_of_lapse,
      deriv_of_spatial_rotational_killing_vector);
  hydro::initial_data::divergence_rotational_shift_stress(
      divergence_rotational_shift_stress, rotational_shift,
      derivative_rotational_shift_over_lapse, lapse, spatial_metric);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define REGION(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class TovVariables<DTYPE(data), REGION(data)>;

GENERATE_INSTANTIATIONS(
    INSTANTIATE, (double, DataVector),
    (RelativisticEuler::Solutions::tov_detail::StarRegion::Interior,
     RelativisticEuler::Solutions::tov_detail::StarRegion::Exterior))

#undef DTYPE
#undef INSTANTIATE

}  // namespace IrrotationalBns::InitialData::tov_detail

PUP::able::PUP_ID IrrotationalBns::InitialData::TovStar::my_PUP_ID =
    0;  // NOLINT
