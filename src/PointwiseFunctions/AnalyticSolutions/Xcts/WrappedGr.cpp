// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/CcsnCollapse.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/PrimitiveSources.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::Solutions {
namespace detail {

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<DataType, Dim> /*meta*/) const {
  *spatial_metric = get<gr::Tags::SpatialMetric<DataType, Dim>>(gr_solution);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::InverseSpatialMetric<DataType, Dim> /*meta*/) const {
  *inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, Dim>>(gr_solution);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {
  *deriv_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                        tmpl::size_t<Dim>, Frame::Inertial>>(gr_solution);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  const auto& conformal_factor =
      cache->get_var(*this, Xcts::Tags::ConformalFactor<DataType>{});
  *conformal_metric = get<gr::Tags::SpatialMetric<DataType, Dim>>(gr_solution);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric->get(i, j) /= pow<4>(get(conformal_factor));
    }
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> inv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  const auto& conformal_factor =
      cache->get_var(*this, Xcts::Tags::ConformalFactor<DataType>{});
  *inv_conformal_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, Dim>>(gr_solution);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      inv_conformal_metric->get(i, j) *= pow<4>(get(conformal_factor));
    }
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(*this, Xcts::Tags::ConformalFactor<DataType>{});
  const auto& deriv_conformal_factor = cache->get_var(
      *this, ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& conformal_metric = cache->get_var(
      *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  *deriv_conformal_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                        tmpl::size_t<Dim>, Frame::Inertial>>(gr_solution);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        deriv_conformal_metric->get(i, j, k) /= pow<4>(get(conformal_factor));
        deriv_conformal_metric->get(i, j, k) -= 4. / get(conformal_factor) *
                                                conformal_metric.get(j, k) *
                                                deriv_conformal_factor.get(i);
      }
    }
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, Dim>>(gr_solution);
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, Dim>>(gr_solution);
  trace(trace_extrinsic_curvature, extrinsic_curvature, inv_spatial_metric);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactor<DataType> /*meta*/) const {
  get(*conformal_factor) = 1.;
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> conformal_factor_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(conformal_factor_gradient->begin(),
            conformal_factor_gradient->end(), 0.);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  *lapse = get<gr::Tags::Lapse<DataType>>(gr_solution);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LapseTimesConformalFactor<DataType> /*meta*/) const {
  *lapse_times_conformal_factor =
      cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& conformal_factor =
      cache->get_var(*this, Xcts::Tags::ConformalFactor<DataType>{});
  get(*lapse_times_conformal_factor) *= get(conformal_factor);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  *lapse_times_conformal_factor_minus_one =
      cache->get_var(*this, Xcts::Tags::LapseTimesConformalFactor<DataType>{});
  get(*lapse_times_conformal_factor_minus_one) -= 1.;
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*>
        lapse_times_conformal_factor_gradient,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(*this, Xcts::Tags::ConformalFactor<DataType>{});
  const auto& deriv_conformal_factor = cache->get_var(
      *this, ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>{});
  *lapse_times_conformal_factor_gradient =
      get<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                        Frame::Inertial>>(gr_solution);
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(gr_solution);
  for (size_t i = 0; i < Dim; ++i) {
    lapse_times_conformal_factor_gradient->get(i) *= get(conformal_factor);
    lapse_times_conformal_factor_gradient->get(i) +=
        get(lapse) * deriv_conformal_factor.get(i);
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  *shift_excess = get<gr::Tags::Shift<DataType, Dim>>(gr_solution);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  *deriv_shift_excess =
      get<::Tags::deriv<gr::Tags::Shift<DataType, Dim>, tmpl::size_t<Dim>,
                        Frame::Inertial>>(gr_solution);
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::ExtrinsicCurvature<DataType, 3> /*meta*/) const {
  *extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, Dim>>(gr_solution);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    [[maybe_unused]] const gsl::not_null<Scalar<DataType>*>
        magnetic_field_dot_spatial_velocity,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::MagneticFieldDotSpatialVelocity<DataType> /*meta*/) const {
  if constexpr (HasMhd) {
    const auto& spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataType, Dim, Frame::Inertial>>(
            hydro_solution);
    const auto& magnetic_field =
        get<hydro::Tags::MagneticField<DataType, Dim, Frame::Inertial>>(
            hydro_solution);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataType, Dim>>(gr_solution);
    tenex::evaluate(magnetic_field_dot_spatial_velocity,
                    magnetic_field(ti::I) * spatial_velocity(ti::J) *
                        spatial_metric(ti::i, ti::j));
  } else {
    ERROR(
        "The 'MagneticFieldDotSpatialVelocity' should not be needed in vacuum "
        "GR.");
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    [[maybe_unused]] const gsl::not_null<Scalar<DataType>*>
        comoving_magnetic_field_squared,
    [[maybe_unused]] const gsl::not_null<Cache*> cache,
    hydro::Tags::ComovingMagneticFieldSquared<DataType> /*meta*/) const {
  if constexpr (HasMhd) {
    const auto lorentz_factor = provider_lorentz_factor();
    const auto& magnetic_field =
        get<hydro::Tags::MagneticField<DataType, Dim, Frame::Inertial>>(
            hydro_solution);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataType, Dim>>(gr_solution);
    const auto magnetic_field_squared =
        tenex::evaluate(magnetic_field(ti::I) * magnetic_field(ti::J) *
                        spatial_metric(ti::i, ti::j));
    const auto& magnetic_field_dot_spatial_velocity = cache->get_var(
        *this, hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>{});
    hydro::comoving_magnetic_field_squared(
        comoving_magnetic_field_squared, magnetic_field_squared,
        magnetic_field_dot_spatial_velocity, lorentz_factor);
  } else {
    ERROR(
        "The 'ComovingMagneticFieldSquared' should not be needed in vacuum "
        "GR.");
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType, bool HasMhd>
const typename WrappedGrVariables<DataType, HasMhd>::PrimitiveSources&
WrappedGrVariables<DataType, HasMhd>::primitive_sources() const
  requires(HasMhd)
{
  if (not primitive_sources_.has_value()) {
    const auto& rho =
        get<hydro::Tags::RestMassDensity<DataType>>(hydro_solution);
    const auto& epsilon =
        get<hydro::Tags::SpecificInternalEnergy<DataType>>(hydro_solution);
    const auto& pressure = get<hydro::Tags::Pressure<DataType>>(hydro_solution);
    const auto& velocity =
        get<hydro::Tags::SpatialVelocity<DataType, Dim>>(hydro_solution);
    const auto& magnetic_field =
        get<hydro::Tags::MagneticField<DataType, Dim>>(hydro_solution);
    const auto& metric =
        get<gr::Tags::SpatialMetric<DataType, Dim>>(gr_solution);
    std::array<DataType, 3> v{};
    std::array<DataType, 3> b{};
    std::array<std::array<DataType, 3>, 3> g{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(v, i) = velocity.get(i);
      gsl::at(b, i) = magnetic_field.get(i);
      for (size_t j = 0; j < Dim; ++j) {
        gsl::at(gsl::at(g, i), j) = metric.get(i, j);
      }
    }
    // Direct primitive H is intentional, including below CcsnCollapse's
    // historical h-denominator floor. The library validates the primitives;
    // the wrapper neither repairs them nor reconstructs epsilon from h.
    // Store only a complete successful result; errors propagate to the
    // caller.
    primitive_sources_.emplace(hydro::primitive_sources::joint_sources(
        get(rho), get(epsilon), get(pressure), v, b, g));
  }
  return *primitive_sources_;
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  if constexpr (HasMhd) {
    get(*energy_density) = primitive_sources()[0];
  } else {
    std::fill(energy_density->begin(), energy_density->end(), 0.);
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<Scalar<DataType>*> stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  if constexpr (HasMhd) {
    get(*stress_trace) = primitive_sources()[4];
  } else {
    std::fill(stress_trace->begin(), stress_trace->end(), 0.);
  }
}

template <typename DataType, bool HasMhd>
void WrappedGrVariables<DataType, HasMhd>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
    const {
  if constexpr (HasMhd) {
    const auto& sources = primitive_sources();
    for (size_t i = 0; i < Dim; ++i) {
      momentum_density->get(i) = gsl::at(sources, i + 1);
    }
  } else {
    std::fill(momentum_density->begin(), momentum_density->end(), 0.);
  }
}

template class WrappedGrVariables<double, false>;
template class WrappedGrVariables<DataVector, false>;
template class WrappedGrVariables<double, true>;
template class WrappedGrVariables<DataVector, true>;

}  // namespace detail

template <typename GrSolution, bool HasMhd>
PUP::able::PUP_ID WrappedGr<GrSolution, HasMhd>::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::Solutions

// Instantiate implementations for common variables
template class Xcts::Solutions::CommonVariables<
    double,
    typename Xcts::Solutions::detail::WrappedGrVariables<double, false>::Cache>;
template class Xcts::Solutions::CommonVariables<
    DataVector, typename Xcts::Solutions::detail::WrappedGrVariables<
                    DataVector, false>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    double,
    typename Xcts::Solutions::detail::WrappedGrVariables<double, false>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::Solutions::detail::WrappedGrVariables<
                    DataVector, false>::Cache>;

#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_GR(_, data) \
  template class Xcts::Solutions::WrappedGr<STYPE(data), false>;
#define INSTANTIATE_GRMHD(_, data) \
  template class Xcts::Solutions::WrappedGr<STYPE(data), true>;

GENERATE_INSTANTIATIONS(INSTANTIATE_GR, (gr::Solutions::KerrSchild,
                                         gr::Solutions::SphericalKerrSchild,
                                         gr::Solutions::HarmonicSchwarzschild))
GENERATE_INSTANTIATIONS(INSTANTIATE_GRMHD,
                        (grmhd::AnalyticData::CcsnCollapse,
                         grmhd::AnalyticData::MagnetizedTovStar,
                         RelativisticEuler::Solutions::RotatingStar))

#undef STYPE
#undef INSTANTIATE_GR
#undef INSTANTIATE_GRMHD
