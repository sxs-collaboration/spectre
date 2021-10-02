// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"                  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <complex>

namespace RelativisticEuler::Solutions {

template <typename RadialSolution>
TovStar<RadialSolution>::TovStar(const double central_rest_mass_density,
                                 const double polytropic_constant,
                                 const double polytropic_exponent)
    : central_rest_mass_density_(central_rest_mass_density),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_} {}

template <typename RadialSolution>
void TovStar<RadialSolution>::pup(PUP::er& p) {
  p | central_rest_mass_density_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
}

template <>
const gr::Solutions::TovSolution&
TovStar<gr::Solutions::TovSolution>::radial_tov_solution() const {
  static const gr::Solutions::TovSolution solution(
      equation_of_state_, central_rest_mass_density_, 0.0);
  return solution;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  return radial_vars.rest_mass_density;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  return radial_vars.pressure;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  return radial_vars.specific_internal_energy;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  return radial_vars.specific_enthalpy;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<Scalar<DataType>>(x, 1.0);
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<Scalar<DataType>>(x, 0.0);
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Lapse<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  return Scalar<DataType>{exp(radial_vars.metric_time_potential)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    typename TovStar<RadialSolution>::template DerivLapse<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<typename TovStar<RadialSolution>::template DerivLapse<
        DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  tnsr::i<DataType, 3, Frame::Inertial> d_lapse{
      exp(radial_vars.metric_time_potential) *
      radial_vars.dr_metric_time_potential / radial_vars.radial_coordinate};
  for (size_t i = 0; i < 3; ++i) {
    d_lapse.get(i) *= x.get(i);
  }
  return d_lapse;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    typename TovStar<RadialSolution>::template DerivShift<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<typename TovStar<RadialSolution>::template DerivShift<
        DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<tnsr::iJ<DataType, 3>>(x, 0.0);
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  tnsr::ii<DataType, 3, Frame::Inertial> spatial_metric{
      (exp(2.0 * radial_vars.metric_radial_potential) -
       exp(2.0 * radial_vars.metric_angular_potential)) /
      square(radial_vars.radial_coordinate)};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      spatial_metric.get(i, j) *= x.get(i) * x.get(j);
    }
    spatial_metric.get(i, i) += exp(2.0 * radial_vars.metric_angular_potential);
  }
  return spatial_metric;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    typename TovStar<RadialSolution>::template DerivSpatialMetric<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<typename TovStar<RadialSolution>::template DerivSpatialMetric<
        DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  tnsr::ijj<DataType, 3, Frame::Inertial> deriv_spatial_metric{
      2.0 *
      (exp(2.0 * radial_vars.metric_radial_potential) *
           (radial_vars.dr_metric_radial_potential -
            1.0 / radial_vars.radial_coordinate) -
       exp(2.0 * radial_vars.metric_angular_potential) *
           (radial_vars.dr_metric_angular_potential -
            1.0 / radial_vars.radial_coordinate)) /
      cube(radial_vars.radial_coordinate)};
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        deriv_spatial_metric.get(k, i, j) *= x.get(k) * x.get(i) * x.get(j);
        if (k == i) {
          deriv_spatial_metric.get(k, i, j) +=
              (exp(2.0 * radial_vars.metric_radial_potential) -
               exp(2.0 * radial_vars.metric_angular_potential)) /
              square(radial_vars.radial_coordinate) * x.get(j);
        }
        if (k == j) {
          deriv_spatial_metric.get(k, i, j) +=
              (exp(2.0 * radial_vars.metric_radial_potential) -
               exp(2.0 * radial_vars.metric_angular_potential)) /
              square(radial_vars.radial_coordinate) * x.get(i);
        }
        if (i == j) {
          deriv_spatial_metric.get(k, i, j) +=
              2.0 * exp(2.0 * radial_vars.metric_angular_potential) *
              radial_vars.dr_metric_angular_potential * x.get(k) /
              radial_vars.radial_coordinate;
        }
      }
    }
  }
  return deriv_spatial_metric;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  return Scalar<DataType>{exp(radial_vars.metric_radial_potential +
                              2.0 * radial_vars.metric_angular_potential)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const {
  tnsr::II<DataType, 3, Frame::Inertial> g{
      exp(-2.0 * radial_vars.metric_radial_potential) -
      exp(-2.0 * radial_vars.metric_angular_potential)};
  for (size_t d0 = 0; d0 < 3; ++d0) {
    for (size_t d1 = d0; d1 < 3; ++d1) {
      g.get(d0, d1) *=
          x.get(d0) * x.get(d1) / square(radial_vars.radial_coordinate);
    }
    g.get(d0, d0) += exp(-2.0 * radial_vars.metric_angular_potential);
  }
  return g;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<tnsr::ii<DataType, 3>>(x, 0.0);
}

template <typename RadialSolution>
template <typename DataType, typename Tag>
tuples::TaggedTuple<::Tags::dt<Tag>> TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x, tmpl::list<::Tags::dt<Tag>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const {
  return make_with_value<typename ::Tags::dt<Tag>::type>(get<0>(x), 0.0);
}

template <typename LocalRadialSolution>
bool operator==(const TovStar<LocalRadialSolution>& lhs,
                const TovStar<LocalRadialSolution>& rhs) {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_density_ == rhs.central_rest_mass_density_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

template <typename RadialSolution>
bool operator!=(const TovStar<RadialSolution>& lhs,
                const TovStar<RadialSolution>& rhs) {
  return not(lhs == rhs);
}

#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE_VARS(_, data)                                              \
  template tuples::TaggedTuple<hydro::Tags::RestMassDensity<DTYPE(data)>>      \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::RestMassDensity<DTYPE(data)>> /*meta*/,          \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>     \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> meta,             \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>             \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/,                 \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                        \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,   \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>        \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/,            \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DTYPE(data), 3>>   \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<hydro::Tags::MagneticField<DTYPE(data), 3>>     \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3>> /*meta*/,         \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      hydro::Tags::DivergenceCleaningField<DTYPE(data)>>                       \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/,  \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& /*x*/,                                    \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/,                       \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      typename TovStar<STYPE(data)>::template DerivLapse<DTYPE(data)>>         \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<typename TovStar<STYPE(data)>::template DerivLapse<DTYPE(     \
          data)>> /*meta*/,                                                    \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>                        \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& /*x*/,                                    \
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>> /*meta*/,   \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      typename TovStar<STYPE(data)>::template DerivShift<DTYPE(data)>>         \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<typename TovStar<STYPE(data)>::template DerivShift<DTYPE(     \
          data)>> /*meta*/,                                                    \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<3, Frame::Inertial, DTYPE(data)>>                \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& /*x*/,                                    \
      tmpl::list<                                                              \
          gr::Tags::SpatialMetric<3, Frame::Inertial, DTYPE(data)>> /*meta*/,  \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      typename TovStar<STYPE(data)>::template DerivSpatialMetric<DTYPE(data)>> \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<typename TovStar<STYPE(                                       \
          data)>::template DerivSpatialMetric<DTYPE(data)>> /*meta*/,          \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& /*x*/,                                    \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/,        \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DTYPE(data)>>         \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& /*x*/,                                    \
      tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame::Inertial,            \
                                                DTYPE(data)>> /*meta*/,        \
      const RadialVariables<DTYPE(data)>& radial_vars) const;                  \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DTYPE(data)>>           \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& /*x*/,                                    \
      tmpl::list<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial,              \
                                              DTYPE(data)>> /*meta*/,          \
      const RadialVariables<DTYPE(data)>& radial_vars) const;

#define INSTANTIATE_DT_VARS(_, data)                                           \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/,           \
      const RadialVariables<DTYPE(data)>& /*radial_vars*/) const;              \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>>            \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>> /*meta*/,         \
      const RadialVariables<DTYPE(data)>& /*radial_vars*/) const;              \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DTYPE(data)>>>    \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::SpatialMetric<3, Frame::Inertial, DTYPE(data)>>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& /*radial_vars*/) const;

#define INSTANTIATE(_, data)                                 \
  template class TovStar<STYPE(data)>;                       \
  template bool operator==(const TovStar<STYPE(data)>& lhs,  \
                           const TovStar<STYPE(data)>& rhs); \
  template bool operator!=(const TovStar<STYPE(data)>& lhs,  \
                           const TovStar<STYPE(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE_VARS, (gr::Solutions::TovSolution),
                        (double, DataVector))
GENERATE_INSTANTIATIONS(INSTANTIATE_DT_VARS, (gr::Solutions::TovSolution),
                        (double, DataVector))
GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::TovSolution))

#undef DTYPE
#undef STYPE
#undef INSTANTIATE
#undef INSTANTIATE_VARS
}  // namespace RelativisticEuler::Solutions
