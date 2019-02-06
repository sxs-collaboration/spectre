// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"

// We don't do anything complex, but IWYU thinks it's needed for some math
// functions
// IWYU pragma: no_include <complex>

/// \cond
namespace RelativisticEuler {
namespace Solutions {

FishboneMoncriefDisk::FishboneMoncriefDisk(
    const double bh_mass, const double bh_dimless_spin,
    const double inner_edge_radius, const double max_pressure_radius,
    const double polytropic_constant, const double polytropic_exponent) noexcept
    : bh_mass_(bh_mass),
      bh_spin_a_(bh_mass * bh_dimless_spin),
      inner_edge_radius_(bh_mass * inner_edge_radius),
      max_pressure_radius_(bh_mass * max_pressure_radius),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      background_spacetime_{
          bh_mass_, {{0.0, 0.0, bh_dimless_spin}}, {{0.0, 0.0, 0.0}}} {
  const double sqrt_m = sqrt(bh_mass_);
  const double a_sqrt_m = bh_spin_a_ * sqrt_m;
  const double& rmax = max_pressure_radius_;
  const double sqrt_rmax = sqrt(rmax);
  const double rmax_sqrt_rmax = rmax * sqrt_rmax;
  const double rmax_squared = square(rmax);
  angular_momentum_ =
      sqrt_m * (rmax_sqrt_rmax + a_sqrt_m) *
      (square(bh_spin_a_) - 2.0 * a_sqrt_m * sqrt_rmax + rmax_squared) /
      (2.0 * a_sqrt_m * rmax_sqrt_rmax +
       (rmax - 3.0 * bh_mass_) * rmax_squared);
}

void FishboneMoncriefDisk::pup(PUP::er& p) noexcept {
  p | bh_mass_;
  p | bh_spin_a_;
  p | inner_edge_radius_;
  p | max_pressure_radius_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | angular_momentum_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
DataType FishboneMoncriefDisk::sigma(const DataType& r_sqrd,
                                     const DataType& sin_theta_sqrd) const
    noexcept {
  return r_sqrd + square(bh_spin_a_) * (1.0 - sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::inv_ucase_a(const DataType& r_sqrd,
                                           const DataType& sin_theta_sqrd,
                                           const DataType& delta) const
    noexcept {
  const double a_sqrd = square(bh_spin_a_);
  const DataType r_sqrd_plus_a_sqrd = r_sqrd + a_sqrd;
  return 1.0 / (square(r_sqrd_plus_a_sqrd) - delta * a_sqrd * sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::four_velocity_t_sqrd(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd) const noexcept {
  const DataType delta =
      r_sqrd - 2.0 * bh_mass_ * sqrt(r_sqrd) + square(bh_spin_a_);
  const DataType prefactor = 0.5 / (inv_ucase_a(r_sqrd, sin_theta_sqrd, delta) *
                                    sigma(r_sqrd, sin_theta_sqrd));
  return prefactor *
         (1.0 + sqrt(1.0 + square(angular_momentum_) * delta /
                               (square(prefactor) * sin_theta_sqrd))) /
         delta;
}

template <typename DataType>
DataType FishboneMoncriefDisk::angular_velocity(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd) const noexcept {
  const DataType r = sqrt(r_sqrd);
  return inv_ucase_a(
             r_sqrd, sin_theta_sqrd,
             DataType{r_sqrd - 2.0 * bh_mass_ * r + square(bh_spin_a_)}) *
         (angular_momentum_ * sigma(r_sqrd, sin_theta_sqrd) /
              (four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd) * sin_theta_sqrd) +
          2.0 * bh_mass_ * r * bh_spin_a_);
}

template <typename DataType>
DataType FishboneMoncriefDisk::potential(const DataType& r_sqrd,
                                         const DataType& sin_theta_sqrd) const
    noexcept {
  return angular_momentum_ * angular_velocity(r_sqrd, sin_theta_sqrd) -
         log(sqrt(four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd)));
}

template <typename DataType, bool NeedSpacetime>
FishboneMoncriefDisk::IntermediateVariables<DataType, NeedSpacetime>::
    IntermediateVariables(const double bh_spin_a,
                          const gr::Solutions::KerrSchild& background_spacetime,
                          const tnsr::I<DataType, 3>& x, const double t,
                          size_t in_spatial_velocity_index,
                          size_t in_lorentz_factor_index) noexcept
    : spatial_velocity_index(in_spatial_velocity_index),
      lorentz_factor_index(in_lorentz_factor_index) {
  const double a_squared = bh_spin_a * bh_spin_a;
  sin_theta_squared = square(get<0>(x)) + square(get<1>(x));
  r_squared = 0.5 * (sin_theta_squared + square(get<2>(x)) - a_squared);
  r_squared += sqrt(square(r_squared) + a_squared * square(get<2>(x)));
  sin_theta_squared /= (r_squared + a_squared);

  if (NeedSpacetime) {
    kerr_schild_soln = background_spacetime.variables(
        x, t, gr::Solutions::KerrSchild::tags<DataType>{});
  }
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& vars,
    const size_t index) const noexcept {
  const auto specific_enthalpy = get<hydro::Tags::SpecificEnthalpy<DataType>>(
      variables(x, tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>>{}, vars,
                index));
  auto rest_mass_density = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&rest_mass_density, &specific_enthalpy, this ](
                const size_t s, const double /*potential_at_s*/) noexcept {
        get_element(get(rest_mass_density), s) =
            get(equation_of_state_.rest_mass_density_from_enthalpy(
                Scalar<double>{get_element(get(specific_enthalpy), s)}));
      });
  return {std::move(rest_mass_density)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& vars,
    const size_t /*index*/) const noexcept {
  const double inner_edge_potential =
      potential(square(inner_edge_radius_), 1.0);
  auto specific_enthalpy = make_with_value<Scalar<DataType>>(x, 1.0);
  variables_impl(vars,
                 [&specific_enthalpy, inner_edge_potential ](
                     const size_t s, const double potential_at_s) noexcept {
                   get_element(get(specific_enthalpy), s) =
                       exp(inner_edge_potential - potential_at_s);
                 });
  return {std::move(specific_enthalpy)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& vars,
    const size_t index) const noexcept {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars,
                index));
  auto pressure = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&pressure, &rest_mass_density, this ](
                const size_t s, const double /*potential_at_s*/) noexcept {
        get_element(get(pressure), s) =
            get(equation_of_state_.pressure_from_density(
                Scalar<double>{get_element(get(rest_mass_density), s)}));
      });
  return {std::move(pressure)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& vars,
    const size_t index) const noexcept {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars,
                index));
  auto specific_internal_energy = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&specific_internal_energy, &rest_mass_density, this ](
                const size_t s, const double /*potential_at_s*/) noexcept {
        get_element(get(specific_internal_energy), s) =
            get(equation_of_state_.specific_internal_energy_from_density(
                Scalar<double>{get_element(get(rest_mass_density), s)}));
      });
  return {std::move(specific_internal_energy)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/,
    const IntermediateVariables<DataType, true>& vars,
    const size_t /*index*/) const noexcept {
  auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  variables_impl(vars, [
    &spatial_velocity, &vars, &x, this
  ](const size_t s, const double /*potential_at_s*/) noexcept {
    const double ang_velocity = angular_velocity(
        get_element(vars.r_squared, s), get_element(vars.sin_theta_squared, s));

    auto transport_velocity = make_array<3>(0.0);
    transport_velocity[0] -= ang_velocity * get_element(x.get(1), s);
    transport_velocity[1] += ang_velocity * get_element(x.get(0), s);

    for (size_t i = 0; i < 3; ++i) {
      get_element(spatial_velocity.get(i), s) =
          (gsl::at(transport_velocity, i) +
           get_element(get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(
                           vars.kerr_schild_soln)
                           .get(i),
                       s)) /
          get_element(
              get(get<gr::Tags::Lapse<DataType>>(vars.kerr_schild_soln)), s);
    }
  });
  return {std::move(spatial_velocity)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
    const IntermediateVariables<DataType, true>& vars, const size_t index) const
    noexcept {
  const auto spatial_velocity = get<hydro::Tags::SpatialVelocity<DataType, 3>>(
      variables(x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{},
                vars, index));
  Scalar<DataType> lorentz_factor{
      1.0 /
      sqrt(1.0 - get(dot_product(
                     spatial_velocity, spatial_velocity,
                     get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(
                         vars.kerr_schild_soln))))};
  return {std::move(lorentz_factor)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& /*vars*/,
    const size_t /*index*/) const noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      x, 0.0)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& /*vars*/,
    const size_t /*index*/) const noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.0)};
}

template <typename DataType, bool NeedSpacetime, typename Func>
void FishboneMoncriefDisk::variables_impl(
    const IntermediateVariables<DataType, NeedSpacetime>& vars, Func f) const
    noexcept {
  const DataType& r_squared = vars.r_squared;
  const DataType& sin_theta_squared = vars.sin_theta_squared;
  const double inner_edge_potential =
      potential(square(inner_edge_radius_), 1.0);

  // fill the disk with matter
  for (size_t s = 0; s < get_size(r_squared); ++s) {
    const double r_squared_s = get_element(r_squared, s);
    const double sin_theta_squared_s = get_element(sin_theta_squared, s);

    // the disk won't extend closer to the axis than r sin theta = rin
    // so no need to evaluate the potential there
    if (sqrt(r_squared_s * sin_theta_squared_s) >= inner_edge_radius_) {
      const double potential_s = potential(r_squared_s, sin_theta_squared_s);
      // the fluid can only be where W(r, theta) < W_in
      if (potential_s < inner_edge_potential) {
        f(s, potential_s);
      }
    }
  }
}

bool operator==(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) noexcept {
  return lhs.bh_mass_ == rhs.bh_mass_ and lhs.bh_spin_a_ == rhs.bh_spin_a_ and
         lhs.inner_edge_radius_ == rhs.inner_edge_radius_ and
         lhs.max_pressure_radius_ == rhs.max_pressure_radius_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define NEED_SPACETIME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template class FishboneMoncriefDisk::IntermediateVariables<                 \
      DTYPE(data), NEED_SPACETIME(data)>;                                     \
  template tuples::TaggedTuple<hydro::Tags::RestMassDensity<DTYPE(data)>>     \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::RestMassDensity<DTYPE(data)>> /*meta*/,         \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                           \
      const size_t) const noexcept;                                           \
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>    \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> /*meta*/,        \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                           \
      const size_t) const noexcept;                                           \
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>            \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/,                \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                           \
      const size_t) const noexcept;                                           \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                       \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,  \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                           \
      const size_t) const noexcept;                                           \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::MagneticField<DTYPE(data), 3, Frame::Inertial>>            \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,                   \
                                            Frame::Inertial>> /*meta*/,       \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                           \
      const size_t) const noexcept;                                           \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::DivergenceCleaningField<DTYPE(data)>>                      \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/, \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                           \
      const size_t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (true, false))
#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                 \
  template tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> \
  FishboneMoncriefDisk::variables(                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                      \
      tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> /*meta*/,     \
      const FishboneMoncriefDisk::IntermediateVariables<DTYPE(data), true>&  \
          vars,                                                              \
      const size_t) const noexcept;                                          \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>      \
  FishboneMoncriefDisk::variables(                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                      \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/,          \
      const FishboneMoncriefDisk::IntermediateVariables<DTYPE(data), true>&  \
          vars,                                                              \
      const size_t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef NEED_SPACETIME
#undef INSTANTIATE
}  // namespace Solutions
}  // namespace RelativisticEuler
/// \endcond
