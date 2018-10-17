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
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace RelativisticEuler {
namespace Solutions {

FishboneMoncriefDisk::FishboneMoncriefDisk(
    const double black_hole_mass, const double black_hole_spin,
    const double inner_edge_radius, const double max_pressure_radius,
    const double polytropic_constant, const double polytropic_exponent) noexcept
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      inner_edge_radius_(inner_edge_radius),
      max_pressure_radius_(max_pressure_radius),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      background_spacetime_{
          black_hole_mass_, {{0.0, 0.0, black_hole_spin_}}, {{0.0, 0.0, 0.0}}} {
}

void FishboneMoncriefDisk::pup(PUP::er& p) noexcept {
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | inner_edge_radius_;
  p | max_pressure_radius_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
DataType FishboneMoncriefDisk::sigma(const DataType& r_sqrd,
                                     const DataType& sin_theta_sqrd) const
    noexcept {
  return r_sqrd + square(black_hole_spin_) * (1.0 - sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::inv_ucase_a(const DataType& r_sqrd,
                                           const DataType& sin_theta_sqrd,
                                           const DataType& delta) const
    noexcept {
  const double a_sqrd = square(black_hole_spin_);
  const DataType r_sqrd_plus_a_sqrd = r_sqrd + a_sqrd;
  return 1.0 / (square(r_sqrd_plus_a_sqrd) - delta * a_sqrd * sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::four_velocity_t_sqrd(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd,
    const double angular_momentum) const noexcept {
  const DataType delta =
      r_sqrd - 2.0 * black_hole_mass_ * sqrt(r_sqrd) + square(black_hole_spin_);
  const DataType prefactor = 0.5 / (inv_ucase_a(r_sqrd, sin_theta_sqrd, delta) *
                                    sigma(r_sqrd, sin_theta_sqrd));
  return prefactor * (1.0 + sqrt(1.0 +
                                 square(angular_momentum) * delta /
                                     (square(prefactor) * sin_theta_sqrd))) /
         delta;
}

template <typename DataType>
DataType FishboneMoncriefDisk::angular_velocity(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd,
    const double angular_momentum) const noexcept {
  const DataType r = sqrt(r_sqrd);
  return inv_ucase_a(r_sqrd, sin_theta_sqrd,
                     DataType{r_sqrd - 2.0 * black_hole_mass_ * r +
                              square(black_hole_spin_)}) *
         (angular_momentum * sigma(r_sqrd, sin_theta_sqrd) /
              (four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd, angular_momentum) *
               sin_theta_sqrd) +
          2.0 * black_hole_mass_ * r * black_hole_spin_);
}

template <typename DataType>
DataType FishboneMoncriefDisk::potential(const DataType& r_sqrd,
                                         const DataType& sin_theta_sqrd,
                                         const double angular_momentum) const
    noexcept {
  return angular_momentum *
             angular_velocity(r_sqrd, sin_theta_sqrd, angular_momentum) -
         log(sqrt(
             four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd, angular_momentum)));
}

template <typename DataType, bool NeedSpacetime>
FishboneMoncriefDisk::IntermediateVariables<DataType, NeedSpacetime>::
    IntermediateVariables(const double black_hole_mass,
                          const double black_hole_spin,
                          const double max_pressure_radius,
                          const gr::Solutions::KerrSchild& background_spacetime,
                          const tnsr::I<DataType, 3>& x,
                          const double t) noexcept {
  const double a_squared = black_hole_spin * black_hole_spin;

  DataType z_squared = square(x.get(2));
  r_squared = [&x, &z_squared, &a_squared ]() noexcept {
    DataType temp =
        0.5 * (square(x.get(0)) + square(x.get(1)) + z_squared - a_squared);
    temp += sqrt(square(temp) + a_squared * z_squared);
    return temp;
  }
  ();
  z_squared /= -r_squared;
  z_squared += 1.0;
  sin_theta_squared = std::move(z_squared);

  angular_momentum = [
    a_squared, black_hole_mass, black_hole_spin,
    max_pressure_radius
  ]() noexcept {
    const double sqrt_m = sqrt(black_hole_mass);
    const double a_sqrt_m = black_hole_spin * sqrt_m;
    const double& rmax = max_pressure_radius;
    const double sqrt_rmax = sqrt(rmax);
    const double rmax_sqrt_rmax = rmax * sqrt_rmax;
    const double rmax_sqrd = rmax * rmax;
    return sqrt_m * (rmax_sqrt_rmax + a_sqrt_m) *
           (a_squared - 2.0 * a_sqrt_m * sqrt_rmax + rmax_sqrd) /
           (2.0 * a_sqrt_m * rmax_sqrt_rmax +
            (rmax - 3.0 * black_hole_mass) * rmax_sqrd);
  }
  ();

  if (NeedSpacetime) {
    auto kerr_schild_metric = background_spacetime.variables(
        x, t, gr::Solutions::KerrSchild::tags<DataType>{});

    inv_lapse = std::move(get<gr::Tags::Lapse<DataType>>(kerr_schild_metric));
    get(inv_lapse) = 1.0 / get(inv_lapse);
    shift = std::move(
        get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(kerr_schild_metric));
    spatial_metric =
        std::move(get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(
            kerr_schild_metric));
  }
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept {
  const auto specific_enthalpy =
      get<hydro::Tags::SpecificEnthalpy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>>{}, vars));
  auto rest_mass_density = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&rest_mass_density, &specific_enthalpy,
             this ](const size_t s, const double /*potential_at_s*/) noexcept {
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
    const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept {
  const double inner_edge_potential =
      potential(square(inner_edge_radius_), 1.0, vars.angular_momentum);
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
    const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept {
  const auto rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataType>>(variables(
          x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));
  auto pressure = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&pressure, &rest_mass_density,
             this ](const size_t s, const double /*potential_at_s*/) noexcept {
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
    const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept {
  const auto rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataType>>(variables(
          x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));
  auto specific_internal_energy = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(
      vars, [&specific_internal_energy, &rest_mass_density,
             this ](const size_t s, const double /*potential_at_s*/) noexcept {
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
    const IntermediateVariables<DataType, true>& vars) const noexcept {
  auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  variables_impl(vars, [&spatial_velocity, &vars, &x,
                        this ](const size_t s,
                               const double /*potential_at_s*/) noexcept {
    const double ang_velocity = angular_velocity(
        get_element(vars.r_squared, s), get_element(vars.sin_theta_squared, s),
        vars.angular_momentum);

    auto transport_velocity = make_array<3>(0.0);
    transport_velocity[0] -= ang_velocity * get_element(x.get(1), s);
    transport_velocity[1] += ang_velocity * get_element(x.get(0), s);

    for (size_t i = 0; i < 3; ++i) {
      get_element(spatial_velocity.get(i), s) =
          get_element(get(vars.inv_lapse), s) *
          (gsl::at(transport_velocity, i) + get_element(vars.shift.get(i), s));
    }
  });
  return {std::move(spatial_velocity)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
    const IntermediateVariables<DataType, true>& vars) const noexcept {
  const auto spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataType, 3>>(variables(
          x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}, vars));
  Scalar<DataType> lorentz_factor{
      1.0 / sqrt(1.0 - get(dot_product(spatial_velocity, spatial_velocity,
                                       vars.spatial_metric)))};
  return {std::move(lorentz_factor)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& /*vars*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>>(
      x, 0.0)};
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& /*vars*/) const
    noexcept {
  return {make_with_value<
      db::item_type<hydro::Tags::DivergenceCleaningField<DataType>>>(x, 0.0)};
}

template <typename DataType, bool NeedSpacetime, typename Func>
void FishboneMoncriefDisk::variables_impl(
    const IntermediateVariables<DataType, NeedSpacetime>& vars, Func f) const
    noexcept {
  const DataType& r_squared = vars.r_squared;
  const DataType& sin_theta_squared = vars.sin_theta_squared;
  const double angular_momentum = vars.angular_momentum;
  const double inner_edge_potential =
      potential(square(inner_edge_radius_), 1.0, angular_momentum);

  // fill the disk with matter
  for (size_t s = 0; s < get_size(r_squared); ++s) {
    const double r_squared_s = get_element(r_squared, s);
    const double sin_theta_squared_s = get_element(sin_theta_squared, s);

    // the disk won't extend closer to the axis than r sin theta = rin
    // so no need to evaluate the potential there
    if (sqrt(r_squared_s * sin_theta_squared_s) >= inner_edge_radius_) {
      const double potential_s =
          potential(r_squared_s, sin_theta_squared_s, angular_momentum);
      // the fluid can only be where W(r, theta) < W_in
      if (potential_s < inner_edge_potential) {
        f(s, potential_s);
      }
    }
  }
}

bool operator==(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) noexcept {
  return lhs.black_hole_mass() == rhs.black_hole_mass() and
         lhs.black_hole_spin() == rhs.black_hole_spin() and
         lhs.inner_edge_radius() == rhs.inner_edge_radius() and
         lhs.max_pressure_radius() == rhs.max_pressure_radius() and
         lhs.polytropic_constant() == rhs.polytropic_constant() and
         lhs.polytropic_exponent() == rhs.polytropic_exponent();
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
          DTYPE(data), NEED_SPACETIME(data)>& vars) const noexcept;           \
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>    \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> /*meta*/,        \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars) const noexcept;           \
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>            \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/,                \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars) const noexcept;           \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                       \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,  \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars) const noexcept;           \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::MagneticField<DTYPE(data), 3, Frame::Inertial>>            \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,                   \
                                            Frame::Inertial>> /*meta*/,       \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars) const noexcept;           \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::DivergenceCleaningField<DTYPE(data)>>                      \
  FishboneMoncriefDisk::variables(                                            \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/, \
      const FishboneMoncriefDisk::IntermediateVariables<                      \
          DTYPE(data), NEED_SPACETIME(data)>& vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (true, false))
#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                 \
  template tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> \
  FishboneMoncriefDisk::variables(                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                      \
      tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> /*meta*/,     \
      const FishboneMoncriefDisk::IntermediateVariables<DTYPE(data), true>&  \
          vars) const noexcept;                                              \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>      \
  FishboneMoncriefDisk::variables(                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                      \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/,          \
      const FishboneMoncriefDisk::IntermediateVariables<DTYPE(data), true>&  \
          vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef NEED_SPACETIME
#undef INSTANTIATE
}  // namespace Solutions
}  // namespace RelativisticEuler
/// \endcond
