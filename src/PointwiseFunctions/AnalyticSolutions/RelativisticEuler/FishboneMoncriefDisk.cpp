// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace RelativisticEuler::Solutions {

FishboneMoncriefDisk::FishboneMoncriefDisk(CkMigrateMessage* msg)
    : InitialData(msg) {}

FishboneMoncriefDisk::FishboneMoncriefDisk(const double bh_mass,
                                           const double bh_dimless_spin,
                                           const double inner_edge_radius,
                                           const double max_pressure_radius,
                                           const double polytropic_constant,
                                           const double polytropic_exponent,
                                           const double noise)
    : bh_mass_(bh_mass),
      bh_spin_a_(bh_mass * bh_dimless_spin),
      inner_edge_radius_(bh_mass * inner_edge_radius),
      max_pressure_radius_(bh_mass * max_pressure_radius),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      noise_(noise),
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

  // compute rho_max_ here:
  const double potential_at_rmax = potential(rmax_squared, 1.0);
  const double potential_at_rin = potential(square(inner_edge_radius_), 1.0);
  const double h_at_rmax = exp(potential_at_rin - potential_at_rmax);
  rho_max_ = get(equation_of_state_.rest_mass_density_from_enthalpy(
      Scalar<double>{h_at_rmax}));
}

std::unique_ptr<evolution::initial_data::InitialData>
FishboneMoncriefDisk::get_clone() const {
  return std::make_unique<FishboneMoncriefDisk>(*this);
}

void FishboneMoncriefDisk::pup(PUP::er& p) {
  InitialData::pup(p);
  p | bh_mass_;
  p | bh_spin_a_;
  p | inner_edge_radius_;
  p | max_pressure_radius_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | angular_momentum_;
  p | rho_max_;
  p | noise_;
  p | equation_of_state_;
  p | background_spacetime_;
}
// Sigma in Fishbone&Moncrief eqn (3.5)
template <typename DataType>
DataType FishboneMoncriefDisk::sigma(const DataType& r_sqrd,
                                     const DataType& sin_theta_sqrd) const {
  return r_sqrd + square(bh_spin_a_) * (1.0 - sin_theta_sqrd);
}
// Inverse of A in Fishbone&Moncrief eqn (3.5)
template <typename DataType>
DataType FishboneMoncriefDisk::inv_ucase_a(const DataType& r_sqrd,
                                           const DataType& sin_theta_sqrd,
                                           const DataType& delta) const {
  const double a_sqrd = square(bh_spin_a_);
  const DataType r_sqrd_plus_a_sqrd = r_sqrd + a_sqrd;
  return 1.0 / (square(r_sqrd_plus_a_sqrd) - delta * a_sqrd * sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::four_velocity_t_sqrd(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd) const {
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
    const DataType& r_sqrd, const DataType& sin_theta_sqrd) const {
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
                                         const DataType& sin_theta_sqrd) const {
  return angular_momentum_ * angular_velocity(r_sqrd, sin_theta_sqrd) -
         log(sqrt(four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd)));
}

template <typename DataType>
FishboneMoncriefDisk::IntermediateVariables<DataType>::IntermediateVariables(
    const tnsr::I<DataType, 3>& x) {
  sin_theta_squared = square(get<0>(x)) + square(get<1>(x));
  r_squared = sin_theta_squared + square(get<2>(x));
  sin_theta_squared /= r_squared;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto specific_enthalpy =
      get<hydro::Tags::SpecificEnthalpy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>>{}, vars));
  auto rest_mass_density = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&rest_mass_density, &specific_enthalpy, this](
                           const size_t s, const double /*potential_at_s*/) {
    get_element(get(rest_mass_density), s) =
        (1 / rho_max_) *
        get(equation_of_state_.rest_mass_density_from_enthalpy(
            Scalar<double>{get_element(get(specific_enthalpy), s)}));
  });
  return {std::move(rest_mass_density)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /* vars */) const {
  // Need to add EoS call to get correct electron fraction
  // when using tables

  auto ye = make_with_value<Scalar<DataType>>(x, 0.1);

  return {std::move(ye)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const double inner_edge_potential =
      potential(square(inner_edge_radius_), 1.0);
  auto specific_enthalpy = make_with_value<Scalar<DataType>>(x, 1.0);
  variables_impl(vars, [&specific_enthalpy, inner_edge_potential](
                           const size_t s, const double potential_at_s) {
    get_element(get(specific_enthalpy), s) =
        exp(inner_edge_potential - potential_at_s);
  });
  return {std::move(specific_enthalpy)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-noise_, noise_);
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));
  auto pressure = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&pressure, &rest_mass_density, &gen, &dis, this](
                           const size_t s, const double /*potential_at_s*/) {
    get_element(get(pressure), s) =
        (1 / rho_max_) * (1 + dis(gen)) *
        get(equation_of_state_.pressure_from_density(
            Scalar<double>{rho_max_ * get_element(get(rest_mass_density), s)}));
  });
  return {std::move(pressure)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));
  auto specific_internal_energy = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&specific_internal_energy, &rest_mass_density, this](
                           const size_t s, const double /*potential_at_s*/) {
    get_element(get(specific_internal_energy), s) =
        get(equation_of_state_.specific_internal_energy_from_density(
            Scalar<double>{rho_max_ * get_element(get(rest_mass_density), s)}));
  });
  return {std::move(specific_internal_energy)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}, vars));

  auto temperature = make_with_value<Scalar<DataType>>(x, 0.0);
  variables_impl(vars, [&temperature, &rest_mass_density, this](
                           const size_t s, const double /*potential_at_s*/) {
    get_element(get(temperature), s) =
        get(equation_of_state_.temperature_from_density(
            Scalar<double>{get_element(get(rest_mass_density), s)}));
  });
  return {std::move(temperature)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  using inv_jacobian =
      gr::Solutions::SphericalKerrSchild::internal_tags::inv_jacobian<
          DataType, Frame::Inertial>;
  const auto inv_jacobians = get<inv_jacobian>(background_spacetime_.variables(
      x, 0.0, tmpl::list<inv_jacobian>{},
      make_not_null(&vars->sph_kerr_schild_cache)));
  const auto shifts =
      get<gr::Tags::Shift<DataType, 3>>(background_spacetime_.variables(
          x, 0.0, tmpl::list<gr::Tags::Shift<DataType, 3>>{},
          make_not_null(&vars->sph_kerr_schild_cache)));
  const auto lapses =
      get<gr::Tags::Lapse<DataType>>(background_spacetime_.variables(
          x, 0.0, tmpl::list<gr::Tags::Lapse<DataType>>{},
          make_not_null(&vars->sph_kerr_schild_cache)));
  variables_impl(vars, [&spatial_velocity, &vars, &x, &inv_jacobians, &shifts,
                        &lapses,
                        this](const size_t s, const double /*potential_at_s*/) {
    const double ang_velocity =
        angular_velocity(get_element(vars->r_squared, s),
                         get_element(vars->sin_theta_squared, s));
    // We first compute the transport velocity in Kerr-Schild coordinates
    // and then transform this vector back to Spherical Kerr-Schild coordinates.
    auto transport_velocity_ks = make_array<3>(0.0);
    auto transport_velocity_sks = make_array<3>(0.0);
    const double sks_to_ks_factor =
        sqrt(square(bh_spin_a_) + get_element(vars->r_squared, s)) /
        sqrt(get_element(vars->r_squared, s));
    transport_velocity_ks[0] -=
        ang_velocity * get_element(x.get(1), s) * sks_to_ks_factor;
    transport_velocity_ks[1] +=
        ang_velocity * get_element(x.get(0), s) * sks_to_ks_factor;
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(transport_velocity_sks, j) +=
            get_element(inv_jacobians.get(j, i), s) *
            gsl::at(transport_velocity_ks, i);
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      get_element(spatial_velocity.get(i), s) =
          (gsl::at(transport_velocity_sks, i) + get_element(shifts.get(i), s)) /
          get_element(get(lapses), s);
    }
  });
  return {std::move(spatial_velocity)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  const auto spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataType, 3>>(variables(
          x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}, vars));
  Scalar<DataType> lorentz_factor{
      1.0 /
      sqrt(1.0 - get(dot_product(
                     spatial_velocity, spatial_velocity,
                     get<gr::Tags::SpatialMetric<DataType, 3>>(
                         background_spacetime_.variables(
                             x, 0.0,
                             tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{},
                             make_not_null(&vars->sph_kerr_schild_cache))))))};
  return {std::move(lorentz_factor)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /*vars*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
    gsl::not_null<IntermediateVariables<DataType>*> /*vars*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType, typename Func>
void FishboneMoncriefDisk::variables_impl(
    gsl::not_null<IntermediateVariables<DataType>*> vars, Func f) const {
  const DataType& r_squared = vars->r_squared;
  const DataType& sin_theta_squared = vars->sin_theta_squared;
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

PUP::able::PUP_ID FishboneMoncriefDisk::my_PUP_ID = 0;

bool operator==(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) {
  return lhs.bh_mass_ == rhs.bh_mass_ and lhs.bh_spin_a_ == rhs.bh_spin_a_ and
         lhs.inner_edge_radius_ == rhs.inner_edge_radius_ and
         lhs.max_pressure_radius_ == rhs.max_pressure_radius_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_ and
         lhs.noise_ == rhs.noise_ and lhs.rho_max_ == rhs.rho_max_;
}

bool operator!=(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>;     \
  template tuples::TaggedTuple<hydro::Tags::RestMassDensity<DTYPE(data)>>      \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::RestMassDensity<DTYPE(data)>> /*meta*/,          \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::ElectronFraction<DTYPE(data)>>     \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::ElectronFraction<DTYPE(data)>> /*meta*/,         \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>     \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> /*meta*/,         \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>             \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/,                 \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<                                                \
      hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                        \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,   \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::Temperature<DTYPE(data)>>          \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::Temperature<DTYPE(data)>> /*meta*/,              \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::MagneticField<DTYPE(data), 3>>     \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,                    \
                                            Frame::Inertial>> /*meta*/,        \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<                                                \
      hydro::Tags::DivergenceCleaningField<DTYPE(data)>>                       \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/,  \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DTYPE(data), 3>>   \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3>> /*meta*/,       \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>        \
  FishboneMoncriefDisk::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/,            \
      gsl::not_null<FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;                                                         \
  template DTYPE(data) FishboneMoncriefDisk::potential(                        \
      const DTYPE(data) & r_sqrd, const DTYPE(data) & sin_theta_sqrd) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace RelativisticEuler::Solutions
