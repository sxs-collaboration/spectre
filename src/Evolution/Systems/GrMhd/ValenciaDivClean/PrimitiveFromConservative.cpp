// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"

#include <iomanip>
#include <limits>
#include <optional>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAlHydro.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace grmhd::ValenciaDivClean {

template <typename OrderedListOfPrimitiveRecoverySchemes, bool ErrorOnFailure>
template <bool EnforcePhysicality, size_t ThermodynamicDim>
bool PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes,
                               ErrorOnFailure>::
    apply(const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
          const gsl::not_null<Scalar<DataVector>*> electron_fraction,
          const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
          const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
              spatial_velocity,
          const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
              magnetic_field,
          const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
          const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
          const gsl::not_null<Scalar<DataVector>*> pressure,
          const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
          const gsl::not_null<Scalar<DataVector>*> temperature,
          const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
          const Scalar<DataVector>& tilde_tau,
          const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
          const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
          const Scalar<DataVector>& tilde_phi,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
          const Scalar<DataVector>& sqrt_det_spatial_metric,
          const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
              equation_of_state,
          const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
              primitive_from_conservative_options) {
  get(*divergence_cleaning_field) =
      get(tilde_phi) / get(sqrt_det_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field->get(i) = tilde_b.get(i) / get(sqrt_det_spatial_metric);
  }
  const size_t number_of_points = get<0>(tilde_b).size();
  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                 ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                 ::Tags::TempScalar<4>, ::Tags::TempI<5, 3, Frame::Inertial>>>
      temp_buffer(number_of_points);

  DataVector& tau = get(get<::Tags::TempScalar<0>>(temp_buffer));
  tau = get(tilde_tau) / get(sqrt_det_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial>& tilde_s_upper =
      get<::Tags::TempI<5, 3, Frame::Inertial>>(temp_buffer);
  raise_or_lower_index(make_not_null(&tilde_s_upper), tilde_s,
                       inv_spatial_metric);

  Scalar<DataVector>& momentum_density_squared =
      get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&momentum_density_squared), tilde_s, tilde_s_upper);
  get(momentum_density_squared) /= square(get(sqrt_det_spatial_metric));

  Scalar<DataVector>& momentum_density_dot_magnetic_field =
      get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&momentum_density_dot_magnetic_field), tilde_s,
              *magnetic_field);
  get(momentum_density_dot_magnetic_field) /= get(sqrt_det_spatial_metric);

  Scalar<DataVector>& magnetic_field_squared =
      get<::Tags::TempScalar<3>>(temp_buffer);
  dot_product(make_not_null(&magnetic_field_squared), *magnetic_field,
              *magnetic_field, spatial_metric);

  DataVector& rest_mass_density_times_lorentz_factor =
      get(get<::Tags::TempScalar<4>>(temp_buffer));
  rest_mass_density_times_lorentz_factor =
      get(tilde_d) / get(sqrt_det_spatial_metric);

  // Parameters for quick exit from inversion
  const double cutoffD =
      primitive_from_conservative_options.cutoff_d_for_inversion();
  const double floorD =
      primitive_from_conservative_options.density_when_skipping_inversion();

  // This may need bounds
  // limit Ye to table bounds once that is implemented
  for (size_t s = 0; s < number_of_points; ++s) {
    get(*electron_fraction)[s] =
        std::min(0.5, std::max(get(tilde_ye)[s] / get(tilde_d)[s], 0.));

    std::optional<PrimitiveRecoverySchemes::PrimitiveRecoveryData>
        primitive_data = std::nullopt;
    // Quick exit from inversion in low-density regions where we will
    // apply atmosphere corrections anyways.
    if (rest_mass_density_times_lorentz_factor[s] < cutoffD) {
      if constexpr (ThermodynamicDim == 2) {
        const double specific_energy_at_point =
            get(equation_of_state
                    .specific_internal_energy_from_density_and_temperature(
                        Scalar<double>{floorD}, Scalar<double>{0.0}));
        const double pressure_at_point =
            get(equation_of_state.pressure_from_density_and_energy(
                Scalar<double>{floorD},
                Scalar<double>{specific_energy_at_point}));
        const double specific_enthalpy_at_point =
            1.0 + specific_energy_at_point + pressure_at_point / floorD;
        primitive_data = PrimitiveRecoverySchemes::PrimitiveRecoveryData{
            floorD,
            1.0,
            pressure_at_point,
            specific_energy_at_point,
            floorD * specific_enthalpy_at_point,
            get(*electron_fraction)[s]};
      }
      else if constexpr (ThermodynamicDim == 1) {
        const double specific_energy_at_point =
            get(equation_of_state
                    .specific_internal_energy_from_density(
                        Scalar<double>{floorD}));
        const double pressure_at_point =
            get(equation_of_state.pressure_from_density(
                Scalar<double>{floorD}));
        const double specific_enthalpy_at_point =
            1.0 + specific_energy_at_point + pressure_at_point / floorD;
        primitive_data = PrimitiveRecoverySchemes::PrimitiveRecoveryData{
            floorD,
            1.0,
            pressure_at_point,
            specific_energy_at_point,
            floorD * specific_enthalpy_at_point,
            get(*electron_fraction)[s]};
      }
    } else {
      auto apply_scheme =
          [&pressure, &primitive_data, &tau, &momentum_density_squared,
           &momentum_density_dot_magnetic_field, &magnetic_field_squared,
           &rest_mass_density_times_lorentz_factor, &equation_of_state, &s,
           &electron_fraction](auto scheme) {
            using primitive_recovery_scheme = tmpl::type_from<decltype(scheme)>;
            if (not primitive_data.has_value()) {
              primitive_data =
                  primitive_recovery_scheme::template apply<ThermodynamicDim>(
                      get(*pressure)[s], tau[s],
                      get(momentum_density_squared)[s],
                      get(momentum_density_dot_magnetic_field)[s],
                      get(magnetic_field_squared)[s],
                      rest_mass_density_times_lorentz_factor[s],
                      get(*electron_fraction)[s], equation_of_state);
            }
          };

      // Check consistency
      if (use_hydro_optimization and
          (get(magnetic_field_squared)[s] <
           100.0 * std::numeric_limits<double>::epsilon() * tau[s])) {
        tmpl::for_each<
            tmpl::list<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::
                           KastaunEtAlHydro<EnforcePhysicality>>>(apply_scheme);
      } else {
        tmpl::for_each<OrderedListOfPrimitiveRecoverySchemes>(apply_scheme);
      }
    }

    if (primitive_data.has_value()) {
      get(*rest_mass_density)[s] = primitive_data.value().rest_mass_density;
      const double coefficient_of_b =
          get(momentum_density_dot_magnetic_field)[s] /
          (primitive_data.value().rho_h_w_squared *
           (primitive_data.value().rho_h_w_squared +
            get(magnetic_field_squared)[s]));
      const double coefficient_of_s =
          1.0 / (get(sqrt_det_spatial_metric)[s] *
                 (primitive_data.value().rho_h_w_squared +
                  get(magnetic_field_squared)[s]));
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity->get(i)[s] =
            coefficient_of_b * magnetic_field->get(i)[s] +
            coefficient_of_s * tilde_s_upper.get(i)[s];
      }
      get(*lorentz_factor)[s] = primitive_data.value().lorentz_factor;
      get(*pressure)[s] = primitive_data.value().pressure;
      if constexpr (ThermodynamicDim != 1) {
        get(*specific_internal_energy)[s] =
            primitive_data.value().specific_internal_energy;
        get(*specific_enthalpy)[s] = primitive_data.value().rho_h_w_squared /
                                     (primitive_data.value().rest_mass_density *
                                      primitive_data.value().lorentz_factor *
                                      primitive_data.value().lorentz_factor);
      }
    } else {
      if constexpr (ErrorOnFailure) {
        ERROR("All primitive inversion schemes failed at s = "
              << s << ".\n"
              << std::setprecision(17) << "tau = " << tau[s] << "\n"
              << "rest_mass_density_times_lorentz_factor = "
              << rest_mass_density_times_lorentz_factor[s] << "\n"
              << "momentum_density_squared = "
              << get(momentum_density_squared)[s] << "\n"
              << "momentum_density_dot_magnetic_field = "
              << get(momentum_density_dot_magnetic_field)[s] << "\n"
              << "magnetic_field_squared = " << get(magnetic_field_squared)[s]
              << "\n"
              << "rest_mass_density_times_lorentz_factor = "
              << rest_mass_density_times_lorentz_factor[s] << "\n"
              << "previous_rest_mass_density = " << get(*rest_mass_density)[s]
              << "\n"
              << "previous_pressure = " << get(*pressure)[s] << "\n"
              << "previous_lorentz_factor = " << get(*lorentz_factor)[s]
              << "\n");
      } else {
        return false;
      }
    }
  }
  if constexpr (ThermodynamicDim == 1) {
    // Since the primitive recovery scheme is not restricted to lie on the
    // EOS-satisfying sub-manifold, we project back to the sub-manifold by
    // recomputing the specific internal energy and specific enthalpy from the
    // EOS.
    *specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density(
            *rest_mass_density);
    *temperature =
        equation_of_state.temperature_from_density(*rest_mass_density);
    hydro::relativistic_specific_enthalpy(specific_enthalpy, *rest_mass_density,
                                          *specific_internal_energy, *pressure);
  } else if constexpr (ThermodynamicDim == 2) {
    *temperature = equation_of_state.temperature_from_density_and_energy(
        *rest_mass_density, *specific_internal_energy);
  } else if constexpr (ThermodynamicDim == 3) {
    *temperature = equation_of_state.temperature_from_density_and_energy(
        *rest_mass_density, *specific_internal_energy, *electron_fraction);
  }
  return true;
}
}  // namespace grmhd::ValenciaDivClean

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define ERROR_ON_FAILURE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                        \
  template struct grmhd::ValenciaDivClean::PrimitiveFromConservative< \
      RECOVERY(data), ERROR_ON_FAILURE(data)>;

using NewmanHamlinThenPalenzuelaEtAl = tmpl::list<
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;
using KastaunThenNewmanThenPalenzuela = tmpl::list<
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;

GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
     NewmanHamlinThenPalenzuelaEtAl),
    (true, false))

#undef INSTANTIATION

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(2, data)

#define PHYSICALITY(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATION(_, data)                                               \
  template bool grmhd::ValenciaDivClean::PrimitiveFromConservative<          \
      RECOVERY(data), ERROR_ON_FAILURE(data)>::apply<PHYSICALITY(data),      \
                                                     THERMODIM(data)>(       \
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,            \
      const gsl::not_null<Scalar<DataVector>*> electron_fraction,            \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,     \
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>          \
          spatial_velocity,                                                  \
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>          \
          magnetic_field,                                                    \
      const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,    \
      const gsl::not_null<Scalar<DataVector>*> lorentz_factor,               \
      const gsl::not_null<Scalar<DataVector>*> pressure,                     \
      const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,            \
      const gsl::not_null<Scalar<DataVector>*> temperature,                  \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye, \
      const Scalar<DataVector>& tilde_tau,                                   \
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,                \
      const Scalar<DataVector>& tilde_phi,                                   \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,        \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,    \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                     \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&        \
          equation_of_state,                                                 \
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&       \
          primitive_from_conservative_options);

GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (tmpl::list<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
     tmpl::list<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::
                    KastaunEtAlHydro<true>>,
     tmpl::list<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::
                    KastaunEtAlHydro<false>>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
     NewmanHamlinThenPalenzuelaEtAl, KastaunThenNewmanThenPalenzuela),
    (true, false), (1, 2), (true, false))

#undef INSTANTIATION
#undef THERMODIM
#undef PHYSICALITY
#undef RECOVERY
