// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Hllc.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler::BoundaryCorrections {
template <size_t Dim>
Hllc<Dim>::Hllc(CkMigrateMessage* msg) noexcept
    : BoundaryCorrection<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<BoundaryCorrection<Dim>> Hllc<Dim>::get_clone() const noexcept {
  return std::make_unique<Hllc>(*this);
}

template <size_t Dim>
void Hllc<Dim>::pup(PUP::er& p) {
  BoundaryCorrection<Dim>::pup(p);
}

template <size_t Dim>
template <size_t ThermodynamicDim>
double Hllc<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        packaged_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_pressure,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        packaged_normal_dot_flux_momentum_density,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_energy_density,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_velocity,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,

    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& energy_density,

    const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_mass_density,
    const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_momentum_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_energy_density,

    const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
    const Scalar<DataVector>& specific_internal_energy,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) const noexcept {
  {
    // Compute pressure
    if constexpr (ThermodynamicDim == 1) {
      *packaged_pressure =
          equation_of_state.pressure_from_density(mass_density);
    } else if constexpr (ThermodynamicDim == 2) {
      *packaged_pressure = equation_of_state.pressure_from_density_and_energy(
          mass_density, specific_internal_energy);
    }

    // Compute sound speed
    Scalar<DataVector>& sound_speed = *packaged_mass_density;
    sound_speed_squared(make_not_null(&sound_speed), mass_density,
                        specific_internal_energy, equation_of_state);
    get(sound_speed) = sqrt(get(sound_speed));

    // Compute normal velocity of fluid w.r.t. mesh
    Scalar<DataVector>& normal_dot_velocity = *packaged_energy_density;
    dot_product(make_not_null(&normal_dot_velocity), velocity, normal_covector);

    // Compute the normal velocity and largest outgoing / ingoing characteristic
    // speeds. Note that the notion of being 'largest ingoing' means taking the
    // most negative value given that the positive direction is outward normal.
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) =
          get(normal_dot_velocity) + get(sound_speed) -
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) =
          get(normal_dot_velocity) - get(sound_speed) -
          get(*normal_dot_mesh_velocity);
      get(*packaged_normal_dot_velocity) =
          get(normal_dot_velocity) - get(*normal_dot_mesh_velocity);
    } else {
      get(*packaged_largest_outgoing_char_speed) =
          get(normal_dot_velocity) + get(sound_speed);
      get(*packaged_largest_ingoing_char_speed) =
          get(normal_dot_velocity) - get(sound_speed);
      get(*packaged_normal_dot_velocity) = get(normal_dot_velocity);
    }
  }

  *packaged_mass_density = mass_density;
  *packaged_momentum_density = momentum_density;
  *packaged_energy_density = energy_density;
  *packaged_interface_unit_normal = normal_covector;

  normal_dot_flux(packaged_normal_dot_flux_mass_density, normal_covector,
                  flux_mass_density);
  normal_dot_flux(packaged_normal_dot_flux_momentum_density, normal_covector,
                  flux_momentum_density);
  normal_dot_flux(packaged_normal_dot_flux_energy_density, normal_covector,
                  flux_energy_density);

  return fmax(max(get(*packaged_largest_outgoing_char_speed)),
              -min(get(*packaged_largest_ingoing_char_speed)));
}

template <size_t Dim>
void Hllc<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_energy_density,
    const Scalar<DataVector>& mass_density_int,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_int,
    const Scalar<DataVector>& energy_density_int,
    const Scalar<DataVector>& pressure_int,
    const Scalar<DataVector>& normal_dot_flux_mass_density_int,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_momentum_density_int,
    const Scalar<DataVector>& normal_dot_flux_energy_density_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_int,
    const Scalar<DataVector>& normal_dot_velocity_int,
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
    const Scalar<DataVector>& mass_density_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_ext,
    const Scalar<DataVector>& energy_density_ext,
    const Scalar<DataVector>& pressure_ext,
    const Scalar<DataVector>& normal_dot_flux_mass_density_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_momentum_density_ext,
    const Scalar<DataVector>& normal_dot_flux_energy_density_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_ext,
    const Scalar<DataVector>& normal_dot_velocity_ext,
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const dg::Formulation dg_formulation) const noexcept {
  // Allocate a temp buffer
  const size_t vector_size = get(mass_density_int).size();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>>>
      temps{vector_size};

  // Determine lambda_max and lambda_min from the characteristic speeds from
  // interior and exterior
  get(get<::Tags::TempScalar<0>>(temps)) =
      min(0., get(largest_ingoing_char_speed_int),
          -get(largest_outgoing_char_speed_ext));
  const DataVector& lambda_min = get(get<::Tags::TempScalar<0>>(temps));
  get(get<::Tags::TempScalar<1>>(temps)) =
      max(0., get(largest_outgoing_char_speed_int),
          -get(largest_ingoing_char_speed_ext));
  const DataVector& lambda_max = get(get<::Tags::TempScalar<1>>(temps));

  // Compute lambda_star, the contact wave speed. (cf. Eq 10.37 of Toro2009,
  // note that here lambda instead of S was used to denote characteristic
  // speeds)
  get(get<::Tags::TempScalar<2>>(temps)) =
      (get(pressure_ext) - get(pressure_int) +
       get(mass_density_int) * get(normal_dot_velocity_int) *
           (lambda_min - get(normal_dot_velocity_int)) +
       get(mass_density_ext) * get(normal_dot_velocity_ext) *
           (lambda_max + get(normal_dot_velocity_ext))) /
      (get(mass_density_int) * (lambda_min - get(normal_dot_velocity_int)) -
       get(mass_density_ext) * (lambda_max + get(normal_dot_velocity_ext)));
  const DataVector& lambda_star = get(get<::Tags::TempScalar<2>>(temps));

  // Precompute common numerical factors for state variables in star region
  // (cf. Eq 10.39 of Toro2009)
  get(get<::Tags::TempScalar<3>>(temps)) =
      (lambda_min - get(normal_dot_velocity_int)) / (lambda_min - lambda_star);
  const DataVector& prefactor_int = get(get<::Tags::TempScalar<3>>(temps));
  get(get<::Tags::TempScalar<4>>(temps)) =
      (lambda_max + get(normal_dot_velocity_ext)) / (lambda_max - lambda_star);
  const DataVector& prefactor_ext = get(get<::Tags::TempScalar<4>>(temps));

  for (size_t i = 0; i < vector_size; ++i) {
    // check if lambda_star falls in the correct range [lambda_min,lambda_max]
    ASSERT(
        (lambda_star[i] <= lambda_max[i]) and (lambda_star[i] >= lambda_min[i]),
        "lambda_star in HLLC boundary correction is not consistent : "
            << "\n lambda_min  = " << lambda_min[i] << "\n lambda_*    = "
            << lambda_star[i] << "\n lambda_max  = " << lambda_max[i]);

    if (dg_formulation == dg::Formulation::WeakInertial) {
      // Compute intermediate flux F_star (cf. Eq 10.71 - 10.73 of Toro2009)
      if (lambda_star[i] >= 0.0) {
        get(*boundary_correction_mass_density)[i] =
            get(normal_dot_flux_mass_density_int)[i] +
            lambda_min[i] * (prefactor_int[i] - 1.0) * get(mass_density_int)[i];
        for (size_t spatial_index = 0; spatial_index < Dim; ++spatial_index) {
          boundary_correction_momentum_density->get(spatial_index)[i] =
              normal_dot_flux_momentum_density_int.get(spatial_index)[i] +
              lambda_min[i] *
                  (get(mass_density_int)[i] * prefactor_int[i] *
                       (lambda_star[i] - get(normal_dot_velocity_int)[i]) *
                       interface_unit_normal_int.get(spatial_index)[i] +
                   momentum_density_int.get(spatial_index)[i] *
                       (prefactor_int[i] - 1.0));
        }
        get(*boundary_correction_energy_density)[i] =
            get(normal_dot_flux_energy_density_int)[i] +
            lambda_min[i] *
                ((prefactor_int[i] - 1.0) *
                     (get(energy_density_int)[i] + get(pressure_int)[i]) +
                 prefactor_int[i] * get(mass_density_int)[i] * lambda_star[i] *
                     (lambda_star[i] - get(normal_dot_velocity_int)[i]));
      } else {
        get(*boundary_correction_mass_density)[i] =
            -get(normal_dot_flux_mass_density_ext)[i] +
            lambda_max[i] * (prefactor_ext[i] - 1.0) * get(mass_density_ext)[i];
        for (size_t spatial_index = 0; spatial_index < Dim; ++spatial_index) {
          boundary_correction_momentum_density->get(spatial_index)[i] =
              -normal_dot_flux_momentum_density_ext.get(spatial_index)[i] +
              lambda_max[i] *
                  (get(mass_density_ext)[i] * prefactor_ext[i] *
                       (lambda_star[i] + get(normal_dot_velocity_ext)[i]) *
                       (-interface_unit_normal_ext.get(spatial_index)[i]) +
                   momentum_density_ext.get(spatial_index)[i] *
                       (prefactor_ext[i] - 1.0));
        }
        get(*boundary_correction_energy_density)[i] =
            -get(normal_dot_flux_energy_density_ext)[i] +
            lambda_max[i] *
                ((prefactor_ext[i] - 1.0) *
                     (get(energy_density_ext)[i] + get(pressure_ext)[i]) +
                 prefactor_ext[i] * get(mass_density_ext)[i] * lambda_star[i] *
                     (lambda_star[i] + get(normal_dot_velocity_ext)[i]));
      }
    } else {
      // Compute boundary correction for strong formulation
      if (lambda_star[i] >= 0.0) {
        get(*boundary_correction_mass_density)[i] =
            lambda_min[i] * (prefactor_int[i] - 1.0) * get(mass_density_int)[i];
        for (size_t spatial_index = 0; spatial_index < Dim; ++spatial_index) {
          boundary_correction_momentum_density->get(spatial_index)[i] =
              lambda_min[i] *
              (get(mass_density_int)[i] * prefactor_int[i] *
                   (lambda_star[i] - get(normal_dot_velocity_int)[i]) *
                   interface_unit_normal_int.get(spatial_index)[i] +
               momentum_density_int.get(spatial_index)[i] *
                   (prefactor_int[i] - 1.0));
        }
        get(*boundary_correction_energy_density)[i] =
            lambda_min[i] *
            ((prefactor_int[i] - 1.0) *
                 (get(energy_density_int)[i] + get(pressure_int)[i]) +
             prefactor_int[i] * get(mass_density_int)[i] * lambda_star[i] *
                 (lambda_star[i] - get(normal_dot_velocity_int)[i]));
      } else {
        get(*boundary_correction_mass_density)[i] =
            -get(normal_dot_flux_mass_density_int)[i] -
            get(normal_dot_flux_mass_density_ext)[i] +
            lambda_max[i] * (prefactor_ext[i] - 1.0) * get(mass_density_ext)[i];
        for (size_t spatial_index = 0; spatial_index < Dim; ++spatial_index) {
          boundary_correction_momentum_density->get(spatial_index)[i] =
              -normal_dot_flux_momentum_density_int.get(spatial_index)[i] -
              normal_dot_flux_momentum_density_ext.get(spatial_index)[i] +
              lambda_max[i] *
                  (get(mass_density_ext)[i] * prefactor_ext[i] *
                       (lambda_star[i] + get(normal_dot_velocity_ext)[i]) *
                       (-interface_unit_normal_ext.get(spatial_index)[i]) +
                   momentum_density_ext.get(spatial_index)[i] *
                       (prefactor_ext[i] - 1.0));
        }
        get(*boundary_correction_energy_density)[i] =
            -get(normal_dot_flux_energy_density_int)[i] -
            get(normal_dot_flux_energy_density_ext)[i] +
            lambda_max[i] *
                ((prefactor_ext[i] - 1.0) *
                     (get(energy_density_ext)[i] + get(pressure_ext)[i]) +
                 prefactor_ext[i] * get(mass_density_ext)[i] * lambda_star[i] *
                     (lambda_star[i] + get(normal_dot_velocity_ext)[i]));
      }
    }
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Hllc<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Hllc<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                                 \
  template double Hllc<DIM(data)>::dg_package_data<THERMODIM(data)>(           \
      gsl::not_null<Scalar<DataVector>*> packaged_mass_density,                \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>          \
          packaged_momentum_density,                                           \
      gsl::not_null<Scalar<DataVector>*> packaged_energy_density,              \
      gsl::not_null<Scalar<DataVector>*> packaged_pressure,                    \
      gsl::not_null<Scalar<DataVector>*>                                       \
          packaged_normal_dot_flux_mass_density,                               \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>          \
          packaged_normal_dot_flux_momentum_density,                           \
      gsl::not_null<Scalar<DataVector>*>                                       \
          packaged_normal_dot_flux_energy_density,                             \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>          \
          packaged_interface_unit_normal,                                      \
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_velocity,         \
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed, \
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,  \
                                                                               \
      const Scalar<DataVector>& mass_density,                                  \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& momentum_density, \
      const Scalar<DataVector>& energy_density,                                \
                                                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                   \
          flux_mass_density,                                                   \
      const tnsr::IJ<DataVector, DIM(data), Frame::Inertial>&                  \
          flux_momentum_density,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                   \
          flux_energy_density,                                                 \
                                                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& velocity,         \
      const Scalar<DataVector>& specific_internal_energy,                      \
                                                                               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& normal_covector,  \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,       \
      const EquationsOfState::EquationOfState<false, THERMODIM(data)>&         \
          equation_of_state) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef THERMODIM
#undef DIM
}  // namespace NewtonianEuler::BoundaryCorrections
