// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/Rusanov.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace RelativisticEuler::Valencia::BoundaryCorrections {
template <size_t Dim>
Rusanov<Dim>::Rusanov(CkMigrateMessage* msg) noexcept
    : BoundaryCorrection<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<BoundaryCorrection<Dim>> Rusanov<Dim>::get_clone()
    const noexcept {
  return std::make_unique<Rusanov>(*this);
}

template <size_t Dim>
void Rusanov<Dim>::pup(PUP::er& p) {
  BoundaryCorrection<Dim>::pup(p);
}

template <size_t Dim>
template <size_t ThermodynamicDim>
double Rusanov<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,

    const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, Dim, Frame::Inertial>& flux_tilde_s,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const noexcept {
  {
    // First compute v^2 and cs^2, storing them in return arguments to avoid
    // new allocations
    Scalar<DataVector>& spatial_velocity_squared =
        *packaged_normal_dot_flux_tilde_d;
    dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
                spatial_velocity, spatial_metric);

    Scalar<DataVector>& sound_speed_squared =
        *packaged_normal_dot_flux_tilde_tau;
    hydro::sound_speed_squared(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               specific_enthalpy, equation_of_state);

    // Create the array into which to compute the char speeds, then point each
    // of the DataVectors into one of the return args for the evolved fields
    std::array<DataVector, Dim + 2> char_speeds;
    char_speeds[0].set_data_ref(make_not_null(&get(*packaged_tilde_d)));
    char_speeds[Dim + 1].set_data_ref(make_not_null(&get(*packaged_tilde_tau)));
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(char_speeds, i + 1)
          .set_data_ref(make_not_null(&packaged_tilde_s->get(i)));
    }

    // Call into characteristic_speed function because speeds are tedious
    characteristic_speeds(make_not_null(&char_speeds), lapse, shift,
                          spatial_velocity, spatial_velocity_squared,
                          sound_speed_squared, normal_covector);

    // There are 3 different values in the Dim+2 char speed, so we do not need
    // to maximize over all entries in the std::array
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_abs_char_speed) =
          max(abs(char_speeds[0] - get(*normal_dot_mesh_velocity)),
              abs(char_speeds[1] - get(*normal_dot_mesh_velocity)),
              abs(char_speeds[Dim + 1] - get(*normal_dot_mesh_velocity)));
    } else {
      get(*packaged_abs_char_speed) = max(
          abs(char_speeds[0]), abs(char_speeds[1]), abs(char_speeds[Dim + 1]));
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;

  dot_product(packaged_normal_dot_flux_tilde_d, flux_tilde_d, normal_covector);
  dot_product(packaged_normal_dot_flux_tilde_tau, flux_tilde_tau,
              normal_covector);
  for (size_t i = 0; i < Dim; ++i) {
    packaged_normal_dot_flux_tilde_s->get(i) =
        get<0>(normal_covector) * flux_tilde_s.get(0, i);
    for (size_t j = 1; j < Dim; ++j) {
      packaged_normal_dot_flux_tilde_s->get(i) +=
          normal_covector.get(j) * flux_tilde_s.get(j, i);
    }
  }

  return max(get(*packaged_abs_char_speed));
}

template <size_t Dim>
void Rusanov<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_tilde_s_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_tilde_s_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) const noexcept {
  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_tilde_d) =
        0.5 * (get(normal_dot_flux_tilde_d_int) -
               get(normal_dot_flux_tilde_d_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_d_ext) - get(tilde_d_int));
    get(*boundary_correction_tilde_tau) =
        0.5 * (get(normal_dot_flux_tilde_tau_int) -
               get(normal_dot_flux_tilde_tau_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_tau_ext) - get(tilde_tau_int));
    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_tilde_s->get(i) =
          0.5 * (normal_dot_flux_tilde_s_int.get(i) -
                 normal_dot_flux_tilde_s_ext.get(i)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (tilde_s_ext.get(i) - tilde_s_int.get(i));
    }
  } else {
    get(*boundary_correction_tilde_d) =
        -0.5 * (get(normal_dot_flux_tilde_d_int) +
                get(normal_dot_flux_tilde_d_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_d_ext) - get(tilde_d_int));
    get(*boundary_correction_tilde_tau) =
        -0.5 * (get(normal_dot_flux_tilde_tau_int) +
                get(normal_dot_flux_tilde_tau_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_tau_ext) - get(tilde_tau_int));
    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_tilde_s->get(i) =
          -0.5 * (normal_dot_flux_tilde_s_int.get(i) +
                  normal_dot_flux_tilde_s_ext.get(i)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (tilde_s_ext.get(i) - tilde_s_int.get(i));
    }
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Rusanov<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Rusanov<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                                 \
  template double Rusanov<DIM(data)>::dg_package_data<THERMODIM(data)>(        \
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,                     \
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,                   \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>          \
          packaged_tilde_s,                                                    \
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,     \
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,   \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>          \
          packaged_normal_dot_flux_tilde_s,                                    \
      gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,              \
                                                                               \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& tilde_s,          \
                                                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& flux_tilde_d,     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& flux_tilde_tau,   \
      const tnsr::Ij<DataVector, DIM(data), Frame::Inertial>& flux_tilde_s,    \
                                                                               \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data)>& shift,                             \
      const tnsr::ii<DataVector, DIM(data)>& spatial_metric,                   \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
                                                                               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& normal_covector,  \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& normal_vector,    \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,       \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&          \
          equation_of_state) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef THERMODIM
#undef DIM

}  // namespace RelativisticEuler::Valencia::BoundaryCorrections
/// \endcond
