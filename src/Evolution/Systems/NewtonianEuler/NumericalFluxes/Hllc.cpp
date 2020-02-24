// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/Hllc.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {
namespace NumericalFluxes {

template <size_t Dim, typename Frame>
void Hllc<Dim, Frame>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& normal_dot_flux_mass_density,
    const tnsr::I<DataVector, Dim, Frame>& normal_dot_flux_momentum_density,
    const Scalar<DataVector>& normal_dot_flux_energy_density,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim, Frame>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim, Frame>& velocity,
    const Scalar<DataVector>& pressure,
    const db::const_item_type<char_speeds_tag>& characteristic_speeds,
    const tnsr::i<DataVector, Dim, Frame>& interface_unit_normal) const
    noexcept {
  get<::Tags::NormalDotFlux<Tags::MassDensityCons<DataVector>>>(
      *packaged_data) = normal_dot_flux_mass_density;
  get<::Tags::NormalDotFlux<Tags::MomentumDensity<DataVector, Dim, Frame>>>(
      *packaged_data) = normal_dot_flux_momentum_density;
  get<::Tags::NormalDotFlux<Tags::EnergyDensity<DataVector>>>(*packaged_data) =
      normal_dot_flux_energy_density;
  get<Tags::MassDensityCons<DataVector>>(*packaged_data) = mass_density;
  get<Tags::MomentumDensity<DataVector, Dim, Frame>>(*packaged_data) =
      momentum_density;
  get<Tags::EnergyDensity<DataVector>>(*packaged_data) = energy_density;
  get<Tags::Pressure<DataVector>>(*packaged_data) = pressure;
  get<::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>>(
      *packaged_data) = interface_unit_normal;
  get<NormalVelocity>(*packaged_data) =
      dot_product(interface_unit_normal, velocity);

  // When packaging interior data, LargestIngoingSpeed and LargestOutgoingSpeed
  // will hold the min and max char speeds, respectively. On the other hand,
  // when packaging exterior data, the characteristic speeds will be computed
  // along *minus* the exterior normal, so LargestIngoingSpeed will hold *minus*
  // the max speed, while LargestOutgoingSpeed will store *minus* the min speed.
  get<LargestIngoingSpeed>(*packaged_data) =
      make_with_value<Scalar<DataVector>>(
          characteristic_speeds[0],
          std::numeric_limits<double>::signaling_NaN());
  get<LargestOutgoingSpeed>(*packaged_data) =
      make_with_value<Scalar<DataVector>>(
          characteristic_speeds[0],
          std::numeric_limits<double>::signaling_NaN());
  for (size_t s = 0; s < characteristic_speeds[0].size(); ++s) {
    get(get<LargestIngoingSpeed>(*packaged_data))[s] = (*std::min_element(
        characteristic_speeds.begin(), characteristic_speeds.end(),
        [&s](const DataVector& a, const DataVector& b) noexcept {
          return a[s] < b[s];
        }))[s];
    get(get<LargestOutgoingSpeed>(*packaged_data))[s] = (*std::max_element(
        characteristic_speeds.begin(), characteristic_speeds.end(),
        [&s](const DataVector& a, const DataVector& b) noexcept {
          return a[s] < b[s];
        }))[s];
  }
}

template <size_t Dim, typename Frame>
void Hllc<Dim, Frame>::operator()(
    const gsl::not_null<Scalar<DataVector>*>
        normal_dot_numerical_flux_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame>*>
        normal_dot_numerical_flux_momentum_density,
    const gsl::not_null<Scalar<DataVector>*>
        normal_dot_numerical_flux_energy_density,
    const Scalar<DataVector>& normal_dot_flux_mass_density_int,
    const tnsr::I<DataVector, Dim, Frame>& normal_dot_flux_momentum_density_int,
    const Scalar<DataVector>& normal_dot_flux_energy_density_int,
    const Scalar<DataVector>& mass_density_int,
    const tnsr::I<DataVector, Dim, Frame>& momentum_density_int,
    const Scalar<DataVector>& energy_density_int,
    const Scalar<DataVector>& pressure_int,
    const tnsr::i<DataVector, Dim, Frame>& interface_unit_normal,
    const Scalar<DataVector>& normal_velocity_int,
    const Scalar<DataVector>& largest_ingoing_speed_int,
    const Scalar<DataVector>& largest_outgoing_speed_int,
    const Scalar<DataVector>& minus_normal_dot_flux_mass_density_ext,
    const tnsr::I<DataVector, Dim, Frame>&
        minus_normal_dot_flux_momentum_density_ext,
    const Scalar<DataVector>& minus_normal_dot_flux_energy_density_ext,
    const Scalar<DataVector>& mass_density_ext,
    const tnsr::I<DataVector, Dim, Frame>& momentum_density_ext,
    const Scalar<DataVector>& energy_density_ext,
    const Scalar<DataVector>& pressure_ext,
    const tnsr::i<DataVector, Dim, Frame>& minus_interface_unit_normal,
    const Scalar<DataVector>& minus_normal_velocity_ext,
    // names are inverted w.r.t interior data. See package_data()
    const Scalar<DataVector>& minus_largest_outgoing_speed_ext,
    const Scalar<DataVector>& minus_largest_ingoing_speed_ext) const noexcept {
  DataVector largest_ingoing_speed = make_with_value<DataVector>(
      largest_ingoing_speed_int, std::numeric_limits<double>::signaling_NaN());
  DataVector largest_outgoing_speed = make_with_value<DataVector>(
      largest_outgoing_speed_int, std::numeric_limits<double>::signaling_NaN());
  for (size_t s = 0; s < largest_ingoing_speed.size(); ++s) {
    largest_ingoing_speed[s] =
        std::min({get(largest_ingoing_speed_int)[s],
                  -get(minus_largest_ingoing_speed_ext)[s], 0.0});
    largest_outgoing_speed[s] =
        std::max({get(largest_outgoing_speed_int)[s],
                  -get(minus_largest_outgoing_speed_ext)[s], 0.0});
  }
  const DataVector signal_speed_star =
      (get(pressure_int) +
       get(normal_dot_flux_mass_density_int) *
           (get(normal_velocity_int) - largest_ingoing_speed) -
       get(pressure_ext) -
       get(minus_normal_dot_flux_mass_density_ext) *
           (get(minus_normal_velocity_ext) + largest_outgoing_speed)) /
      (get(normal_dot_flux_mass_density_int) -
       largest_ingoing_speed * get(mass_density_int) +
       get(minus_normal_dot_flux_mass_density_ext) +
       largest_outgoing_speed * get(mass_density_ext));

  for (size_t s = 0; s < largest_ingoing_speed.size(); ++s) {
    const double s_min = largest_ingoing_speed[s];
    const double s_max = largest_outgoing_speed[s];
    const double s_star = signal_speed_star[s];
    ASSERT((s_min <= s_star) and (s_star <= s_max),
           "Signal speeds in HLLC Riemann solver not consistent: "
               << "s_min = " << s_min << ", c_* = " << s_star
               << ", s_max = " << s_max);

    const auto pressure_star = [&s_star](
        const double mass_density, const double pressure,
        const double normal_velocity, const double signal_speed) noexcept {
      return pressure + mass_density * (normal_velocity - signal_speed) *
                            (normal_velocity - s_star);
    };
    const auto numerical_flux_star = [&s_star](
        const double n_dot_f, const double u, const double d_star,
        const double p_star, const double signal_speed) noexcept {
      return (s_star * (n_dot_f - signal_speed * u) -
              signal_speed * p_star * d_star) /
             (s_star - signal_speed);
    };

    if (s_star > 0.0) {
      const double pressure_star_int =
          pressure_star(get(mass_density_int)[s], get(pressure_int)[s],
                        get(normal_velocity_int)[s], s_min);

      get(*normal_dot_numerical_flux_mass_density)[s] = numerical_flux_star(
          get(normal_dot_flux_mass_density_int)[s], get(mass_density_int)[s],
          0.0, pressure_star_int, s_min);

      get(*normal_dot_numerical_flux_energy_density)[s] = numerical_flux_star(
          get(normal_dot_flux_energy_density_int)[s],
          get(energy_density_int)[s], s_star, pressure_star_int, s_min);

      for (size_t i = 0; i < Dim; ++i) {
        normal_dot_numerical_flux_momentum_density->get(i)[s] =
            numerical_flux_star(normal_dot_flux_momentum_density_int.get(i)[s],
                                momentum_density_int.get(i)[s],
                                interface_unit_normal.get(i)[s],
                                pressure_star_int, s_min);
      }
    } else {
      const double pressure_star_ext =
          pressure_star(get(mass_density_ext)[s], get(pressure_ext)[s],
                        -1.0 * get(minus_normal_velocity_ext)[s], s_max);

      get(*normal_dot_numerical_flux_mass_density)[s] = numerical_flux_star(
          -1.0 * get(minus_normal_dot_flux_mass_density_ext)[s],
          get(mass_density_ext)[s], 0.0, pressure_star_ext, s_max);

      get(*normal_dot_numerical_flux_energy_density)[s] = numerical_flux_star(
          -1.0 * get(minus_normal_dot_flux_energy_density_ext)[s],
          get(energy_density_ext)[s], s_star, pressure_star_ext, s_max);

      for (size_t i = 0; i < Dim; ++i) {
        normal_dot_numerical_flux_momentum_density->get(i)[s] =
            numerical_flux_star(
                -1.0 * minus_normal_dot_flux_momentum_density_ext.get(i)[s],
                momentum_density_ext.get(i)[s],
                -1.0 * minus_interface_unit_normal.get(i)[s], pressure_star_ext,
                s_max);
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) template struct Hllc<DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
}  // namespace NumericalFluxes
}  // namespace NewtonianEuler
/// \endcond
