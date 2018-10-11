// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"

#include <boost/optional/optional.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_include <array>

/// \cond
namespace grmhd {
namespace ValenciaDivClean {

template <typename PrimitiveRecoveryScheme, size_t ThermodynamicDim>
void PrimitiveFromConservative<PrimitiveRecoveryScheme, ThermodynamicDim>::
    apply(
        gsl::not_null<Scalar<DataVector>*> rest_mass_density,
        gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
        gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
            spatial_velocity,
        gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
        gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
        gsl::not_null<Scalar<DataVector>*> lorentz_factor,
        gsl::not_null<Scalar<DataVector>*> pressure,
        gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
        const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
        const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
        const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
        const Scalar<DataVector>& tilde_phi,
        const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
        const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
        const Scalar<DataVector>& sqrt_det_spatial_metric,
        const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
            equation_of_state) noexcept {
  get(*divergence_cleaning_field) =
      get(tilde_phi) / get(sqrt_det_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field->get(i) = tilde_b.get(i) / get(sqrt_det_spatial_metric);
  }
  const DataVector total_energy_density =
      (get(tilde_tau) + get(tilde_d)) / get(sqrt_det_spatial_metric);
  const auto tilde_s_upper = raise_or_lower_index(tilde_s, inv_spatial_metric);
  const DataVector momentum_density_squared =
      get(dot_product(tilde_s, tilde_s_upper)) /
      square(get(sqrt_det_spatial_metric));
  const DataVector momentum_density_dot_magnetic_field =
      get(dot_product(tilde_s, *magnetic_field)) / get(sqrt_det_spatial_metric);
  const DataVector magnetic_field_squared =
      get(dot_product(*magnetic_field, *magnetic_field, spatial_metric));
  const DataVector rest_mass_density_times_lorentz_factor =
      get(tilde_d) / get(sqrt_det_spatial_metric);

  for (size_t s = 0; s < total_energy_density.size(); ++s) {
    const boost::optional<PrimitiveRecoverySchemes::PrimitiveRecoveryData>
        primitive_data =
            PrimitiveRecoveryScheme::template apply<ThermodynamicDim>(
                total_energy_density[s], momentum_density_squared[s],
                momentum_density_dot_magnetic_field[s],
                magnetic_field_squared[s],
                rest_mass_density_times_lorentz_factor[s], equation_of_state);

    if (primitive_data) {
      get(*rest_mass_density)[s] = primitive_data.get().rest_mass_density;
      const double coefficient_of_b =
          momentum_density_dot_magnetic_field[s] /
          (primitive_data.get().rho_h_w_squared *
           (primitive_data.get().rho_h_w_squared + magnetic_field_squared[s]));
      const double coefficient_of_s =
          1.0 /
          (get(sqrt_det_spatial_metric)[s] *
           (primitive_data.get().rho_h_w_squared + magnetic_field_squared[s]));
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity->get(i)[s] =
            coefficient_of_b * magnetic_field->get(i)[s] +
            coefficient_of_s * tilde_s_upper.get(i)[s];
      }
      get(*lorentz_factor)[s] = primitive_data.get().lorentz_factor;
      get(*pressure)[s] = primitive_data.get().pressure;
    } else {
      ERROR(PrimitiveRecoveryScheme::name()
            << " primitive inversion scheme failed.");
    }
  }
  *specific_internal_energy = make_overloader(
      [&rest_mass_density](const EquationsOfState::EquationOfState<true, 1>&
                               the_equation_of_state) noexcept {
        return the_equation_of_state.specific_internal_energy_from_density(
            *rest_mass_density);
      },
      [&rest_mass_density,
       &pressure ](const EquationsOfState::EquationOfState<true, 2>&
                       the_equation_of_state) noexcept {
        return the_equation_of_state
            .specific_internal_energy_from_density_and_pressure(
                *rest_mass_density, *pressure);
      })(equation_of_state);
  *specific_enthalpy = hydro::specific_enthalpy(
      *rest_mass_density, *specific_internal_energy, *pressure);
}
}  // namespace ValenciaDivClean
}  // namespace grmhd

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                        \
  template struct grmhd::ValenciaDivClean::PrimitiveFromConservative< \
      RECOVERY(data), THERMODIM(data)>;

GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
     grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl),
    (1, 2))

#undef INSTANTIATION
#undef THERMODIM
#undef RECOVERY
/// \endcond
