// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnDgGrid.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::subcell {
template <typename RecoveryScheme>
template <size_t ThermodynamicDim>
bool TciOnDgGrid<RecoveryScheme>::apply(
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> dg_prim_vars,
    const Scalar<DataVector>& subcell_tilde_d,
    const Scalar<DataVector>& subcell_tilde_tau,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Mesh<3>& dg_mesh, const TciOptions& tci_options,
    const double persson_exponent) {
  constexpr double persson_tci_epsilon = 1.0e-18;
  const size_t number_of_points = dg_mesh.number_of_grid_points();
  Variables<hydro::grmhd_tags<DataVector>> temp_prims(number_of_points);

  // require: tilde_d/sqrt{gamma} >= 0.0 (or some positive user-specified value)
  if (min(get(tilde_d) / get(sqrt_det_spatial_metric)) <
          tci_options.minimum_rest_mass_density_times_lorentz_factor or
      min(get(subcell_tilde_d)) <
          tci_options.minimum_rest_mass_density_times_lorentz_factor) {
    return true;
  }

  // require: tilde_tau >= 0.0 (or some positive user-specified value)
  if (min(get(tilde_tau)) < tci_options.minimum_tilde_tau or
      min(get(subcell_tilde_tau)) < tci_options.minimum_tilde_tau) {
    return true;
  }

  // Check if we are in atmosphere (but not negative tildeD), and if so, then we
  // continue using DG on this element.
  if (max(get(tilde_d) / get(sqrt_det_spatial_metric) /
          get(get<hydro::Tags::LorentzFactor<DataVector>>(*dg_prim_vars))) <
          tci_options.atmosphere_density and
      max(get(get<hydro::Tags::RestMassDensity<DataVector>>(*dg_prim_vars))) <
          tci_options.atmosphere_density) {
    // In atmosphere, we only need to recover the primitive variables for the
    // magnetic field and divergence cleaning field
    for (size_t i = 0; i < 3; ++i) {
      get<hydro::Tags::MagneticField<DataVector, 3>>(*dg_prim_vars).get(i) =
          tilde_b.get(i) / get(sqrt_det_spatial_metric);
    }
    get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(*dg_prim_vars)) =
        get(tilde_phi) / get(sqrt_det_spatial_metric);

    return false;
  }

  {
    // require: tilde{B}^2 <= 2sqrt{gamma}(1-epsilon_B)\tilde{tau}
    Scalar<DataVector>& tilde_b_squared =
        get<hydro::Tags::RestMassDensity<DataVector>>(temp_prims);
    dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
                spatial_metric);
    for (size_t i = 0; i < number_of_points; ++i) {
      if (get(tilde_b_squared)[i] >
          (1.0 - tci_options.safety_factor_for_magnetic_field) * 2.0 *
              get(tilde_tau)[i] * get(sqrt_det_spatial_metric)[i]) {
        return true;
      }
    }
  }

  // Try to recover the primitive variables.
  // We assign them to a temporary so that if recovery fails at any of the
  // points we can use the valid primitives at the current time to provide a
  // high-order initial guess for the recovery on the subcells.
  //
  // Copy over the pressure since it's used as an initial guess in some
  // recovery schemes.
  get<hydro::Tags::Pressure<DataVector>>(temp_prims) =
      get<hydro::Tags::Pressure<DataVector>>(*dg_prim_vars);
  if (not grmhd::ValenciaDivClean::
          PrimitiveFromConservative<tmpl::list<RecoveryScheme>, false>::apply(
              make_not_null(
                  &get<hydro::Tags::RestMassDensity<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                      temp_prims)),
              make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                  temp_prims)),
              make_not_null(
                  &get<hydro::Tags::MagneticField<DataVector, 3>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::DivergenceCleaningField<DataVector>>(
                      temp_prims)),
              make_not_null(
                  &get<hydro::Tags::LorentzFactor<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::Pressure<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::SpecificEnthalpy<DataVector>>(temp_prims)),
              tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_metric,
              inv_spatial_metric, sqrt_det_spatial_metric, eos)) {
    return true;
  }

  // Check if we are in atmosphere after recovery. Unlikely we'd hit this and
  // not the check before the recovery, but just in case.
  if (max(get(get<hydro::Tags::RestMassDensity<DataVector>>(temp_prims))) <
      tci_options.atmosphere_density) {
    *dg_prim_vars = std::move(temp_prims);
    return false;
  }

  // Check that tilde_d and tilde_tau satisfy the Persson TCI
  if (evolution::dg::subcell::persson_tci(tilde_d, dg_mesh, persson_exponent,
                                          persson_tci_epsilon) or
      evolution::dg::subcell::persson_tci(tilde_tau, dg_mesh, persson_exponent,
                                          persson_tci_epsilon)) {
    return true;
  }
  // Check Cartesian magnitude of magnetic field satisfies the Persson TCI
  const Scalar<DataVector> tilde_b_magnitude =
      tci_options.magnetic_field_cutoff.has_value() ? magnitude(tilde_b)
                                                    : Scalar<DataVector>{};
  if (tci_options.magnetic_field_cutoff.has_value() and
      max(get(tilde_b_magnitude)) >
          tci_options.magnetic_field_cutoff.value() and
      evolution::dg::subcell::persson_tci(
          tilde_b_magnitude, dg_mesh, persson_exponent, persson_tci_epsilon)) {
    return true;
  }

  *dg_prim_vars = std::move(temp_prims);
  return false;
}

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class TciOnDgGrid<RECOVERY(data)>;
GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
     grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
     grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl))
#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template bool TciOnDgGrid<RECOVERY(data)>::apply<THERMO_DIM(data)>(         \
      const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>          \
          dg_prim_vars,                                                       \
      const Scalar<DataVector>& subcell_tilde_d,                              \
      const Scalar<DataVector>& subcell_tilde_tau,                            \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau, \
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DataVector>& tilde_phi,                                    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                      \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos,   \
      const Mesh<3>& dg_mesh, const TciOptions& tci_options,                  \
      const double persson_exponent);
GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
     grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
     grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl),
    (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef RECOVERY
}  // namespace grmhd::ValenciaDivClean::subcell
