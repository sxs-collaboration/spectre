// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/FreeOutflow.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
FreeOutflow::FreeOutflow(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
FreeOutflow::get_clone() const {
  return std::make_unique<FreeOutflow>(*this);
}

void FreeOutflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID FreeOutflow::my_PUP_ID = 0;

std::optional<std::string> FreeOutflow::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_vector,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_enthalpy,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,

    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,
    const Scalar<DataVector>& interior_sqrt_det_spatial_metric) {
  const size_t number_of_grid_points = get(interior_rest_mass_density).size();

  // temp buffer to store
  //  * n_i v^i
  //  * v_{ghost}^i
  //  * divergence cleaning field on ghost zone
  Variables<tmpl::list<::Tags::TempScalar<0>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       ::Tags::TempScalar<1>>>
      temp_buffer{number_of_grid_points};
  auto& normal_dot_interior_spatial_velocity =
      get<::Tags::TempScalar<0>>(temp_buffer);
  auto& exterior_spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(temp_buffer);
  auto& exterior_divergence_cleaning_field =
      get<::Tags::TempScalar<1>>(temp_buffer);

  get(*lapse) = get(interior_lapse);
  for (size_t i = 0; i < 3; ++i) {
    (*shift).get(i) = interior_shift.get(i);
    for (size_t j = 0; j < 3; ++j) {
      (*inv_spatial_metric).get(i, j) = interior_inv_spatial_metric.get(i, j);
    }
  }

  // copy-paste interior spatial velocity to exterior spatial velocity, but
  // kill ingoing normal component to zero
  dot_product(make_not_null(&normal_dot_interior_spatial_velocity),
              outward_directed_normal_covector, interior_spatial_velocity);
  for (size_t i = 0; i < number_of_grid_points; ++i) {
    if (get(normal_dot_interior_spatial_velocity)[i] >= 0.0) {
      for (size_t spatial_index = 0; spatial_index < 3; ++spatial_index) {
        exterior_spatial_velocity.get(spatial_index)[i] =
            interior_spatial_velocity.get(spatial_index)[i];
      }
    } else {
      for (size_t spatial_index = 0; spatial_index < 3; ++spatial_index) {
        exterior_spatial_velocity.get(spatial_index)[i] =
            interior_spatial_velocity.get(spatial_index)[i] -
            get(normal_dot_interior_spatial_velocity)[i] *
                outward_directed_normal_vector.get(spatial_index)[i];
      }
    }
  }

  get(exterior_divergence_cleaning_field) = 0.0;

  ConservativeFromPrimitive::apply(
      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
      interior_rest_mass_density, interior_specific_internal_energy,
      interior_specific_enthalpy, interior_pressure, exterior_spatial_velocity,
      interior_lorentz_factor, interior_magnetic_field,
      interior_sqrt_det_spatial_metric, interior_spatial_metric,
      exterior_divergence_cleaning_field);

  ComputeFluxes::apply(
      tilde_d_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      *tilde_d, *tilde_tau, *tilde_s, *tilde_b, *tilde_phi, *lapse, *shift,
      interior_sqrt_det_spatial_metric, interior_spatial_metric,
      *inv_spatial_metric, interior_pressure, exterior_spatial_velocity,
      interior_lorentz_factor, interior_magnetic_field);

  return {};
}

}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
