// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/TimeDerivative.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Evolution/Systems/ScalarTensor/Sources/ScalarSource.hpp"
#include "Evolution/Systems/ScalarTensor/StressEnergy.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor {
void TimeDerivative::apply(
    // GH dt variables
    const gsl::not_null<tnsr::aa<DataVector, dim>*> dt_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, dim>*> dt_pi,
    const gsl::not_null<tnsr::iaa<DataVector, dim>*> dt_phi,
    // Scalar dt variables
    const gsl::not_null<Scalar<DataVector>*> dt_psi_scalar,
    const gsl::not_null<Scalar<DataVector>*> dt_pi_scalar,
    const gsl::not_null<tnsr::i<DataVector, dim, Frame::Inertial>*>
        dt_phi_scalar,

    // GH temporal variables
    const gsl::not_null<Scalar<DataVector>*> temp_gamma1,
    const gsl::not_null<Scalar<DataVector>*> temp_gamma2,
    const gsl::not_null<tnsr::a<DataVector, dim>*> temp_gauge_function,
    const gsl::not_null<tnsr::ab<DataVector, dim>*>
        temp_spacetime_deriv_gauge_function,
    const gsl::not_null<Scalar<DataVector>*> gamma1gamma2,
    const gsl::not_null<Scalar<DataVector>*> half_half_pi_two_normals,
    const gsl::not_null<Scalar<DataVector>*> normal_dot_gauge_constraint,
    const gsl::not_null<Scalar<DataVector>*> gamma1_plus_1,
    const gsl::not_null<tnsr::a<DataVector, dim>*> pi_one_normal,
    const gsl::not_null<tnsr::a<DataVector, dim>*> gauge_constraint,
    const gsl::not_null<tnsr::i<DataVector, dim>*> half_phi_two_normals,
    const gsl::not_null<tnsr::aa<DataVector, dim>*>
        shift_dot_three_index_constraint,
    const gsl::not_null<tnsr::aa<DataVector, dim>*>
        mesh_velocity_dot_three_index_constraint,
    const gsl::not_null<tnsr::ia<DataVector, dim>*> phi_one_normal,
    const gsl::not_null<tnsr::aB<DataVector, dim>*> pi_2_up,
    const gsl::not_null<tnsr::iaa<DataVector, dim>*> three_index_constraint,
    const gsl::not_null<tnsr::Iaa<DataVector, dim>*> phi_1_up,
    const gsl::not_null<tnsr::iaB<DataVector, dim>*> phi_3_up,
    const gsl::not_null<tnsr::abC<DataVector, dim>*>
        christoffel_first_kind_3_up,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, dim>*> shift,
    const gsl::not_null<tnsr::II<DataVector, dim>*> inverse_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> det_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> sqrt_det_spatial_metric,
    const gsl::not_null<tnsr::AA<DataVector, dim>*> inverse_spacetime_metric,
    const gsl::not_null<tnsr::abb<DataVector, dim>*> christoffel_first_kind,
    const gsl::not_null<tnsr::Abb<DataVector, dim>*> christoffel_second_kind,
    const gsl::not_null<tnsr::a<DataVector, dim>*> trace_christoffel,
    const gsl::not_null<tnsr::A<DataVector, dim>*> normal_spacetime_vector,

    // Scalar temporal variables
    const gsl::not_null<Scalar<DataVector>*> result_gamma1_scalar,
    const gsl::not_null<Scalar<DataVector>*> result_gamma2_scalar,

    // Extra temporal tags
    const gsl::not_null<tnsr::aa<DataVector, dim>*> stress_energy,

    // GH spatial derivatives
    const tnsr::iaa<DataVector, dim>& d_spacetime_metric,
    const tnsr::iaa<DataVector, dim>& d_pi,
    const tnsr::ijaa<DataVector, dim>& d_phi,

    // scalar spatial derivatives
    const tnsr::i<DataVector, dim>& d_psi_scalar,
    const tnsr::i<DataVector, dim>& d_pi_scalar,
    const tnsr::ij<DataVector, dim>& d_phi_scalar,

    // GH argument variables
    const tnsr::aa<DataVector, dim>& spacetime_metric,
    const tnsr::aa<DataVector, dim>& pi, const tnsr::iaa<DataVector, dim>& phi,
    const Scalar<DataVector>& gamma0, const Scalar<DataVector>& gamma1,
    const Scalar<DataVector>& gamma2,
    const gh::gauges::GaugeCondition& gauge_condition, const Mesh<dim>& mesh,
    double time,
    const tnsr::I<DataVector, dim, Frame::Inertial>& inertial_coords,
    const InverseJacobian<DataVector, dim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const std::optional<tnsr::I<DataVector, dim, Frame::Inertial>>&
        mesh_velocity,

    // Scalar argument variables
    const Scalar<DataVector>& pi_scalar,
    const tnsr::i<DataVector, dim>& phi_scalar,
    const Scalar<DataVector>& lapse_scalar,
    const tnsr::I<DataVector, dim>& shift_scalar,
    const tnsr::i<DataVector, dim>& deriv_lapse,
    const tnsr::iJ<DataVector, dim>& deriv_shift,
    const tnsr::II<DataVector, dim>& upper_spatial_metric,
    const tnsr::I<DataVector, dim>& trace_spatial_christoffel,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& gamma1_scalar,
    const Scalar<DataVector>& gamma2_scalar,

    const Scalar<DataVector>& scalar_source) {
  // Compute sourceless part of the RHS of the metric equations
  gh::TimeDerivative<dim>::apply(
      // GH dt variables
      dt_spacetime_metric, dt_pi, dt_phi,

      // GH temporal variables
      temp_gamma1, temp_gamma2, temp_gauge_function,
      temp_spacetime_deriv_gauge_function, gamma1gamma2,
      half_half_pi_two_normals, normal_dot_gauge_constraint, gamma1_plus_1,
      pi_one_normal, gauge_constraint, half_phi_two_normals,
      shift_dot_three_index_constraint,
      mesh_velocity_dot_three_index_constraint, phi_one_normal, pi_2_up,
      three_index_constraint, phi_1_up, phi_3_up, christoffel_first_kind_3_up,
      lapse, shift, inverse_spatial_metric, det_spatial_metric,
      sqrt_det_spatial_metric, inverse_spacetime_metric, christoffel_first_kind,
      christoffel_second_kind, trace_christoffel, normal_spacetime_vector,

      // GH argument variables
      d_spacetime_metric, d_pi, d_phi, spacetime_metric, pi, phi, gamma0,
      gamma1, gamma2, gauge_condition, mesh, time, inertial_coords,
      inverse_jacobian, mesh_velocity);

  // Compute sourceless part of the RHS of the scalar equation
  CurvedScalarWave::TimeDerivative<dim>::apply(
      // Scalar dt variables
      dt_psi_scalar, dt_pi_scalar, dt_phi_scalar,

      // Scalar temporal variables
      lapse, shift, inverse_spatial_metric,

      result_gamma1_scalar, result_gamma2_scalar,

      // Scalar argument variables
      d_psi_scalar, d_pi_scalar, d_phi_scalar, pi_scalar, phi_scalar,

      lapse_scalar, shift_scalar,

      deriv_lapse, deriv_shift, upper_spatial_metric, trace_spatial_christoffel,
      trace_extrinsic_curvature, gamma1_scalar, gamma2_scalar);

  // Compute the (trace-reversed) stress energy tensor here
  trace_reversed_stress_energy(stress_energy, pi_scalar, phi_scalar,
                               lapse_scalar, shift_scalar);

  add_stress_energy_term_to_dt_pi(dt_pi, *stress_energy, lapse_scalar);

  add_scalar_source_to_dt_pi_scalar(dt_pi_scalar, scalar_source, lapse_scalar);
}
}  // namespace ScalarTensor
