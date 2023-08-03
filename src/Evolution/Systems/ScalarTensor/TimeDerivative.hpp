// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor {
/*!
 * \brief Compute the RHS terms of the evolution equations for the scalar tensor
 * system.
 *
 * \details The bulk of the computations in this class dispatch to
 * `GeneralizedHarmonic::TimeDerivative` and
 * `CurvedScalarWave::TimeDerivative` as a 'product system' -- each
 * independently operating on its own subset of the supplied variable
 * collections.
 * The additional steps taken are to compute the trace-reversed stress energy
 * tensor associated with the scalar part of the system and add its contribution
 * to the \f$\partial_t \Pi_{a b}\f$ variable in the Generalized Harmonic
 * system, as well as adding any scalar sources to the variable \f$\partial_t
 * \Pi\f$.
 */
struct TimeDerivative {
  static constexpr size_t dim = 3;
  using gh_dt_tags =
      db::wrap_tags_in<::Tags::dt,
                       typename gh::System<dim>::variables_tag::tags_list>;
  using scalar_dt_tags = db::wrap_tags_in<
      ::Tags::dt,
      typename CurvedScalarWave::System<dim>::variables_tag::tags_list>;
  using dt_tags = tmpl::append<gh_dt_tags, scalar_dt_tags>;
  using gh_temp_tags = typename gh::TimeDerivative<dim>::temporary_tags;
  using gh_gradient_tags = typename gh::System<dim>::gradients_tags;
  using gh_arg_tags = typename gh::TimeDerivative<dim>::argument_tags;
  using scalar_temp_tags =
      typename CurvedScalarWave::TimeDerivative<dim>::temporary_tags;
  using scalar_extra_temp_tags =
      tmpl::list<ScalarTensor::Tags::TraceReversedStressEnergy<
          DataVector, dim, ::Frame::Inertial>>;
  using scalar_gradient_tags =
      typename CurvedScalarWave::System<dim>::gradients_tags;
  using gradient_tags = tmpl::append<gh_gradient_tags, scalar_gradient_tags>;
  using scalar_arg_tags =
      typename CurvedScalarWave::TimeDerivative<dim>::argument_tags;
  using temporary_tags = tmpl::remove_duplicates<
      tmpl::append<gh_temp_tags, scalar_temp_tags, scalar_extra_temp_tags>>;
  using argument_tags =
      tmpl::append<gh_arg_tags, scalar_arg_tags,
                   tmpl::list<ScalarTensor::Tags::ScalarSource>>;

  static void apply(
      // GH dt variables
      gsl::not_null<tnsr::aa<DataVector, dim>*> dt_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, dim>*> dt_pi,
      gsl::not_null<tnsr::iaa<DataVector, dim>*> dt_phi,
      // Scalar dt variables
      gsl::not_null<Scalar<DataVector>*> dt_psi_scalar,
      gsl::not_null<Scalar<DataVector>*> dt_pi_scalar,
      gsl::not_null<tnsr::i<DataVector, dim, Frame::Inertial>*> dt_phi_scalar,

      // GH temporal variables
      gsl::not_null<Scalar<DataVector>*> temp_gamma1,
      gsl::not_null<Scalar<DataVector>*> temp_gamma2,
      gsl::not_null<tnsr::a<DataVector, dim>*> temp_gauge_function,
      gsl::not_null<tnsr::ab<DataVector, dim>*>
          temp_spacetime_deriv_gauge_function,
      gsl::not_null<Scalar<DataVector>*> gamma1gamma2,
      gsl::not_null<Scalar<DataVector>*> half_half_pi_two_normals,
      gsl::not_null<Scalar<DataVector>*> normal_dot_gauge_constraint,
      gsl::not_null<Scalar<DataVector>*> gamma1_plus_1,
      gsl::not_null<tnsr::a<DataVector, dim>*> pi_one_normal,
      gsl::not_null<tnsr::a<DataVector, dim>*> gauge_constraint,
      gsl::not_null<tnsr::i<DataVector, dim>*> half_phi_two_normals,
      gsl::not_null<tnsr::aa<DataVector, dim>*>
          shift_dot_three_index_constraint,
      gsl::not_null<tnsr::aa<DataVector, dim>*>
          mesh_velocity_dot_three_index_constraint,
      gsl::not_null<tnsr::ia<DataVector, dim>*> phi_one_normal,
      gsl::not_null<tnsr::aB<DataVector, dim>*> pi_2_up,
      gsl::not_null<tnsr::iaa<DataVector, dim>*> three_index_constraint,
      gsl::not_null<tnsr::Iaa<DataVector, dim>*> phi_1_up,
      gsl::not_null<tnsr::iaB<DataVector, dim>*> phi_3_up,
      gsl::not_null<tnsr::abC<DataVector, dim>*> christoffel_first_kind_3_up,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, dim>*> shift,
      gsl::not_null<tnsr::II<DataVector, dim>*> inverse_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> det_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> sqrt_det_spatial_metric,
      gsl::not_null<tnsr::AA<DataVector, dim>*> inverse_spacetime_metric,
      gsl::not_null<tnsr::abb<DataVector, dim>*> christoffel_first_kind,
      gsl::not_null<tnsr::Abb<DataVector, dim>*> christoffel_second_kind,
      gsl::not_null<tnsr::a<DataVector, dim>*> trace_christoffel,
      gsl::not_null<tnsr::A<DataVector, dim>*> normal_spacetime_vector,

      // Scalar temporal variables
      gsl::not_null<Scalar<DataVector>*> result_gamma1_scalar,
      gsl::not_null<Scalar<DataVector>*> result_gamma2_scalar,

      // Extra temporal tags
      gsl::not_null<tnsr::aa<DataVector, dim>*> stress_energy,

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
      const tnsr::aa<DataVector, dim>& pi,
      const tnsr::iaa<DataVector, dim>& phi, const Scalar<DataVector>& gamma0,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
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

      const Scalar<DataVector>& scalar_source);
};
}  // namespace ScalarTensor
