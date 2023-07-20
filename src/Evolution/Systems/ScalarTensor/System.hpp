// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarTensor/Characteristics.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/TimeDerivative.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the first-order scalar tensor system.
 */
namespace ScalarTensor {
/*!
 * \brief Scalar Tensor system obtained from combining the CurvedScalarWave and
 * gh systems.
 *
 * \details The evolution equations follow from
 * \f{align*}{
 *  R_{ab} &= 8 \pi \, T^{(\Psi, \text{TR})}_{ab} ~, \\
 *  \Box \Psi &= 0~,
 * \f}
 *
 * where \f$\Psi\f$ is the scalar field and the trace-reversed stress-energy
 * tensor of the scalar field is given by
 * \f{align*}{
    T^{(\Psi, \text{TR})}_{ab}
 *      &\equiv T^{(\Psi)}_{ab} - \frac{1}{2} g_{ab} g^{cd} T^{(\Psi)}_{cd} \\
 *      &= \partial_a \Psi \partial_b \Psi ~.
 * \f}
 *
 * Both systems are recast as first-order systems in terms of the variables
 * \f{align*}{
 * & g_{ab}~,                                                           \\
 * & \Pi_{ab} = - \dfrac{1}{\alpha} \left( \partial_t g_{ab} - \beta^k
 * \partial_k g_{ab} \right)~,                                          \\
 * & \Phi_{iab} = \partial_i g_{ab}~,                                   \\
 * & \Psi~,                                                             \\
 * & \Pi = - \dfrac{1}{\alpha} \left(\partial_t \Psi - \beta^k
 * \partial_k \Psi \right)~,                                            \\
 * & \Phi_i = \partial_i \Psi~,
 * \f}
 *
 * where \f$ \alpha \f$ and \f$ \beta^k \f$ are the lapse and shift.
 *
 * The computation of the evolution equations is implemented in each system in
 * gh::TimeDerivative and CurvedScalarWave::TimeDerivative, respectively. We
 * take the additional step of adding the contribution of the trace-reversed
 * stress-energy tensor to the evolution equations of the metric.
 *
 * \note Although both systems are templated in the spatial dimension, we
 * only implement this system in three spatial dimensions.
 */
struct System {
  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 3;

  using gh_system = gh::System<3_st>;
  using scalar_system = CurvedScalarWave::System<3_st>;

  using variables_tag = ::Tags::Variables<
      tmpl::append<typename gh_system::variables_tag::tags_list,
                   typename scalar_system::variables_tag::tags_list>>;

  using flux_variables = tmpl::append<typename gh_system::flux_variables,
                                      typename scalar_system::flux_variables>;

  using gradient_variables =
      tmpl::append<typename gh_system::gradient_variables,
                   typename scalar_system::gradient_variables>;
  using gradients_tags = gradient_variables;

  static constexpr bool is_in_flux_conservative_form = false;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed<>;

  using compute_volume_time_derivative_terms = ScalarTensor::TimeDerivative;
  using inverse_spatial_metric_tag =
      typename gh_system::inverse_spatial_metric_tag;
};

}  // namespace ScalarTensor
