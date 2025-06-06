// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingParameters.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/Tags.hpp"
#include "PointwiseFunctions/ScalarTensor/SourceTags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor {
/// @{
/*!
 * \brief Computes the source term given by the coupling of the scalar to
 * curvature.
 *
 * \details For a scalar field with mass parameter $ m_\Psi $,
 * the wave equation takes the form
 * \begin{align}
 *   \Box \Psi = \mathcal{S} ~,
 * \end{align}
 *
 * where the source is given by
 * \begin{align}
 *   \mathcal{S} \equiv m^2_\Psi \Psi - f'(\Psi) \mathcal{G}~,
 * \end{align}
 * where
 * \begin{align}
 *   \mathcal{G} \equiv 8 (E_{ab} E^{ab} - B_{ab} B^{ab}) ~,
 * \end{align}
 * is the Gauss-Bonnet scalar and the coupling function is given by
 * \begin{align}
 *   f(\Psi) \equiv \lambda \Psi
 *      + \dfrac{1}{16} \left( \eta \Psi^2 + 2 \zeta \Psi^4 \right) ~,
 * \end{align}
 * Here the Gauss-Bonnet scalar (in vacuum) is given in terms of the electric
 * ($ E_{ab} $) and magnetic ($ B_{ab} $) parts of the Weyl scalar.
 *
 */
void gauss_bonnet_scalar_source(
    gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar,
    const Scalar<DataVector>& psi,
    const CouplingParameterOptions& coupling_parameters, double mass_psi,
    std::pair<double, double> start_and_ramp_times, double time);

Scalar<DataVector> gauss_bonnet_scalar_source(
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar,
    const Scalar<DataVector>& psi,
    const CouplingParameterOptions& coupling_parameters, double mass_psi,
    std::pair<double, double> start_and_ramp_times, double time);
/// @}

/*!
 * \brief Multiplies by the coupling function.
 *
 * \details Multiply by the first derivative of the coupling function given by
 * \begin{align}
 *   f(\Psi) \equiv
 *      + \dfrac{1}{16} \left( 4 \lambda \Psi + 2 \eta \Psi^2 + \zeta \Psi^4
 * \right) ~.
 * \end{align}
 *
 */
void multiply_by_negative_deriv_of_coupling_func(
    gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& psi,
    const CouplingParameterOptions& coupling_parameters,
    std::pair<double, double> start_and_ramp_times, double time);

/*!
 * \brief Multiplies by the coupling function.
 *
 * \details Multiply by the second derivative of the coupling function given by
 * \begin{align}
 *   f(\Psi) \equiv
 *      + \dfrac{1}{16} \left( 4 \lambda \Psi + 2 \eta \Psi^2 + \zeta \Psi^4
 * \right) ~.
 * \end{align}
 *
 */
void multiply_by_negative_second_deriv_of_coupling_func(
    gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& psi,
    const CouplingParameterOptions& coupling_parameters,
    std::pair<double, double> start_and_ramp_times, double time);

namespace Tags {
/*!
 * \copydoc ScalarTensor::gauss_bonnet_scalar_source
 */
struct ScalarSourceCompute : ScalarSource, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>, CurvedScalarWave::Tags::Psi,
      ScalarTensor::Tags::CouplingParameters, ScalarTensor::Tags::ScalarMass,
      ScalarTensor::Tags::RampUpParameters, ::Tags::Time>;
  using return_type = Scalar<DataVector>;
  static constexpr void (*function)(
      const gsl::not_null<Scalar<DataVector>*> /* scalar_source */,
      const Scalar<DataVector>& /* weyl_electric_scalar */,
      const Scalar<DataVector>& /* weyl_magnetic_scalar */,
      const Scalar<DataVector>& /* psi */,
      const CouplingParameterOptions& /* coupling_parameters */,
      const double /* mass_psi */,
      const std::pair<double, double> /* start_and_ramp_times */,
      const double /* time */) = &gauss_bonnet_scalar_source;
  using base = ScalarSource;
};
}  // namespace Tags

}  // namespace ScalarTensor
