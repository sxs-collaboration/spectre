// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor {

/*!
 * \brief Add in the source term to the \f$\Pi\f$
 * evolved variable of the ::CurvedScalarWave system.
 *
 * \details The only source term in the wave equation
 * \f[
 *  \Box \Psi = \mathcal{S} ~,
 *  \f]
 *
 * is in the equation for \f$\Pi\f$:
 * \f[
 *  \partial_t \Pi + \text{\{spatial derivative
 * terms\}} = \alpha \mathcal{S}
 * ~,
 * \f]
 *
 * where \f$\mathcal{S}\f$ is the source term (e. g. in the Klein-Gordon
 * equation, the source term is the derivative of the scalar potential
 * \f$\mathcal{S} \equiv \partial V / \partial \Psi \f$.)
 *
 * This function adds that contribution to the existing value of `dt_pi_scalar`.
 * The wave equation terms in the scalar equation should be computed before
 * passing the `dt_pi_scalar` to this function for updating.
 *
 * \param dt_pi_scalar Time derivative terms of $\Pi$. The sourceless part
 * should be computed before with ::CurvedScalarWave::TimeDerivative.
 * \param scalar_source Source term $\mathcal{S}$ for the scalar equation.
 * \param lapse Lapse $\alpha$.
 *
 * \see `CurvedScalarWave::TimeDerivative` for details about the source-less
 * part of the time derivative calculation.
 */
void add_scalar_source_to_dt_pi_scalar(
    gsl::not_null<Scalar<DataVector>*> dt_pi_scalar,
    const Scalar<DataVector>& scalar_source, const Scalar<DataVector>& lapse);

/*!
 * \brief Computes the source term given by the mass of the scalar.
 *
 * \details For a scalar field with mass parameter \f$ m_\Psi \f$,
 * the wave equation takes the form
 * \f[
 *   \Box \Psi = \mathcal{S} ~,
 * \f]
 *
 * where the source is given by
 * \f[
 *   \mathcal{S} \equiv m^2_\Psi \Psi~.
 * \f]
 *
 * Here the mass parameter value is an option that needs to be specified in the
 * input file.
 *
 * \param scalar_source Source term $\mathcal{S}$ for the scalar equation.
 * \param psi Scalar field $\Psi$.
 * \param mass_psi Mass of the scalar field $m_\Psi$.
 *
 * \see `ScalarTensor::Tags::ScalarMass` for details about the mass.
 */
void mass_source(gsl::not_null<Scalar<DataVector>*> scalar_source,
                 const Scalar<DataVector>& psi, const double mass_psi);

namespace Tags {

/*!
 * \brief Compute tag for the scalar source.
 *
 * \details Compute the scalar source from data box items using
 * `mass_source`.
 */
struct ScalarSourceCompute : ScalarSource, db::ComputeTag {
  using argument_tags =
      tmpl::list<CurvedScalarWave::Tags::Psi, ScalarTensor::Tags::ScalarMass>;
  using return_type = Scalar<DataVector>;
  static constexpr void (*function)(const gsl::not_null<return_type*> result,
                                    const Scalar<DataVector>&,
                                    const double) = &mass_source;
  using base = ScalarSource;
};

}  // namespace Tags
}  // namespace ScalarTensor
