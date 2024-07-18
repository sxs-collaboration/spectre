// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor {

/// @{
/*!
 * \brief The scalar charge per unit solid angle.
 *
 * \details This function calculates the integrand of:
 * \f{align*}
 * q = - \dfrac{1}{4 \pi} \oint dA \Phi_i n^{i},
 * \f}
 * where \f$ n^{i} \f$ is the unit (outward) normal of the surface.
 *
 * For a spherically symmetric scalar, this value will coincide with the value
 * as extracted from the $r^{-1}$ decay in
 * \f[ \Psi \sim \phi_\infty + q / r + \cdots~. \f]
 */
void scalar_charge_integrand(const gsl::not_null<Scalar<DataVector>*> result,
                             const tnsr::i<DataVector, 3>& phi,
                             const tnsr::I<DataVector, 3>& unit_normal_vector);
/// @}

} // namespace ScalarTensor

namespace ScalarTensor::StrahlkorperScalar::Tags {

/*!
 * \brief The scalar charge per unit area.
 *
 * \details This tag holds the integrand of:
 * \f{align*}
 * q = - \dfrac{1}{4 \pi} \oint dA \Phi_i n^{i},
 * \f}
 * where \f$ n^{i} \f$ is the unit (outward) normal of the surface.
 *
 * For a spherically symmetric scalar, this value will coincide with the value
 * as extracted from the $r^{-1}$ decay in
 * \f[ \Psi \sim \phi_\infty + q / r + \cdots~. \f]
 */
struct ScalarChargeIntegrand : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Compute tag for the scalar charge per unit area.
 *
 * \see ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrand
 */
struct ScalarChargeIntegrandCompute : ScalarChargeIntegrand, db::ComputeTag {
  static constexpr size_t Dim = 3;
  using base = ScalarChargeIntegrand;
  static constexpr auto function = &ScalarTensor::scalar_charge_integrand;
  using argument_tags =
      tmpl::list<CurvedScalarWave::Tags::Phi<Dim>,
                 ylm::Tags::UnitNormalVector<Frame::Inertial>>;
  using return_type = Scalar<DataVector>;
};

}  // namespace ScalarTensor::StrahlkorperTags
