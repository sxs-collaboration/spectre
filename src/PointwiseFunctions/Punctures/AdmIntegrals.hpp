// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Punctures {

/// @{
/*!
 * \brief The volume integrand for the ADM mass $M_\mathrm{ADM}$
 *
 * The ADM mass for Punctures is (Eq. (12.56) in \cite BaumgarteShapiro)
 *
 * \begin{equation}
 * M_\mathrm{ADM} = \sum_I M_I + \frac{1}{2\pi} \int
 *   \beta \left(\alpha \left(1 + u\right) + 1\right)^{-7}
 * \end{equation}
 *
 * \see Punctures
 */
void adm_mass_integrand(const gsl::not_null<Scalar<DataVector>*> result,
                        const Scalar<DataVector>& field,
                        const Scalar<DataVector>& alpha,
                        const Scalar<DataVector>& beta);

Scalar<DataVector> adm_mass_integrand(const Scalar<DataVector>& field,
                                      const Scalar<DataVector>& alpha,
                                      const Scalar<DataVector>& beta);
/// @}

namespace Tags {

/// @{
struct AdmMassIntegrand : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The volume integrand for the ADM mass $M_\mathrm{ADM}$
 *
 * \see adm_mass_integrand
 */
struct AdmMassIntegrandCompute : AdmMassIntegrand, db::ComputeTag {
  using base = AdmMassIntegrand;
  using argument_tags = tmpl::list<Field, Alpha, Beta>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const Scalar<DataVector>&, const Scalar<DataVector>&,
                           const Scalar<DataVector>&)>(&adm_mass_integrand);
};
/// @}

}  // namespace Tags
}  // namespace Punctures
