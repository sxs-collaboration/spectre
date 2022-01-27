// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Tags {
/*!
 * \brief The square of the scalar field \f$\Psi\f$.
 */
struct PsiSquared : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Compute tag that calculates the square of the scalar field \f$\Psi\f$.
 */
struct PsiSquaredCompute : PsiSquared, db::ComputeTag {
  using base = PsiSquared;
  using return_type = Scalar<DataVector>;
  using argument_tags = tmpl::list<CurvedScalarWave::Tags::Psi>;
  static void function(const gsl::not_null<Scalar<DataVector>*> psi_squared,
                       const Scalar<DataVector>& psi) {
    get(*psi_squared) = get(psi) * get(psi);
  }
};
}  // namespace CurvedScalarWave::Tags
