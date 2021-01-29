// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::ConstitutiveRelations {

/// \cond
template <size_t Dim>
void ConstitutiveRelation<Dim>::stress(
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> stress,
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& x) const noexcept {
  tnsr::II<DataVector, Dim> symmetric_stress{x.begin()->size()};
  this->stress(make_not_null(&symmetric_stress), strain, x);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      stress->get(i, j) = symmetric_stress.get(i, j);
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ConstitutiveRelation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace Elasticity::ConstitutiveRelations
