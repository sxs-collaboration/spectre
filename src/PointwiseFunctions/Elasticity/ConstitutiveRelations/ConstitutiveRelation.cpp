// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity {
namespace ConstitutiveRelations {

template <size_t Dim>
tnsr::II<DataVector, Dim> ConstitutiveRelation<Dim>::stress(
    const tnsr::iJ<DataVector, Dim>& grad_displacement,
    const tnsr::I<DataVector, Dim>& x) const noexcept {
  auto strain =
      make_with_value<tnsr::ii<DataVector, Dim>>(grad_displacement, 0.);
  for (size_t i = 0; i < Dim; i++) {
    // Diagonal elements
    strain.get(i, i) = grad_displacement.get(i, i);
    // Symmetric off-diagonal elements
    for (size_t j = 0; j < i; j++) {
      strain.get(i, j) =
          0.5 * (grad_displacement.get(i, j) + grad_displacement.get(j, i));
    }
  }
  return stress(std::move(strain), x);
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ConstitutiveRelation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace ConstitutiveRelations
}  // namespace Elasticity
