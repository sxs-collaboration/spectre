// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/BoundaryConditions/Zero.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Elasticity::BoundaryConditions::detail {

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
std::string ZeroImpl<Dim, BoundaryConditionType>::name() {
  if constexpr (BoundaryConditionType ==
                elliptic::BoundaryConditionType::Dirichlet) {
    return "Fixed";
  } else {
    return "Free";
  }
}

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
void ZeroImpl<Dim, BoundaryConditionType>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress) {
  if constexpr (BoundaryConditionType ==
                elliptic::BoundaryConditionType::Dirichlet) {
    (void)n_dot_minus_stress;
    for (size_t d = 0; d < Dim; ++d) {
      displacement->get(d) = 0.;
    }
  } else {
    (void)displacement;
    for (size_t d = 0; d < Dim; ++d) {
      n_dot_minus_stress->get(d) = 0.;
    }
  }
}

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
void ZeroImpl<Dim, BoundaryConditionType>::apply_linearized(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress) {
  apply(displacement, n_dot_minus_stress);
}

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
bool operator==(const ZeroImpl<Dim, BoundaryConditionType>& /*lhs*/,
                const ZeroImpl<Dim, BoundaryConditionType>& /*rhs*/) {
  return true;
}

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
bool operator!=(const ZeroImpl<Dim, BoundaryConditionType>& lhs,
                const ZeroImpl<Dim, BoundaryConditionType>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define BCTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template class ZeroImpl<DIM(data), BCTYPE(data)>;                       \
  template bool operator==(const ZeroImpl<DIM(data), BCTYPE(data)>& lhs,  \
                           const ZeroImpl<DIM(data), BCTYPE(data)>& rhs); \
  template bool operator!=(const ZeroImpl<DIM(data), BCTYPE(data)>& lhs,  \
                           const ZeroImpl<DIM(data), BCTYPE(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3),
                        (elliptic::BoundaryConditionType::Dirichlet,
                         elliptic::BoundaryConditionType::Neumann))

#undef DIM
#undef BCTYPE
#undef INSTANTIATE

}  // namespace Elasticity::BoundaryConditions::detail
