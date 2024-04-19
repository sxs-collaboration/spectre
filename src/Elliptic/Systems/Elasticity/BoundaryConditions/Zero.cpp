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

namespace Elasticity::BoundaryConditions {

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
std::string Zero<Dim, BoundaryConditionType>::name() {
  if constexpr (BoundaryConditionType ==
                elliptic::BoundaryConditionType::Dirichlet) {
    return "Fixed";
  } else {
    return "Free";
  }
}

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
void Zero<Dim, BoundaryConditionType>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress,
    const tnsr::iJ<DataVector, Dim>& /*deriv_displacement*/) {
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
void Zero<Dim, BoundaryConditionType>::apply_linearized(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress,
    const tnsr::iJ<DataVector, Dim>& deriv_displacement) {
  apply(displacement, n_dot_minus_stress, deriv_displacement);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define BCTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) template class Zero<DIM(data), BCTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3),
                        (elliptic::BoundaryConditionType::Dirichlet,
                         elliptic::BoundaryConditionType::Neumann))

#undef DIM
#undef BCTYPE
#undef INSTANTIATE

}  // namespace Elasticity::BoundaryConditions
