// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/BoundaryConditions/Neumann.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace IrrotationalBns::BoundaryConditions {

template <size_t Dim>
Neumann<Dim>::Neumann(const Options::Context& context){};

template <size_t Dim>
void Neumann<Dim>::apply(
    gsl::not_null<Scalar<DataVector>*> velocity_potential,
    gsl::not_null<Scalar<DataVector>*> n_dot_auxiliary_velocity) const {
  get(n_dot_auxiliary_velocity) = 0.0;
}

template <size_t Dim>
void Neumann<Dim>::apply_linearized(
    gsl::not_null<Scalar<DataVector>*> velocity_potential_correction,
    gsl::not_null<Scalar<DataVector>*> n_dot_auxiliary_velocity_correction)
    const {
  get(n_dot_auxiliary_velocity_correction) = 0.0;
}

template <size_t Dim>
void Neumann<Dim>::pup(PUP::er& p) {}

template <size_t Dim>
bool operator==(const Neumann<Dim>& lhs, const Neumann<Dim>& rhs) {
  return true;
}

template <size_t Dim>
bool operator!=(const Neumann<Dim>& lhs, const Neumann<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
PUP::able::PUP_ID Neumann<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                           \
  template class Neumann<DIM(data)>;                   \
  template bool operator==(const Neumann<DIM(data)>&,  \
                           const Neumann<DIM(data)>&); \
  template bool operator!=(const Neumann<DIM(data)>&,  \
                           const Neumann<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace IrrotationalBns::BoundaryConditions
