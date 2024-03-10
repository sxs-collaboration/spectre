// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler::Sources {
template <size_t Dim>
NoSource<Dim>::NoSource(CkMigrateMessage* msg) : Source<Dim>{msg} {}

template <size_t Dim>
void NoSource<Dim>::pup(PUP::er& p) {
  Source<Dim>::pup(p);
}

template <size_t Dim>
auto NoSource<Dim>::get_clone() const -> std::unique_ptr<Source<Dim>> {
  return std::make_unique<NoSource<Dim>>(*this);
}

template <size_t Dim>
void NoSource<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> /*source_mass_density_cons*/,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> /*source_momentum_density*/,
    const gsl::not_null<Scalar<DataVector>*> /*source_energy_density*/,
    const Scalar<DataVector>& /*mass_density_cons*/,
    const tnsr::I<DataVector, Dim>& /*momentum_density*/,
    const Scalar<DataVector>& /*energy_density*/,
    const tnsr::I<DataVector, Dim>& /*velocity*/,
    const Scalar<DataVector>& /*pressure*/,
    const Scalar<DataVector>& /*specific_internal_energy*/,
    const EquationsOfState::EquationOfState<false, 2>& /*eos*/,
    const tnsr::I<DataVector, Dim>& /*coords*/, const double /*time*/) const {}

template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID NoSource<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class NoSource<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace NewtonianEuler::Sources
