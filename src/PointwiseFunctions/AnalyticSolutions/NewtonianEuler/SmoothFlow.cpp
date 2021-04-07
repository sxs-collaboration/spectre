// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"

namespace NewtonianEuler::Solutions {

template <size_t Dim>
SmoothFlow<Dim>::SmoothFlow(const std::array<double, Dim>& mean_velocity,
                            const std::array<double, Dim>& wavevector,
                            const double pressure, const double adiabatic_index,
                            const double perturbation_size) noexcept
    : smooth_flow{mean_velocity, wavevector, pressure, adiabatic_index,
                  perturbation_size} {}

template <size_t Dim>
SmoothFlow<Dim>::SmoothFlow(CkMigrateMessage* msg) noexcept
    : smooth_flow(msg) {}

template <size_t Dim>
void SmoothFlow<Dim>::pup(PUP::er& p) noexcept {
  smooth_flow::pup(p);
}

template <size_t Dim>
bool operator==(const SmoothFlow<Dim>& lhs,
                const SmoothFlow<Dim>& rhs) noexcept {
  using smooth_flow = hydro::Solutions::SmoothFlow<Dim, false>;
  return *static_cast<const smooth_flow*>(&lhs) ==
         *static_cast<const smooth_flow*>(&rhs);
}

template <size_t Dim>
bool operator!=(const SmoothFlow<Dim>& lhs,
                const SmoothFlow<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_CLASS(_, data)                                 \
  template class SmoothFlow<DIM(data)>;                            \
  template bool operator==(const SmoothFlow<DIM(data)>&,           \
                           const SmoothFlow<DIM(data)>&) noexcept; \
  template bool operator!=(const SmoothFlow<DIM(data)>&,           \
                           const SmoothFlow<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (1, 2, 3))

#undef INSTANTIATE_CLASS
#undef DIM
}  // namespace NewtonianEuler::Solutions
