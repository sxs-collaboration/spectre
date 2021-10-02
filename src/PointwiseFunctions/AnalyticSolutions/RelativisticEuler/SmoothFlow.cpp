// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"

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

namespace RelativisticEuler::Solutions {

template <size_t Dim>
SmoothFlow<Dim>::SmoothFlow(const std::array<double, Dim>& mean_velocity,
                            const std::array<double, Dim>& wavevector,
                            const double pressure, const double adiabatic_index,
                            const double perturbation_size)
    : smooth_flow{mean_velocity, wavevector, pressure, adiabatic_index,
                  perturbation_size} {}

template <size_t Dim>
SmoothFlow<Dim>::SmoothFlow(CkMigrateMessage* msg) : smooth_flow(msg) {}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, Dim>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::MagneticField<DataType, Dim>> /*meta*/) const {
  return make_with_value<tnsr::I<DataType, Dim>>(get<0>(x), 0.0);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
SmoothFlow<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(get<0>(x), 0.0);
}

template <size_t Dim>
void SmoothFlow<Dim>::pup(PUP::er& p) {
  hydro::Solutions::SmoothFlow<Dim, true>::pup(p);
  p | background_spacetime_;
}

template <size_t Dim>
bool operator==(const SmoothFlow<Dim>& lhs, const SmoothFlow<Dim>& rhs) {
  using smooth_flow = hydro::Solutions::SmoothFlow<Dim, true>;
  return *static_cast<const smooth_flow*>(&lhs) ==
             *static_cast<const smooth_flow*>(&rhs) and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

template <size_t Dim>
bool operator!=(const SmoothFlow<Dim>& lhs, const SmoothFlow<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DATA_TYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_CLASS(_, data)                        \
  template class SmoothFlow<DIM(data)>;                   \
  template bool operator==(const SmoothFlow<DIM(data)>&,  \
                           const SmoothFlow<DIM(data)>&); \
  template bool operator!=(const SmoothFlow<DIM(data)>&,  \
                           const SmoothFlow<DIM(data)>&);

#define INSTANTIATE_FUNCTIONS(_, data)                                      \
  template tuples::TaggedTuple<                                             \
      hydro::Tags::DivergenceCleaningField<DATA_TYPE(data)>>                \
  SmoothFlow<DIM(data)>::variables(                                         \
      const tnsr::I<DATA_TYPE(data), DIM(data)>& x, double /*t*/,           \
      tmpl::list<                                                           \
          hydro::Tags::DivergenceCleaningField<DATA_TYPE(data)>> /*meta*/)  \
      const;                                                                \
  template tuples::TaggedTuple<                                             \
      hydro::Tags::MagneticField<DATA_TYPE(data), DIM(data)>>               \
  SmoothFlow<DIM(data)>::variables(                                         \
      const tnsr::I<DATA_TYPE(data), DIM(data)>& x, double /*t*/,           \
      tmpl::list<                                                           \
          hydro::Tags::MagneticField<DATA_TYPE(data), DIM(data)>> /*meta*/) \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_FUNCTIONS, (1, 2, 3), (double, DataVector))

#undef INSTANTIATE_CLASS
#undef INSTANTIATE_FUNCTIONS
#undef DIM
#undef DATA_TYPE
}  // namespace RelativisticEuler::Solutions
