// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::gauges {
Harmonic::Harmonic(CkMigrateMessage* const msg) : GaugeCondition(msg) {}

void Harmonic::pup(PUP::er& p) { GaugeCondition::pup(p); }

std::unique_ptr<GaugeCondition> Harmonic::get_clone() const {
  return std::make_unique<Harmonic>(*this);
}

template <size_t SpatialDim>
void Harmonic::gauge_and_spacetime_derivative(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*>
        gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
        d4_gauge_h,
    const double /*time*/,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& /*inertial_coords*/)
    const {
  for (auto& component : *gauge_h) {
    component = 0.0;
  }
  for (auto& component : *d4_gauge_h) {
    component = 0.0;
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Harmonic::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template void Harmonic::gauge_and_spacetime_derivative(                    \
      const gsl::not_null<tnsr::a<DataVector, DIM(data), Frame::Inertial>*>  \
          gauge_h,                                                           \
      const gsl::not_null<tnsr::ab<DataVector, DIM(data), Frame::Inertial>*> \
          d4_gauge_h,                                                        \
      double /*time*/,                                                       \
      const tnsr::I<DataVector, DIM(data),                                   \
                    Frame::Inertial>& /*inertial_coords*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
}  // namespace gh::gauges
