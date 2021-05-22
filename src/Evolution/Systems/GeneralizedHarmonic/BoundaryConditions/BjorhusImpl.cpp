// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::BoundaryConditions::Bjorhus {
template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_psi(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_psi,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const std::array<DataType, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*bc_dt_v_psi)) !=
               get_size(get<0>(unit_interface_normal_vector)))) {
    *bc_dt_v_psi = tnsr::aa<DataType, VolumeDim, Frame::Inertial>{
        get_size(get<0>(unit_interface_normal_vector))};
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_v_psi->get(a, b) = char_speeds[0] *
                               unit_interface_normal_vector.get(0) *
                               three_index_constraint.get(0, a, b);
      for (size_t i = 1; i < VolumeDim; ++i) {
        bc_dt_v_psi->get(a, b) += char_speeds[0] *
                                  unit_interface_normal_vector.get(i) *
                                  three_index_constraint.get(i, a, b);
      }
    }
  }
}
}  // namespace GeneralizedHarmonic::BoundaryConditions::Bjorhus

// Explicit Instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                        \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::  \
      constraint_preserving_bjorhus_corrections_dt_v_psi(           \
          const gsl::not_null<                                      \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>   \
              bc_dt_v_psi,                                          \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&   \
              unit_interface_normal_vector,                         \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& \
              three_index_constraint,                               \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
