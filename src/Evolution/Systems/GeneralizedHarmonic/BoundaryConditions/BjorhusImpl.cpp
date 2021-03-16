// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
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

template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_zero(
    const gsl::not_null<tnsr::iaa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_zero,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const std::array<DataType, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*bc_dt_v_zero)) !=
               get_size(get<0>(unit_interface_normal_vector)))) {
    *bc_dt_v_zero = tnsr::iaa<DataType, VolumeDim, Frame::Inertial>{
        get_size(get<0>(unit_interface_normal_vector))};
  }
  std::fill(bc_dt_v_zero->begin(), bc_dt_v_zero->end(), 0.);

  if (LIKELY(VolumeDim == 3)) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        // Lets say this term is T2_{iab} := - n_l \beta^l n^j C_{jiab}.
        // But we store D_{iab} = LeviCivita^{ijk} dphi_{jkab},
        // and C_{ijab} = LeviCivita^{kij} D_{kab}
        // where D is `four_index_constraint`.
        // therefore, T2_{iab} =  char_speed<VZero> n^j C_{jiab}
        // (since char_speed<VZero> = - n_l \beta^l), and therefore:
        // T2_{iab} = char_speed<VZero> n^j LeviCivita^{ikj} D_{kab}.
        // Let LeviCivitaIterator be indexed by
        // it[0] <--> i,
        // it[1] <--> j,
        // it[2] <--> k, then
        // T2_{it[0], ab} += char_speed<VZero> n^it[2] it.sign() D_{it[1], ab};
        for (LeviCivitaIterator<VolumeDim> it; it; ++it) {
          bc_dt_v_zero->get(it[0], a, b) +=
              it.sign() * char_speeds[1] *
              unit_interface_normal_vector.get(it[2]) *
              four_index_constraint.get(it[1], a, b);
        }
      }
    }
  } else if (LIKELY(VolumeDim == 2)) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        // Lets say this term is T2_{kab} := - n_l \beta^l n^j C_{jkab}.
        // In 2+1 spacetime, we store the four index constraint to
        // be D_{1ab} = C_{12ab}, C_{2ab} = C_{21ab}. Therefore,
        // T_{kab} = -n_l \beta^l (n^1 C_{1kab} + n^2 C_{2kab}), i.e.
        // T_{1ab} = -n_l \beta^l n^2 D_{2ab}, T_{2ab} = -n_l \beta^l n^1
        // D_{1ab}.
        bc_dt_v_zero->get(0, a, b) +=
            char_speeds[1] * (unit_interface_normal_vector.get(1) *
                              four_index_constraint.get(1, a, b));
        bc_dt_v_zero->get(1, a, b) +=
            char_speeds[1] * (unit_interface_normal_vector.get(0) *
                              four_index_constraint.get(0, a, b));
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
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;  \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::  \
      constraint_preserving_bjorhus_corrections_dt_v_zero(          \
          const gsl::not_null<                                      \
              tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>*>  \
              bc_dt_v_zero,                                         \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&   \
              unit_interface_normal_vector,                         \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& \
              four_index_constraint,                                \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
