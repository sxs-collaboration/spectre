// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

namespace RadiationTransport {
namespace M1Grey {

namespace detail {
void compute_fluxes_impl(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_M,
    const Scalar<DataVector>& tilde_e,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::II<DataVector, 3, Frame::Inertial>& tilde_p,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        inv_spatial_metric) noexcept {
  constexpr size_t spatial_dim = 3;

  raise_or_lower_index(tilde_s_M, tilde_s, inv_spatial_metric);

  for (size_t i = 0; i < spatial_dim; ++i) {
    tilde_e_flux->get(i) =
        get(lapse) * tilde_s_M->get(i) - shift.get(i) * get(tilde_e);
    for (size_t j = 0; j < spatial_dim; ++j) {
      tilde_s_flux->get(i, j) = -shift.get(i) * tilde_s.get(j);
      for (size_t k = 0; k < spatial_dim; ++k) {
        tilde_s_flux->get(i, j) +=
            get(lapse) * tilde_p.get(i, k) * spatial_metric.get(j, k);
      }
    }
  }
}

}  // namespace detail
}  // namespace M1Grey
}  // namespace RadiationTransport
