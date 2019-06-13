// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/M1HydroCoupling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace RadiationTransport {
namespace M1Grey {

namespace detail {
void compute_m1_hydro_coupling_impl(
    const gsl::not_null<Scalar<DataVector>*> source_n,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> source_i,
    const Scalar<DataVector>& emissivity,
    const Scalar<DataVector>& absorption_opacity,
    const Scalar<DataVector>& scattering_opacity,
    const Scalar<DataVector>& comoving_energy_density,
    const Scalar<DataVector>& comoving_momentum_density_normal,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        comoving_momentum_density_spatial,
    const tnsr::I<DataVector, 3, Frame::Inertial>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) noexcept {
  Variables<tmpl::list<
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>>
      temp_tensors(get(lapse).size());
  // Dimension of spatial tensors
  constexpr size_t spatial_dim = 3;

  auto& eta_minus_kJ = get(sqrt_det_spatial_metric) * get(emissivity) -
                       get(absorption_opacity) * get(comoving_energy_density);
  auto& Kt = get(absorption_opacity) + get(scattering_opacity);

  auto& fluid_velocity_i =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>(
          temp_tensors);
  raise_or_lower_index(make_not_null(&fluid_velocity_i), fluid_velocity,
                       spatial_metric);

  get(*source_n) = get(lapse) * get(fluid_lorentz_factor) * eta_minus_kJ +
                   get(lapse) * get(comoving_momentum_density_normal) * Kt;
  for (size_t i = 0; i < spatial_dim; i++) {
    source_i->get(i) =
        get(lapse) * get(fluid_lorentz_factor) * fluid_velocity_i.get(i) *
            eta_minus_kJ -
        get(lapse) * comoving_momentum_density_spatial.get(i) * Kt;
  }
}

}  // namespace detail
}  // namespace M1Grey
}  // namespace RadiationTransport
