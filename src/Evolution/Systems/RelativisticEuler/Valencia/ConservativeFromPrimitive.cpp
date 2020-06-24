// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace RelativisticEuler::Valencia {

template <size_t Dim>
void ConservativeFromPrimitive<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> tilde_s,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric) noexcept {
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, Dim>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>>>
      temp_tensors{get(rest_mass_density).size()};
  auto& spatial_velocity_oneform =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, Dim>>(temp_tensors);
  raise_or_lower_index(make_not_null(&spatial_velocity_oneform),
                       spatial_velocity, spatial_metric);
  auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity_oneform);

  get(*tilde_d) = get(sqrt_det_spatial_metric) * get(rest_mass_density) *
                  get(lorentz_factor);

  get(*tilde_tau) = get(sqrt_det_spatial_metric) * square(get(lorentz_factor)) *
                    (get(rest_mass_density) *
                         (get(specific_internal_energy) +
                          get(spatial_velocity_squared) * get(lorentz_factor) /
                              (get(lorentz_factor) + 1.)) +
                     get(pressure) * get(spatial_velocity_squared));

  for (size_t i = 0; i < Dim; ++i) {
    tilde_s->get(i) = get(*tilde_d) * get(lorentz_factor) *
                      get(specific_enthalpy) * spatial_velocity_oneform.get(i);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template class ConservativeFromPrimitive<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace RelativisticEuler::Valencia
/// \endcond
