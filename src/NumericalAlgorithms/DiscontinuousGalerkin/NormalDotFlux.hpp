// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/OuterProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// @{
/*!
 * \brief Contract a surface normal covector with the first index of a flux
 * tensor or variables
 *
 * \details
 * Returns \f$n_i F^i_{j\ldots}\f$, where the flux tensor \f$F\f$ must have an
 * upper spatial first index and may have arbitrary extra indices.
 */
template <size_t VolumeDim, typename Fr, typename Symm,
          typename... RemainingIndices,
          typename ResultTensor = Tensor<DataVector, tmpl::pop_front<Symm>,
                                         index_list<RemainingIndices...>>>
void normal_dot_flux(
    const gsl::not_null<ResultTensor*> normal_dot_flux,
    const tnsr::i<DataVector, VolumeDim, Fr>& normal,
    const Tensor<DataVector, Symm,
                 index_list<SpatialIndex<VolumeDim, UpLo::Up, Fr>,
                            RemainingIndices...>>& flux_tensor) {
  for (auto it = normal_dot_flux->begin(); it != normal_dot_flux->end(); it++) {
    const auto result_indices = normal_dot_flux->get_tensor_index(it);
    *it = get<0>(normal) * flux_tensor.get(prepend(result_indices, size_t{0}));
    for (size_t d = 1; d < VolumeDim; d++) {
      *it += normal.get(d) * flux_tensor.get(prepend(result_indices, d));
    }
  }
}

template <typename... ReturnTags, typename... FluxTags, size_t VolumeDim,
          typename Fr>
void normal_dot_flux(
    const gsl::not_null<Variables<tmpl::list<ReturnTags...>>*> result,
    const tnsr::i<DataVector, VolumeDim, Fr>& normal,
    const Variables<tmpl::list<FluxTags...>>& fluxes) {
  if (result->number_of_grid_points() != fluxes.number_of_grid_points()) {
    result->initialize(fluxes.number_of_grid_points());
  }
  EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
      make_not_null(&get<ReturnTags>(*result)), normal, get<FluxTags>(fluxes)));
}

template <typename TagsList, size_t VolumeDim, typename Fr>
auto normal_dot_flux(
    const tnsr::i<DataVector, VolumeDim, Fr>& normal,
    const Variables<db::wrap_tags_in<::Tags::Flux, TagsList,
                                     tmpl::size_t<VolumeDim>, Fr>>& fluxes) {
  Variables<db::wrap_tags_in<::Tags::NormalDotFlux, TagsList>> result{};
  normal_dot_flux(make_not_null(&result), normal, fluxes);
  return result;
}
/// @}

/// @{
/*!
 * \brief Multiplies a surface normal covector with a tensor or variables
 *
 * \details
 * Returns the outer product $n_j v_{k\ldots}$, where $n_j$ is the `normal` and
 * $v_{k\ldots}$ is the `rhs`.
 *
 * Note that this quantity is a "normal dot flux" where the flux involves
 * a Kronecker delta. For example:
 *
 * \f{equation}
 * n_j v_{k\ldots} = n_i \delta^i_j v_{k\ldots} = n_i F^i_{jk\ldots}
 * \f}
 *
 * This makes this quantity useful for optimizations of DG formulations for
 * "sparse" (i.e., Kronecker delta) fluxes.
 */
template <size_t VolumeDim, typename Fr, typename Symm, typename Indices>
void normal_times_flux(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, Symm, Indices>, VolumeDim, UpLo::Lo, Fr>*>
        normal_times_flux,
    const tnsr::i<DataVector, VolumeDim, Fr>& normal,
    const Tensor<DataVector, Symm, Indices>& rhs) {
  outer_product(normal_times_flux, normal, rhs);
}

template <typename... ReturnTags, typename... FluxTags, size_t VolumeDim,
          typename Fr>
void normal_times_flux(
    const gsl::not_null<Variables<tmpl::list<ReturnTags...>>*> result,
    const tnsr::i<DataVector, VolumeDim, Fr>& normal,
    const Variables<tmpl::list<FluxTags...>>& fluxes) {
  if (result->number_of_grid_points() != fluxes.number_of_grid_points()) {
    result->initialize(fluxes.number_of_grid_points());
  }
  EXPAND_PACK_LEFT_TO_RIGHT(normal_times_flux(
      make_not_null(&get<ReturnTags>(*result)), normal, get<FluxTags>(fluxes)));
}
/// @}
