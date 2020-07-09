// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/StdArrayHelpers.hpp"

template <typename FluxTags, size_t Dim, typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::div, FluxTags>> divergence(
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept {
  Variables<db::wrap_tags_in<Tags::div, FluxTags>> divergence_of_F(
      F.number_of_grid_points());
  divergence(make_not_null(&divergence_of_F), F, mesh, inverse_jacobian);
  return divergence_of_F;
}

template <typename FluxTags, size_t Dim, typename DerivativeFrame>
void divergence(
    const gsl::not_null<Variables<db::wrap_tags_in<Tags::div, FluxTags>>*>
        divergence_of_F,
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept {
  if (UNLIKELY(divergence_of_F->number_of_grid_points() !=
               mesh.number_of_grid_points())) {
    divergence_of_F->initialize(mesh.number_of_grid_points());
  }

  const auto logical_partial_derivatives_of_F =
      logical_partial_derivatives<FluxTags>(F, mesh);

  tmpl::for_each<FluxTags>([
    &divergence_of_F, &inverse_jacobian, &logical_partial_derivatives_of_F
  ](auto tag) noexcept {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    using DivFluxTag = Tags::div<FluxTag>;

    using first_index =
        tmpl::front<typename FluxTag::type::index_list>;
    static_assert(
        std::is_same_v<typename first_index::Frame, DerivativeFrame> and
            first_index::ul == UpLo::Up,
        "First index of tensor cannot be contracted with derivative "
        "because either it is in the wrong frame or it has the wrong "
        "valence");

    auto& divergence_of_flux = get<DivFluxTag>(*divergence_of_F);
    for (auto it = divergence_of_flux.begin(); it != divergence_of_flux.end();
         ++it) {
      *it = 0.0;
      const auto div_flux_indices = divergence_of_flux.get_tensor_index(it);
      for (size_t i0 = 0; i0 < Dim; ++i0) {
        const auto flux_indices = prepend(div_flux_indices, i0);
        for (size_t d = 0; d < Dim; ++d) {
          *it += inverse_jacobian.get(d, i0) *
                 get<FluxTag>(gsl::at(logical_partial_derivatives_of_F, d))
                     .get(flux_indices);
        }
      }
    }
  });
}
