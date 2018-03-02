// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/StdArrayHelpers.hpp"

template <typename FluxTags, size_t Dim, typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::div, FluxTags, DerivativeFrame>> divergence(
    const Variables<FluxTags>& F, const Index<Dim>& extents,
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                 tmpl::list<SpatialIndex<Dim, UpLo::Up, Frame::Logical>,
                          SpatialIndex<Dim, UpLo::Lo, DerivativeFrame>>>&
        inverse_jacobian) noexcept {
  const auto logical_partial_derivatives_of_F =
      logical_partial_derivatives<FluxTags>(F, extents);

  Variables<db::wrap_tags_in<Tags::div, FluxTags, DerivativeFrame>>
      divergence_of_F(F.number_of_grid_points(), 0.0);

  tmpl::for_each<FluxTags>([
    &divergence_of_F, &inverse_jacobian, &logical_partial_derivatives_of_F
  ](auto tag) noexcept {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    using DivFluxTag = Tags::div<FluxTag, DerivativeFrame>;
    auto& divergence_of_flux = get<DivFluxTag>(divergence_of_F);
    for (auto it = divergence_of_flux.begin(); it != divergence_of_flux.end();
         ++it) {
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

  return divergence_of_F;
}
