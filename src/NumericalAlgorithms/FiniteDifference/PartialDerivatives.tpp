// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace fd {
namespace detail {
template <size_t Dim>
void logical_partial_derivatives_impl(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        logical_derivatives,
    gsl::span<double>* buffer, const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order);
}  // namespace detail

template <typename DerivativeTags, size_t Dim, typename DerivativeFrame>
void partial_derivatives(
    const gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        partial_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  ASSERT(partial_derivatives->size() == Dim * volume_vars.size(),
         "The partial derivatives Variables must have size "
             << Dim * volume_vars.size()
             << " (Dim * volume_vars.size()) but has size "
             << partial_derivatives->size() << " and "
             << partial_derivatives->number_of_grid_points()
             << " grid points.");
  const size_t logical_derivs_internal_buffer_size =
      Dim == 1
          ? static_cast<size_t>(0)
          : (volume_vars.size() +
             2 * alg::max_element(ghost_cell_vars,
                                  [](const auto& a, const auto& b) {
                                    return a.second.size() < b.second.size();
                                  })
                     ->second.size() +
             volume_vars.size());
  DataVector buffer(Dim * volume_vars.size() +
                    logical_derivs_internal_buffer_size);
  std::array<gsl::span<double>, Dim> logical_partial_derivs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(logical_partial_derivs, i) =
        gsl::make_span(&buffer[i * volume_vars.size()], volume_vars.size());
  }
  if constexpr (Dim > 1) {
    gsl::span<double> span_buffer = gsl::make_span(
        &buffer[Dim * volume_vars.size()], logical_derivs_internal_buffer_size);
    detail::logical_partial_derivatives_impl(
        make_not_null(&logical_partial_derivs), &span_buffer, volume_vars,
        ghost_cell_vars, volume_mesh, number_of_variables, fd_order);
  } else {
    // No buffer in 1d
    logical_partial_derivatives(make_not_null(&logical_partial_derivs),
                                volume_vars, ghost_cell_vars, volume_mesh,
                                number_of_variables, fd_order);
  }

  std::array<const double*, Dim> logical_partial_derivs_ptrs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(logical_partial_derivs_ptrs, i) =
        gsl::at(logical_partial_derivs, i).data();
  }
  ::partial_derivatives_detail::partial_derivatives_impl(
      partial_derivatives, logical_partial_derivs_ptrs,
      Variables<DerivativeTags>::number_of_independent_components,
      inverse_jacobian);
}

}  // namespace fd
