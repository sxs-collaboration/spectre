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
#include "Utilities/Requires.hpp"
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
        d_volume_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  ASSERT(d_volume_vars->size() == Dim * volume_vars.size(),
         "The partial derivatives Variables must have size "
             << Dim * volume_vars.size()
             << " (Dim * volume_vars.size()) but has size "
             << d_volume_vars->size() << " and "
             << d_volume_vars->number_of_grid_points() << " grid points.");
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
      d_volume_vars, logical_partial_derivs_ptrs,
      Variables<DerivativeTags>::number_of_independent_components,
      inverse_jacobian);
}

template <size_t CompDim, typename DerivativeTags, typename VariableTags,
          size_t Dim, typename DerivativeFrame>
void cartoon_partial_derivatives_apply(
    const gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        d_volume_vars,
    const Variables<VariableTags>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  static_assert(Dim == 3);
  static_assert(std::is_same_v<
                tmpl::transform<DerivativeTags,
                                tmpl::bind<tmpl::type_from, tmpl::_1>>,
                tmpl::transform<tmpl::front<tmpl::split_at<
                                    VariableTags, tmpl::size<DerivativeTags>>>,
                                tmpl::bind<tmpl::type_from, tmpl::_1>>>);
  ASSERT((CompDim == 2 and
          volume_mesh.quadrature(2) == Spectral::Quadrature::AxialSymmetry) or
             (CompDim == 1 and (volume_mesh.quadrature(1) ==
                                    Spectral::Quadrature::SphericalSymmetry and
                                volume_mesh.quadrature(2) ==
                                    Spectral::Quadrature::SphericalSymmetry)),
         "Invalid Quadrature combinations: axial symmetry requires 2 "
         "non-Cartoon dimensions, spherical symmetry requires 1 non-Cartoon "
         "dimension. Got: "
             << volume_mesh.quadrature());
  DirectionMap<CompDim, gsl::span<const double>> ghost_cell_vars_slice{};
  for (const auto& direction : Direction<CompDim>::all_directions()) {
    ghost_cell_vars_slice[direction] = ghost_cell_vars.at(
        Direction<3>{direction.dimension(), direction.side()});
  }
  using first_var_tag = tmpl::front<VariableTags>;
  constexpr size_t number_of_var_components =
      Variables<DerivativeTags>::number_of_independent_components;
  const auto volume_vars_span = gsl::make_span(
      get<first_var_tag>(volume_vars)[0].data(),
      number_of_var_components * volume_vars.number_of_grid_points());
  ASSERT(d_volume_vars->size() == Dim * volume_vars_span.size(),
         "The partial derivatives Variables must have size "
             << Dim * volume_vars_span.size()
             << " (Dim * volume_vars_span.size()) but has size "
             << d_volume_vars->size() << " and "
             << d_volume_vars->number_of_grid_points() << " grid points.");
  const size_t logical_derivs_internal_buffer_size =
      CompDim == 1
          ? static_cast<size_t>(0)
          : (volume_vars_span.size() +
             2 * alg::max_element(ghost_cell_vars_slice,
                                  [](const auto& a, const auto& b) {
                                    return a.second.size() < b.second.size();
                                  })
                     ->second.size() +
             volume_vars_span.size());
  DataVector buffer(CompDim * volume_vars_span.size() +
                    logical_derivs_internal_buffer_size);
  std::array<gsl::span<double>, CompDim> logical_derivs{};
  for (size_t i = 0; i < CompDim; ++i) {
    gsl::at(logical_derivs, i) = gsl::make_span(
        &buffer[i * volume_vars_span.size()], volume_vars_span.size());
  }
  if constexpr (CompDim == 1) {
    // No buffer in 1d
    logical_partial_derivatives(
        make_not_null(&logical_derivs), volume_vars_span, ghost_cell_vars_slice,
        volume_mesh.slice_through(0), number_of_variables, fd_order);
  } else {
    gsl::span<double> span_buffer =
        gsl::make_span(&buffer[CompDim * volume_vars_span.size()],
                       logical_derivs_internal_buffer_size);
    detail::logical_partial_derivatives_impl(
        make_not_null(&logical_derivs), &span_buffer, volume_vars_span,
        ghost_cell_vars_slice, volume_mesh.slice_through(0, 1),
        number_of_variables, fd_order);
  }

  std::array<const double*, CompDim> logical_derivs_ptrs{};
  for (size_t i = 0; i < CompDim; ++i) {
    gsl::at(logical_derivs_ptrs, i) = gsl::at(logical_derivs, i).data();
  }
  partial_derivatives_with_cartoon_impl(d_volume_vars, volume_vars,
                                        logical_derivs_ptrs, volume_mesh,
                                        inverse_jacobian, inertial_coords);
}

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame, Requires<Dim == 3>>
void cartoon_partial_derivatives(
    const gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        d_volume_vars,
    const Variables<VariableTags>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  if (volume_mesh.basis(0) != Spectral::Basis::Cartoon and
      volume_mesh.basis(2) == Spectral::Basis::Cartoon) {
    // computational dimension needs to be a constexpr
    if (volume_mesh.basis(1) == Spectral::Basis::Cartoon) {
      cartoon_partial_derivatives_apply<1, DerivativeTags>(
          d_volume_vars, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, fd_order, inverse_jacobian, inertial_coords);
    } else {
      cartoon_partial_derivatives_apply<2, DerivativeTags>(
          d_volume_vars, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, fd_order, inverse_jacobian, inertial_coords);
    }
  } else {
    ERROR("Bases do not match valid Cartoon pattern, got "
          << volume_mesh.basis());
  }
}

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
void partial_derivatives(
    const gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        d_volume_vars,
    const Variables<VariableTags>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  if constexpr (Dim == 3) {
    if (volume_mesh.basis(2) == Spectral::Basis::Cartoon) {
      cartoon_partial_derivatives<DerivativeTags>(
          d_volume_vars, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, fd_order, inverse_jacobian, inertial_coords);
      return;
    }
  }
  (void)inertial_coords;
  ASSERT(
      Dim != 3 or volume_mesh.basis(Dim - 1) != Spectral::Basis::Cartoon,
      "Cartoon basis is only allowed for Dim = 3, got Cartoon basis with Dim = "
          << Dim);
  const auto volume_vars_span = gsl::make_span(
      get<tmpl::front<DerivativeTags>>(volume_vars)[0].data(),
      Variables<DerivativeTags>::number_of_independent_components *
          volume_vars.number_of_grid_points());
  partial_derivatives<DerivativeTags, Dim, DerivativeFrame>(
      d_volume_vars, volume_vars_span, ghost_cell_vars, volume_mesh,
      number_of_variables, fd_order, inverse_jacobian);
}
}  // namespace fd
