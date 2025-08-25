// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/StdArrayHelpers.hpp"

template <typename FluxTags, size_t Dim, typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::div, FluxTags>> divergence(
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  Variables<db::wrap_tags_in<Tags::div, FluxTags>> divergence_of_F(
      F.number_of_grid_points());
  divergence(make_not_null(&divergence_of_F), F, mesh, inverse_jacobian);
  return divergence_of_F;
}

template <typename... DivTags, typename... FluxTags, size_t Dim,
          typename DerivativeFrame>
void divergence(
    const gsl::not_null<Variables<tmpl::list<DivTags...>>*> divergence_of_F,
    const Variables<tmpl::list<FluxTags...>>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  if (UNLIKELY(divergence_of_F->number_of_grid_points() !=
               mesh.number_of_grid_points())) {
    divergence_of_F->initialize(mesh.number_of_grid_points());
  }

  using DerivativeTags = tmpl::list<FluxTags...>;
  using ValueType = typename Variables<DerivativeTags>::value_type;
  const size_t vars_size =
      Variables<DerivativeTags>::number_of_independent_components *
      F.number_of_grid_points();
  const auto logical_derivs_data =
      cpp20::make_unique_for_overwrite<ValueType[]>(
          (Dim > 1 ? (Dim + 2) : Dim) * vars_size);
  std::array<ValueType*, Dim> logical_derivs{};
  std::array<Variables<DerivativeTags>, Dim> logical_partial_derivatives_of_F{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(logical_derivs, i) = &(logical_derivs_data[i * vars_size]);
    gsl::at(logical_partial_derivatives_of_F, i)
        .set_data_ref(gsl::at(logical_derivs, i), vars_size);
  }
  if constexpr (Dim > 1) {
    Variables<DerivativeTags> temp0{&logical_derivs_data[Dim * vars_size],
                                    vars_size};
    Variables<DerivativeTags> temp1{&logical_derivs_data[(Dim + 1) * vars_size],
                                    vars_size};
    partial_derivatives_detail::
        LogicalImpl<Dim, tmpl::list<FluxTags...>, DerivativeTags>::apply(
            make_not_null(&logical_derivs), &temp0, &temp1, F, mesh);
  } else {
    Variables<DerivativeTags>* temp = nullptr;
    partial_derivatives_detail::
        LogicalImpl<Dim, tmpl::list<FluxTags...>, DerivativeTags>::apply(
            make_not_null(&logical_derivs), temp, temp, F, mesh);
  }

  const auto apply_div = [&divergence_of_F, &inverse_jacobian,
                          &logical_partial_derivatives_of_F](auto flux_tag_v,
                                                             auto div_tag_v) {
    using FluxTag = std::decay_t<decltype(flux_tag_v)>;
    using DivFluxTag = std::decay_t<decltype(div_tag_v)>;

    using first_index = tmpl::front<typename FluxTag::type::index_list>;
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
  };
  EXPAND_PACK_LEFT_TO_RIGHT(apply_div(FluxTags{}, DivTags{}));
}

template <size_t CompDim, typename DivTags, typename FluxTags, size_t Dim,
          typename DerivativeFrame>
void cartoon_divergence_apply(
    const gsl::not_null<Variables<DivTags>*> divergence_of_F,
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  static_assert(Dim == 3);
  ASSERT((CompDim == 2 and
          mesh.quadrature(2) == Spectral::Quadrature::AxialSymmetry) or
             (CompDim == 1 and
              (mesh.quadrature(1) == Spectral::Quadrature::SphericalSymmetry and
               mesh.quadrature(2) == Spectral::Quadrature::SphericalSymmetry)),
         "Invalid Quadrature combinations: axial symmetry requires 2 "
         "non-Cartoon dimensions, spherical symmetry requires 1 non-Cartoon "
         "dimension. Got: "
             << mesh.quadrature());

  if (UNLIKELY(divergence_of_F->number_of_grid_points() !=
               mesh.number_of_grid_points())) {
    divergence_of_F->initialize(mesh.number_of_grid_points());
  }

  using DerivativeTags = FluxTags;
  using ValueType = typename Variables<DerivativeTags>::value_type;
  const size_t vars_size =
      Variables<DerivativeTags>::number_of_independent_components *
      F.number_of_grid_points();
  const auto logical_derivs_data =
      cpp20::make_unique_for_overwrite<ValueType[]>(
          (CompDim > 1 ? (CompDim + 2) : CompDim) * vars_size);
  std::array<ValueType*, CompDim> logical_derivs{};
  std::array<Variables<DerivativeTags>, CompDim>
      logical_partial_derivatives_of_F{};
  for (size_t i = 0; i < CompDim; ++i) {
    gsl::at(logical_derivs, i) = &(logical_derivs_data[i * vars_size]);
    gsl::at(logical_partial_derivatives_of_F, i)
        .set_data_ref(gsl::at(logical_derivs, i), vars_size);
  }

  if constexpr (CompDim == 1) {
    Variables<DerivativeTags>* temp = nullptr;
    partial_derivatives_detail::LogicalImpl<CompDim, FluxTags, DerivativeTags>::
        apply(make_not_null(&logical_derivs), temp, temp, F,
              mesh.slice_through(0));
  } else {
    Variables<DerivativeTags> temp0{&logical_derivs_data[CompDim * vars_size],
                                    vars_size};
    Variables<DerivativeTags> temp1{
        &logical_derivs_data[(CompDim + 1) * vars_size], vars_size};
    partial_derivatives_detail::LogicalImpl<CompDim, FluxTags, DerivativeTags>::
        apply(make_not_null(&logical_derivs), &temp0, &temp1, F,
              mesh.slice_through(0, 1));
  }

  const Spectral::Quadrature quad_type = mesh.quadrature(2);
  const auto numerical_deriv_in_this_direction = [](const size_t deriv_num) {
    if constexpr (CompDim == 2) {
      return deriv_num == 0 or deriv_num == 1;
    } else {
      return deriv_num == 0;
    }
  };

  DataVector safe_x_coords = get<0>(inertial_coords);
  const bool element_contains_zero_of_symmetry_axis = equal_within_roundoff(
      0.0, safe_x_coords[0], std::numeric_limits<double>::epsilon() * 100.0,
      max(safe_x_coords));
  // Having x=0 requires both not dividing by zero (done here by setting to
  // arbitrary non-zero value) and remembering to then do L'Hopital
  if (element_contains_zero_of_symmetry_axis) {
    safe_x_coords[0] = 1.0;
    if constexpr (CompDim == 2) {
      const size_t x_extents = mesh.extents(0);
      const size_t y_extents = mesh.extents(1);
      for (size_t i = 1; i < y_extents; ++i) {
        ASSERT(
            equal_within_roundoff(
                0.0, safe_x_coords[i * x_extents],
                std::numeric_limits<double>::epsilon() * 100.0,
                max(safe_x_coords)),
            "The passed inertial coordinates do not follow the required format "
            "of a rectangular mesh with x=0 being located at the first index "
            "of each constant-y subdomain of the x DataVector. x="
                << safe_x_coords);
        safe_x_coords[i * x_extents] = 1.0;
      }
    }
  }
  // If doing L'Hopital's rule, we need to store temporary forms of the data
  // in two different tensors (hence the 0 & 1 labels) for each tensor type
  // They only have to hold the x=0 positions, so of size y_extents
  using FluxTypes = tmpl::remove_duplicates<
      tmpl::transform<FluxTags, tmpl::bind<tmpl::type_from, tmpl::_1>>>;

  using TempTags0 = ::Tags::convert_to_temp_tensors<FluxTypes, 0>;
  using TempTags1 = ::Tags::convert_to_temp_tensors<FluxTypes, 1>;
  using VarTags = tmpl::append<TempTags0, TempTags1>;

  Variables<VarTags> temp_vars{};
  if (element_contains_zero_of_symmetry_axis) {
    const size_t y_extents = mesh.extents(1);
    temp_vars.initialize(y_extents);
  }

  tmpl::for_each<
      FluxTags>([&divergence_of_F, &F, &inverse_jacobian_3d, &mesh,
                 &safe_x_coords, &logical_partial_derivatives_of_F,
                 &numerical_deriv_in_this_direction, &quad_type, &temp_vars,
                 &element_contains_zero_of_symmetry_axis]<typename FluxTag>(
                    tmpl::type_<FluxTag> /*meta*/) {
    using DivFluxTag = tmpl::at<DivTags, tmpl::index_of<FluxTags, FluxTag>>;

    using first_index = tmpl::front<typename FluxTag::type::index_list>;
    static_assert(
        std::is_same_v<typename first_index::Frame, DerivativeFrame> and
            first_index::ul == UpLo::Up,
        "First index of tensor cannot be contracted with derivative "
        "because either it is in the wrong frame or it has the wrong "
        "valence");
    static_assert(first_index::index_type == IndexType::Spatial);

    auto& divergence_of_flux = get<DivFluxTag>(*divergence_of_F);
    auto& flux = get<FluxTag>(F);
    TensorMetafunctions::prepend_spatial_index<typename FluxTag::type, Dim,
                                               UpLo::Lo, Frame::Inertial>
        cart_deriv_tensor;
    cartoon_derivative<typename FluxTag::type, 3, Frame::Inertial>(
        cart_deriv_tensor, flux, safe_x_coords, quad_type);

    if (not element_contains_zero_of_symmetry_axis) {
      for (auto it = divergence_of_flux.begin(); it != divergence_of_flux.end();
           ++it) {
        *it = 0.0;
        const auto div_flux_indices = divergence_of_flux.get_tensor_index(it);
        for (size_t i0 = 0; i0 < Dim; ++i0) {
          const auto flux_indices = prepend(div_flux_indices, i0);
          if (numerical_deriv_in_this_direction(i0)) {
            // only accessing relevenat entries of inverse_jacobian_3d
            for (size_t d = 0; d < CompDim; ++d) {
              *it += inverse_jacobian_3d.get(d, i0) *
                     get<FluxTag>(gsl::at(logical_partial_derivatives_of_F, d))
                         .get(flux_indices);
            }
          } else {
            const auto d_flux_indices = prepend(flux_indices, i0);
            *it += cart_deriv_tensor.get(d_flux_indices);
          }
        }
      }
    } else {
      const size_t x_extents = mesh.extents(0);
      const size_t y_extents = mesh.extents(1);
      const auto insert_with_stride =
          [x_extents, y_extents](gsl::not_null<DataVector*> to_insert_into,
                                 const DataVector& take_from,
                                 const bool accumulate = false) {
            auto& insert_into = *to_insert_into;
            if (accumulate) {
              for (size_t i = 0; i < y_extents; ++i) {
                insert_into[i] += take_from[i * x_extents];
              }
            } else {
              for (size_t i = 0; i < y_extents; ++i) {
                insert_into[i] = take_from[i * x_extents];
              }
            }
          };
      // for storing intermediate x-derivatives
      auto& dx_flux =
          get<::Tags::TempTensor<0, typename FluxTag::type>>(temp_vars);
      auto& contracted_dx_flux =
          get<::Tags::TempTensor<1, typename FluxTag::type>>(temp_vars);

      for (auto it = divergence_of_flux.begin(); it != divergence_of_flux.end();
           ++it) {
        *it = 0.0;
        const auto div_flux_indices = divergence_of_flux.get_tensor_index(it);
        for (size_t first_flux_index = 0; first_flux_index < index_dim<0>(flux);
             ++first_flux_index) {
          auto flux_indices = prepend(div_flux_indices, first_flux_index);
          for (size_t d_i = 0; d_i < Dim; ++d_i) {
            if (numerical_deriv_in_this_direction(d_i)) {
              if (d_i == first_flux_index) {
                const auto d_indices = prepend(div_flux_indices, d_i);
                for (size_t d_contract = 0; d_contract < CompDim;
                     ++d_contract) {
                  *it += inverse_jacobian_3d.get(d_contract, d_i) *
                         get<FluxTag>(gsl::at(logical_partial_derivatives_of_F,
                                              d_contract))
                             .get(d_indices);
                }
              }
              if (d_i == 0) {
                // storing all \partial_x derivatives for L'Hopital
                if (first_flux_index == d_i) {
                  insert_with_stride(make_not_null(&dx_flux.get(flux_indices)),
                                     *it);
                } else {
                  insert_with_stride(
                      make_not_null(&dx_flux.get(flux_indices)),
                      inverse_jacobian_3d.get(0, d_i) *
                          get<FluxTag>(
                              gsl::at(logical_partial_derivatives_of_F, 0))
                              .get(flux_indices));
                  if constexpr (CompDim == 2) {
                    insert_with_stride(
                        make_not_null(&dx_flux.get(flux_indices)),
                        inverse_jacobian_3d.get(1, d_i) *
                            get<FluxTag>(
                                gsl::at(logical_partial_derivatives_of_F, 1))
                                .get(flux_indices),
                        true);
                  }
                }
              }
            } else if (first_flux_index == d_i) {
              const auto d_flux_indices = prepend(flux_indices, d_i);
              *it += cart_deriv_tensor.get(d_flux_indices);
            }
          }
        }
      }
      const size_t start_index =
          quad_type == Spectral::Quadrature::SphericalSymmetry ? 1 : 2;
      for (size_t deriv_index = start_index; deriv_index < Dim; ++deriv_index) {
        cartoon_contraction(make_not_null(&contracted_dx_flux), dx_flux,
                            deriv_index, quad_type);
        for (size_t component_index = 0;
             component_index < contracted_dx_flux.size(); ++component_index) {
          const auto input_index =
              contracted_dx_flux.get_tensor_index(component_index);
          if (gsl::at(input_index, 0) == deriv_index) {
            std::array<size_t, input_index.size() - 1> output_indices{};
            std::copy(input_index.begin() + 1, input_index.end(),
                      output_indices.begin());
            const size_t output_index =
                divergence_of_flux.get_storage_index(output_indices);
            for (size_t i = 0; i < y_extents; ++i) {
              divergence_of_flux[output_index][i * x_extents] +=
                  contracted_dx_flux[component_index][i];
            }
          }
        }
      }
    }
  });
}

template <typename... DivTags, typename... FluxTags, size_t Dim,
          typename DerivativeFrame, Requires<Dim == 3>>
void cartoon_divergence(
    const gsl::not_null<Variables<tmpl::list<DivTags...>>*> divergence_of_F,
    const Variables<tmpl::list<FluxTags...>>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  if (mesh.basis(0) != Spectral::Basis::Cartoon and
      mesh.basis(2) == Spectral::Basis::Cartoon) {
    if (mesh.basis(1) == Spectral::Basis::Cartoon) {
      cartoon_divergence_apply<1, tmpl::list<DivTags...>,
                               tmpl::list<FluxTags...>, Dim, DerivativeFrame>(
          divergence_of_F, F, mesh, inverse_jacobian_3d, inertial_coords);
    } else {
      cartoon_divergence_apply<2, tmpl::list<DivTags...>,
                               tmpl::list<FluxTags...>, Dim, DerivativeFrame>(
          divergence_of_F, F, mesh, inverse_jacobian_3d, inertial_coords);
    }
  } else {
    ERROR("Bases do not match valid Cartoon pattern. Got mesh " << mesh);
  }
}

template <typename FluxTags, size_t Dim>
Variables<db::wrap_tags_in<Tags::div, FluxTags>> logical_divergence(
    const Variables<FluxTags>& flux, const Mesh<Dim>& mesh) {
  Variables<db::wrap_tags_in<Tags::div, FluxTags>> div_flux(
      flux.number_of_grid_points());
  logical_divergence(make_not_null(&div_flux), flux, mesh);
  return div_flux;
}

template <typename... ResultTags, typename... FluxTags, size_t Dim>
void logical_divergence(
    const gsl::not_null<Variables<tmpl::list<ResultTags...>>*> div_flux,
    const Variables<tmpl::list<FluxTags...>>& flux, const Mesh<Dim>& mesh) {
  static_assert(
      (... and
       std::is_same_v<tmpl::front<typename FluxTags::type::index_list>,
                      SpatialIndex<Dim, UpLo::Up, Frame::ElementLogical>>),
      "The first index of each flux must be an upper spatial index in "
      "element-logical coordinates.");
  static_assert(
      (... and
       std::is_same_v<
           typename ResultTags::type,
           TensorMetafunctions::remove_first_index<typename FluxTags::type>>),
      "The result tensors must have the same type as the flux tensors with "
      "their first index removed.");
  div_flux->initialize(mesh.number_of_grid_points(), 0.);
  // Note: This function hasn't been optimized much at all. Feel free to
  // optimize if needed!
  EXPAND_PACK_LEFT_TO_RIGHT(logical_divergence(
      make_not_null(&get<ResultTags>(*div_flux)), get<FluxTags>(flux), mesh));
}
