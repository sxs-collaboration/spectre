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

// helper structs to get TempTensor types for cartoon_divergence_appply()
template <typename TempTensorType>
struct make_temptensor_0 {
  using type = ::Tags::TempTensor<0, TempTensorType>;
};

template <typename TempTensorType>
struct set_label_1;

template <typename TensorType, size_t OldLabel>
struct set_label_1<::Tags::TempTensor<OldLabel, TensorType>> {
  using type = ::Tags::TempTensor<1, TensorType>;
};

template <typename TempTensorType>
struct get_tensor_type;

template <typename TensorType, size_t Label>
struct get_tensor_type<::Tags::TempTensor<Label, TensorType>> {
  using type = TensorType;
};

template <typename List, typename TempTensor>
struct add_unique {
  using tensor_type = typename get_tensor_type<TempTensor>::type;
  using type =
      tmpl::conditional_t<tmpl::list_contains<List, tensor_type>::value, List,
                          tmpl::push_back<List, tensor_type>>;
};

template <size_t Comp_dim, bool need_lhopital, typename DivTags,
          typename FluxTags, size_t Dim, typename DerivativeFrame>
void cartoon_divergence_apply(
    const gsl::not_null<Variables<DivTags>*> divergence_of_F,
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  static_assert(Dim == 3);
  ASSERT((Comp_dim == 2 and
          mesh.quadrature(2) == Spectral::Quadrature::AxialSymmetry) or
             (Comp_dim == 1 and
              (mesh.quadrature(1) == Spectral::Quadrature::SphericalSymmetry and
               mesh.quadrature(2) == Spectral::Quadrature::SphericalSymmetry)),
         "Axial symmetry has only one Cartoon dimension, Spherical symmetry "
         "has two");

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
          (Comp_dim > 1 ? (Comp_dim + 2) : Comp_dim) * vars_size);
  std::array<ValueType*, Comp_dim> logical_derivs{};
  std::array<Variables<DerivativeTags>, Comp_dim>
      logical_partial_derivatives_of_F{};
  for (size_t i = 0; i < Comp_dim; ++i) {
    gsl::at(logical_derivs, i) = &(logical_derivs_data[i * vars_size]);
    gsl::at(logical_partial_derivatives_of_F, i)
        .set_data_ref(gsl::at(logical_derivs, i), vars_size);
  }
  if constexpr (Comp_dim == 2) {
    Variables<DerivativeTags> temp0{&logical_derivs_data[Comp_dim * vars_size],
                                    vars_size};
    Variables<DerivativeTags> temp1{
        &logical_derivs_data[(Comp_dim + 1) * vars_size], vars_size};
    partial_derivatives_detail::
        LogicalImpl<Comp_dim, FluxTags, DerivativeTags>::apply(
            make_not_null(&logical_derivs), &temp0, &temp1, F,
            mesh.slice_through(0, 1));
  } else {
    Variables<DerivativeTags>* temp = nullptr;
    partial_derivatives_detail::
        LogicalImpl<Comp_dim, FluxTags, DerivativeTags>::apply(
            make_not_null(&logical_derivs), temp, temp, F,
            mesh.slice_through(0));
  }

  const Spectral::Quadrature quad_type = mesh.quadrature(2);
  const auto real_deriv_check = [](const size_t deriv_num) {
    if constexpr (Comp_dim == 2) {
      return deriv_num == 0 or deriv_num == 1;
    } else {
      return deriv_num == 0;
    }
  };

  const size_t x_extents = mesh.extents(0);
  const size_t y_extents = mesh.extents(1);
  // If doing L'Hopital's rule, we need to store temporary forms of the data
  // in two different tensors (hence the 0 & 1 labels) for each flux type
  // They only have to hold the x=0 positions, so of size y_extents
  using TempTags0 =
      tmpl::transform<tmpl::fold<FluxTags, tmpl::list<>,
                                 add_unique<tmpl::_state, tmpl::_element>>,
                      make_temptensor_0<tmpl::_1>>;
  using TempTags1 = tmpl::transform<TempTags0, set_label_1<tmpl::_1>>;
  // "default" is scalar, is this okay? not sure how to totally avoid
  // creating Variables
  using VarTags =
      tmpl::conditional_t<need_lhopital, tmpl::append<TempTags0, TempTags1>,
                          tmpl::list<::Tags::TempScalar<0>>>;

  Variables<VarTags> temp_vars{y_extents};

  DataVector safe_x_coords = get<0>(inertial_coords);
  // Having x=0 requires both not dividing by zero (done here by setting to
  // arbitrary non-zero value) and remembering to then do L'Hopital
  if constexpr (need_lhopital) {
    safe_x_coords[0] = 1.0;
    if constexpr (Comp_dim == 2) {
      for (size_t i = 1; i < y_extents; ++i) {
        safe_x_coords[i * x_extents] = 1.0;
      }
    }
  }

  tmpl::for_each<FluxTags>([&divergence_of_F, &F, &inverse_jacobian_3d,
                            &safe_x_coords, &logical_partial_derivatives_of_F,
                            &real_deriv_check, &quad_type, &temp_vars,
                            x_extents, y_extents]<typename FluxTag>(
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

    if constexpr (not need_lhopital) {
      for (auto it = divergence_of_flux.begin(); it != divergence_of_flux.end();
           ++it) {
        *it = 0.0;
        const auto div_flux_indices = divergence_of_flux.get_tensor_index(it);
        for (size_t i0 = 0; i0 < Dim; ++i0) {
          const auto flux_indices = prepend(div_flux_indices, i0);
          if (real_deriv_check(i0)) {
            // only accessing relevenat entries of inverse_jacobian_3d
            for (size_t d = 0; d < Comp_dim; ++d) {
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
      (void)x_extents;
      (void)y_extents;
      (void)temp_vars;
    } else {
      const auto insert_with_stride = [x_extents, y_extents](
                                          DataVector& insert_into,
                                          const DataVector& take_from) {
        for (size_t i = 0; i < y_extents; ++i) {
          insert_into[i] = take_from[i * x_extents];
        }
      };
      const auto accumulate_with_stride = [x_extents, y_extents](
                                              DataVector& insert_into,
                                              const DataVector& take_from) {
        for (size_t i = 0; i < y_extents; ++i) {
          insert_into[i] += take_from[i * x_extents];
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
        for (size_t missing = 0; missing < index_dim<0>(flux); ++missing) {
          auto flux_indices = prepend(div_flux_indices, missing);
          for (size_t i0 = 0; i0 < Dim; ++i0) {
            if (real_deriv_check(i0)) {
              const auto d_indices = prepend(div_flux_indices, i0);
              if (missing == i0) {
                for (size_t d = 0; d < Comp_dim; ++d) {
                  *it +=
                      inverse_jacobian_3d.get(d, i0) *
                      get<FluxTag>(gsl::at(logical_partial_derivatives_of_F, d))
                          .get(d_indices);
                }
              }
              if (i0 == 0) {
                if (missing == i0) {
                  insert_with_stride(dx_flux.get(flux_indices), *it);
                } else {
                  insert_with_stride(
                      dx_flux.get(flux_indices),
                      inverse_jacobian_3d.get(0, i0) *
                          get<FluxTag>(
                              gsl::at(logical_partial_derivatives_of_F, 0))
                              .get(flux_indices));
                  for (size_t d = 1; d < Comp_dim; ++d) {
                    accumulate_with_stride(
                        dx_flux.get(flux_indices),
                        inverse_jacobian_3d.get(d, i0) *
                            get<FluxTag>(
                                gsl::at(logical_partial_derivatives_of_F, d))
                                .get(flux_indices));
                  }
                }
              }
            } else if (missing == i0) {
              const auto d_flux_indices = prepend(flux_indices, i0);
              *it += cart_deriv_tensor.get(d_flux_indices);
            }
          }
        }
      }
      const size_t start_index =
          quad_type == Spectral::Quadrature::SphericalSymmetry ? 1 : 2;
      for (size_t deriv_index = start_index; deriv_index < Dim; ++deriv_index) {
        cartoon_contraction(contracted_dx_flux, dx_flux, deriv_index,
                            quad_type);
        for (size_t component_index = 0;
             component_index < contracted_dx_flux.size(); ++component_index) {
          const auto input_index =
              contracted_dx_flux.get_tensor_index(component_index);
          if (gsl::at(input_index, 0) == deriv_index) {
            std::array<size_t, input_index.size() - 1> output_index;
            std::copy(input_index.begin() + 1, input_index.end(),
                      output_index.begin());
            for (size_t i = 0; i < y_extents; ++i) {
              divergence_of_flux.get(output_index)[i * x_extents] +=
                  contracted_dx_flux.get(input_index)[i];
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
    const bool needs_lhopital =
        equal_within_roundoff(0.0, get<0>(inertial_coords)[0],
                              std::numeric_limits<double>::epsilon() * 100.0,
                              max(get<0>(inertial_coords)));

    if (mesh.basis(1) == Spectral::Basis::Cartoon) {
      if (needs_lhopital) {
        cartoon_divergence_apply<1, true, tmpl::list<DivTags...>,
                                 tmpl::list<FluxTags...>, Dim, DerivativeFrame>(
            divergence_of_F, F, mesh, inverse_jacobian_3d, inertial_coords);
      } else {
        cartoon_divergence_apply<1, false, tmpl::list<DivTags...>,
                                 tmpl::list<FluxTags...>, Dim, DerivativeFrame>(
            divergence_of_F, F, mesh, inverse_jacobian_3d, inertial_coords);
      }
    } else {
      if (needs_lhopital) {
        cartoon_divergence_apply<2, true, tmpl::list<DivTags...>,
                                 tmpl::list<FluxTags...>, Dim, DerivativeFrame>(
            divergence_of_F, F, mesh, inverse_jacobian_3d, inertial_coords);
      } else {
        cartoon_divergence_apply<2, false, tmpl::list<DivTags...>,
                                 tmpl::list<FluxTags...>, Dim, DerivativeFrame>(
            divergence_of_F, F, mesh, inverse_jacobian_3d, inertial_coords);
      }
    }
  } else {
    ERROR("Bases do not match required Cartoon pattern.");
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
