// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Transpose.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/ParityFromSymmetry.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace partial_derivatives_detail {
template <size_t Dim, typename VariableTags, typename DerivativeTags>
struct LogicalImpl;

// This routine has been optimized to perform really well. The following
// describes what optimizations were made.
//
// - The `partial_derivatives` functions below have an overload where the
//   logical derivatives may be passed in instead of being computed. In the
//   overloads where the logical derivatives are not passed in they must be
//   computed. However, it is more efficient to allocate the memory for the
//   logical partial derivatives with respect to each coordinate at once. This
//   requires the `partial_derivatives_impl` to accept raw pointers to doubles
//   for the logical derivatives so it can be used for all overloads.
//
// - The resultant Variables `du` is a not_null pointer so that mutating compute
//   items can be supported.
//
// - The storage indices into the inverse Jacobian are precomputed to avoid
//   having to recompute them for each tensor component of `u`.
//
// - The DataVectors lhs and logical_du are non-owning DataVectors to be able to
//   plug into the optimized expression templates. This requires a `const_cast`
//   even though we will never change the `double*`.
//
// - Loop over every Tensor component in the variables by incrementing a raw
//   pointer to the contiguous data (vs. looping over each Tensor in the
//   variables with a tmpl::for_each then iterating over each component of this
//   Tensor).
//
// - We factor out the `logical_deriv_index == 0` case so that we do not need to
//   zero the memory in `du` before the computation.
template <typename ResultTags, size_t Dim, typename DerivativeFrame,
          typename ValueType = typename Variables<ResultTags>::value_type,
          typename VectorType = typename Variables<ResultTags>::vector_type>
void partial_derivatives_impl(
    const gsl::not_null<Variables<ResultTags>*> du,
    const std::array<const ValueType*, Dim>& logical_partial_derivatives_of_u,
    const size_t number_of_independent_components,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  ValueType* pdu = du->data();
  const size_t num_grid_points = du->number_of_grid_points();
  VectorType lhs{};
  VectorType logical_du{};

  std::array<std::array<size_t, Dim>, Dim> indices{};
  for (size_t deriv_index = 0; deriv_index < Dim; ++deriv_index) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(gsl::at(indices, d), deriv_index) =
          InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>::get_storage_index(d, deriv_index);
    }
  }

  for (size_t component_index = 0;
       component_index < number_of_independent_components; ++component_index) {
    for (size_t deriv_index = 0; deriv_index < Dim; ++deriv_index) {
      lhs.set_data_ref(pdu, num_grid_points);
      // clang-tidy: const cast is fine since we won't modify the data and we
      // need it to easily hook into the expression templates.
      logical_du.set_data_ref(
          const_cast<ValueType*>(                              // NOLINT
              gsl::at(logical_partial_derivatives_of_u, 0)) +  // NOLINT
              component_index * num_grid_points,
          num_grid_points);
      lhs = (*(inverse_jacobian.begin() + gsl::at(indices[0], deriv_index))) *
            logical_du;
      for (size_t logical_deriv_index = 1; logical_deriv_index < Dim;
           ++logical_deriv_index) {
        // clang-tidy: const cast is fine since we won't modify the data and we
        // need it to easily hook into the expression templates.
        logical_du.set_data_ref(const_cast<ValueType*>(  // NOLINT
                                    gsl::at(logical_partial_derivatives_of_u,
                                            logical_deriv_index)) +  // NOLINT
                                    component_index * num_grid_points,
                                num_grid_points);
        lhs +=
            (*(inverse_jacobian.begin() +
               gsl::at(gsl::at(indices, logical_deriv_index), deriv_index))) *
            logical_du;
      }
      // clang-tidy: no pointer arithmetic
      pdu += num_grid_points;  // NOLINT
    }
  }
}
}  // namespace partial_derivatives_detail

template <typename DerivativeTags, typename VariableTags, size_t Dim>
void logical_partial_derivatives(
    const gsl::not_null<std::array<Variables<DerivativeTags>, Dim>*>
        logical_partial_derivatives_of_u,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh) {
  using ValueType = typename Variables<VariableTags>::value_type;
  if (UNLIKELY((*logical_partial_derivatives_of_u)[0].number_of_grid_points() !=
               u.number_of_grid_points())) {
    for (auto& deriv : *logical_partial_derivatives_of_u) {
      deriv.initialize(u.number_of_grid_points());
    }
  }
  std::array<ValueType*, Dim> deriv_pointers{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(deriv_pointers, i) =
        gsl::at(*logical_partial_derivatives_of_u, i).data();
  }
  if constexpr (Dim == 1) {
    Variables<DerivativeTags>* temp = nullptr;
    partial_derivatives_detail::LogicalImpl<Dim, VariableTags, DerivativeTags>::
        apply(make_not_null(&deriv_pointers), temp, temp, u, mesh);
    return;
  } else {
    constexpr size_t n_comp =
        Variables<DerivativeTags>::number_of_independent_components;
    const size_t vars_size = u.number_of_grid_points() * n_comp;
    // For ZernikeB3, temp0 must hold spec_buf + even_data + odd_data
    // for filled_sphere_apply:
    //   n_spectral    = 2*(l_max+1)^2  [SPHEREPACK buffer, m_max = l_max]
    //   n_valid_modes = (l_max+1)^2    [valid coefficients]
    //   total = n_spectral + n_valid_modes = 3*(l_max+1)^2
    // temp1 then only needs even_result + odd_result = n_valid_modes,
    // which is always <= vars_size = n_ylm_phys * n_r * n_comp.
    size_t temp0_size = vars_size;
    if constexpr (Dim == 3) {
      if (mesh.basis(0) == Spectral::Basis::ZernikeB3) {
        const size_t l_max_p1 = mesh.extents(1);  // l_max + 1
        temp0_size = 3 * l_max_p1 * l_max_p1 * mesh.extents(0) * n_comp;
      }
    }
    auto buffer =
        cpp20::make_unique_for_overwrite<ValueType[]>(temp0_size + vars_size);
    Variables<DerivativeTags> temp0(&buffer[0], temp0_size);
    Variables<DerivativeTags> temp1(&buffer[temp0_size], vars_size);
    partial_derivatives_detail::LogicalImpl<Dim, VariableTags, DerivativeTags>::
        apply(make_not_null(&deriv_pointers), &temp0, &temp1, u, mesh);
  }
}

template <typename DerivativeTags, typename VariableTags, size_t Dim>
std::array<Variables<DerivativeTags>, Dim> logical_partial_derivatives(
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh) {
  auto logical_partial_derivatives_of_u =
      make_array<Dim>(Variables<DerivativeTags>(u.number_of_grid_points()));
  logical_partial_derivatives<DerivativeTags>(
      make_not_null(&logical_partial_derivatives_of_u), u, mesh);
  return logical_partial_derivatives_of_u;
}

template <typename ResultTags, typename DerivativeTags, size_t Dim,
          typename DerivativeFrame>
void partial_derivatives(
    const gsl::not_null<Variables<ResultTags>*> du,
    const std::array<Variables<DerivativeTags>, Dim>&
        logical_partial_derivatives_of_u,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  using ValueType = typename Variables<DerivativeTags>::value_type;
  auto& partial_derivatives_of_u = *du;
  // For mutating compute items we must set the size.
  if (UNLIKELY(partial_derivatives_of_u.number_of_grid_points() !=
               logical_partial_derivatives_of_u[0].number_of_grid_points())) {
    partial_derivatives_of_u.initialize(
        logical_partial_derivatives_of_u[0].number_of_grid_points());
  }

  std::array<const ValueType*, Dim> logical_derivs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(logical_derivs, i) =
        gsl::at(logical_partial_derivatives_of_u, i).data();
  }
  partial_derivatives_detail::partial_derivatives_impl(
      make_not_null(&partial_derivatives_of_u), logical_derivs,
      Variables<DerivativeTags>::number_of_independent_components,
      inverse_jacobian);
}

constexpr tnsr::iab<double, 3> Killing_vector_derivatives() {
  // holds derivative of both (0,-y,x,0) for \partial_y and (0,-z,0,x) for
  // \partial_z
  tnsr::iab<double, 3> da_killing_vectors{};
  for (size_t i = 0; i < da_killing_vectors.size(); ++i) {
    da_killing_vectors[i] = 0.0;
  }
  // y derivatives
  da_killing_vectors.get(1, 2, 1) = -1.0;
  da_killing_vectors.get(1, 1, 2) = 1.0;
  // z derivatives
  da_killing_vectors.get(2, 3, 1) = -1.0;
  da_killing_vectors.get(2, 1, 3) = 1.0;
  return da_killing_vectors;
}

template <typename Symm, typename Indices>
void cartoon_contraction(
    const gsl::not_null<Tensor<DataVector, Symm, Indices>*> result_tensor,
    const Tensor<DataVector, Symm, Indices>& tensor, const size_t deriv_index,
    const Spectral::Quadrature quad_type) {
  // This function does the contractions in Eqn. (218) of the SXS book's
  // Numerical Method's chapter, specifically the thing on the RHS that you take
  // the derivative of or divide by x.
  // Note that the result_tensor is written over and will have
  // (*result_tensor)[0].size() == tensor[0].size()
  auto& contract_tensor = *result_tensor;
  ASSERT(quad_type == Spectral::Quadrature::AxialSymmetry or
             quad_type == Spectral::Quadrature::SphericalSymmetry,
         "Must pass a valid Cartoon quadrature");

  constexpr tnsr::iab<double, 3> da_killing_vectors =
      Killing_vector_derivatives();
  const std::array<UpLo, Tensor<DataVector, Symm, Indices>::rank()> valences =
      tensor.index_valences();
  const auto type_index = tensor.index_types();

  const size_t valence_size = valences.size();
  auto input_tensor_array = make_array<valence_size>(0_st);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<size_t, 3> Killing_indices;
  Killing_indices[0] = deriv_index;

  for (size_t i = 0; i < tensor.size(); ++i) {
    const auto tensor_index = tensor.get_tensor_index(i);
    contract_tensor.get(tensor_index) = 0.0 * tensor.get(tensor_index);
    for (size_t rank = 0; rank < valence_size; ++rank) {
      const double sign = (valences[rank] == UpLo::Up) ? 1.0 : -1.0;
      const size_t max_dummy = type_index[rank] == IndexType::Spacetime ? 4 : 3;
      const size_t shift_index =
          type_index[rank] == IndexType::Spacetime ? 0 : 1;

      // loop over dimension of rank
      for (size_t dummy = 0; dummy < max_dummy; ++dummy) {
        if (valences[rank] == UpLo::Lo) {
          // covariant index
          Killing_indices[1] = tensor_index[rank] + shift_index;
          Killing_indices[2] = dummy + shift_index;
        } else {
          // contravariant
          Killing_indices[1] = dummy + shift_index;
          Killing_indices[2] = tensor_index[rank] + shift_index;
        }
        if (da_killing_vectors.get(Killing_indices) != 0) {
          for (size_t j = 0; j < tensor_index.size(); ++j) {
            input_tensor_array[j] = tensor_index[j];
          }
          input_tensor_array[rank] = dummy;

          contract_tensor.get(tensor_index) +=
              sign * tensor.get(input_tensor_array) *
              da_killing_vectors.get(Killing_indices);
        }
      }
    }
  }
}

template <typename InputTensorDataType, size_t Dim, typename Frame,
          Requires<Dim == 3> = nullptr>
void cartoon_derivative(
    TensorMetafunctions::prepend_spatial_index<InputTensorDataType, Dim,
                                               UpLo::Lo, Frame>& d_tensor,
    const InputTensorDataType& tensor, const DataVector& safe_x_coords,
    const Spectral::Quadrature quad_type) {
  // This function assumes `safe_x_coords` does not contain zero (safe in the
  // sense that having x in the denominator of an expression  will not cause
  // an FPE).
  // Therefore, if the domain actually does have zero, the coordinates must be
  // made safe by replacing all occurances of zero with a non-zero value, and
  // the output from this function is then not valid at those coords, one must
  // then manually use L'Hopital's rule
  ASSERT(quad_type == Spectral::Quadrature::AxialSymmetry or
             quad_type == Spectral::Quadrature::SphericalSymmetry,
         "Must pass a valid Cartoon quadrature");
  ASSERT(
      not equal_within_roundoff(0.0, safe_x_coords[0],
                                std::numeric_limits<double>::epsilon() * 100.0,
                                max(safe_x_coords)),
      "Invalid coordinate: safe_x_coords[0] is too close to zero ("
          << safe_x_coords[0]
          << "). Division by this value may cause an FPE if the value is zero. "
             "Maximum coordinate: "
          << max(safe_x_coords));

  const bool spherical_sym =
      quad_type == Spectral::Quadrature::SphericalSymmetry;
  const size_t start_deriv_index = spherical_sym ? 1 : 2;

  InputTensorDataType tensor_holder;

  for (size_t deriv_index = start_deriv_index; deriv_index < 3; ++deriv_index) {
    cartoon_contraction(make_not_null(&tensor_holder), tensor, deriv_index,
                        quad_type);
    for (size_t component_index = 0; component_index < tensor_holder.size();
         ++component_index) {
      const auto input_index = tensor_holder.get_tensor_index(component_index);
      const auto output_index = prepend(input_index, deriv_index);
      d_tensor.get(output_index) =
          tensor_holder.get(input_index) / safe_x_coords;
    }
  }
}

template <size_t CompDim, typename ResultTags, typename VariableTags,
          size_t Dim, typename DerivativeFrame, Requires<Dim == 3> = nullptr,
          typename ValueType = typename Variables<ResultTags>::value_type>
void partial_derivatives_with_cartoon_impl(
    const gsl::not_null<Variables<ResultTags>*> du,
    const Variables<VariableTags>& u,
    const std::array<const ValueType*, CompDim>& const_logical_derivs,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  using DerivativeTags =
      tmpl::front<tmpl::split_at<VariableTags, tmpl::size<ResultTags>>>;
  static_assert(
      std::is_same_v<
          tmpl::transform<ResultTags, tmpl::bind<tmpl::type_from, tmpl::_1>>,
          tmpl::transform<db::wrap_tags_in<Tags::deriv, DerivativeTags,
                                           tmpl::size_t<Dim>, DerivativeFrame>,
                          tmpl::bind<tmpl::type_from, tmpl::_1>>>);

  const Spectral::Quadrature quad_type = mesh.quadrature(2);
  const auto numerical_deriv_in_this_direction = [](const size_t deriv_num) {
    if constexpr (CompDim == 2) {
      return deriv_num == 0 or deriv_num == 1;
    } else {
      return deriv_num == 0;
    }
  };
  // only the 1st (possibly 2nd) dimensions of the Jacobian are relevant here
  const InverseJacobian<DataVector, CompDim, Frame::ElementLogical,
                        DerivativeFrame>
      inverse_jacobian{inverse_jacobian_3d.begin()->size()};
  for (size_t i = 0; i < CompDim; ++i) {
    for (size_t j = 0; j < CompDim; ++j) {
      make_const_view(make_not_null(&inverse_jacobian.get(i, j)),
                      inverse_jacobian_3d.get(i, j), 0,
                      inverse_jacobian_3d.begin()->size());
    }
  }
  // pdu points to du
  ValueType* pdu = du->data();
  const size_t num_grid_points = du->number_of_grid_points();
  DataVector lhs{};
  DataVector logical_du{};
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
  using VariableTypes = tmpl::remove_duplicates<
      tmpl::transform<DerivativeTags, tmpl::bind<tmpl::type_from, tmpl::_1>>>;

  using TempTags0 = ::Tags::convert_to_temp_tensors<VariableTypes, 0>;
  using TempTags1 = ::Tags::convert_to_temp_tensors<VariableTypes, 1>;
  using VarTags = tmpl::append<TempTags0, TempTags1>;

  Variables<VarTags> temp_vars{};
  if (element_contains_zero_of_symmetry_axis) {
    const size_t y_extents = mesh.extents(1);
    temp_vars.initialize(y_extents);
  }

  std::array<std::array<size_t, CompDim>, CompDim> indices{};
  for (size_t deriv_index = 0; deriv_index < CompDim; ++deriv_index) {
    for (size_t d = 0; d < CompDim; ++d) {
      gsl::at(gsl::at(indices, d), deriv_index) =
          InverseJacobian<DataVector, CompDim, Frame::ElementLogical,
                          DerivativeFrame>::get_storage_index(d, deriv_index);
    }
  }

  // the non-cartoon derivatives need to know where they are within `u` to
  // contract with the inverse Jacobian to compute the inertial derivatives
  size_t global_component_index = 0;
  tmpl::for_each<DerivativeTags>(
      [&u, &du, &lhs, &pdu, &num_grid_points, &logical_du,
       &const_logical_derivs, &inverse_jacobian, &indices, &temp_vars,
       &global_component_index, &safe_x_coords,
       &numerical_deriv_in_this_direction, &quad_type, &mesh,
       &element_contains_zero_of_symmetry_axis]<typename TensorTag>(
          tmpl::type_<TensorTag> /*meta*/) {
        auto& tensor = get<TensorTag>(u);

        TensorMetafunctions::prepend_spatial_index<
            typename TensorTag::type, Dim, UpLo::Lo, Frame::Inertial>
            cart_deriv_tensor;
        cartoon_derivative<typename TensorTag::type, 3, Frame::Inertial>(
            cart_deriv_tensor, tensor, safe_x_coords, quad_type);

        for (size_t component_index = 0;
             component_index < TensorTag::type::size();
             ++component_index, ++global_component_index) {
          for (size_t deriv_index = 0; deriv_index < Dim; ++deriv_index) {
            // lhs points to pdu (shifts by num grid points below)
            lhs.set_data_ref(pdu, num_grid_points);

            if (numerical_deriv_in_this_direction(deriv_index)) {
              // do normal derivative for x (and possibly y) derivative
              // clang-tidy: const cast is fine since we won't modify the data
              // and we need it to easily hook into the expression templates.
              logical_du.set_data_ref(
                  const_cast<ValueType*>(                  // NOLINT
                      gsl::at(const_logical_derivs, 0)) +  // NOLINT
                      global_component_index * num_grid_points,
                  num_grid_points);
              lhs = (*(inverse_jacobian.begin() +
                       gsl::at(indices[0], deriv_index))) *
                    logical_du;
              if constexpr (CompDim == 2) {
                // clang-tidy: const cast is fine since we won't modify the data
                // and we need it to easily hook into the expression templates.
                logical_du.set_data_ref(
                    const_cast<ValueType*>(                  // NOLINT
                        gsl::at(const_logical_derivs, 1)) +  // NOLINT
                        global_component_index * num_grid_points,
                    num_grid_points);
                lhs += (*(inverse_jacobian.begin() +
                          gsl::at(gsl::at(indices, 1), deriv_index))) *
                       logical_du;
              }
            } else {
              // do cartoon dervative
              const auto input_tensor_index =
                  tensor.get_tensor_index(component_index);
              const auto output_tensor_index =
                  prepend(input_tensor_index, size_t{deriv_index});
              lhs = cart_deriv_tensor.get(output_tensor_index);
            }
            // clang-tidy: no pointer arithmetic
            pdu += num_grid_points;  // NOLINT
          }
        }
        if (element_contains_zero_of_symmetry_axis) {
          // need to use L'Hopital
          using dtensor_type =
              tmpl::at<ResultTags, tmpl::index_of<DerivativeTags, TensorTag>>;
          auto& dtensor = get<dtensor_type>(*du);

          // if spherical symmetry,the zero is only at the first index
          // if axial symmetry, the zero is repeated every x_extents times
          const size_t x_extents = mesh.extents(0);
          // for spherical symmetry, y_extents = 1, so we could use a double
          // type rather than a size 1 DataVector, but it requires a lot of code
          // reuse
          const size_t y_extents = mesh.extents(1);

          auto& dx_tensor =
              get<::Tags::TempTensor<0, typename TensorTag::type>>(temp_vars);
          auto& contracted_dx_tensor =
              get<::Tags::TempTensor<1, typename TensorTag::type>>(temp_vars);

          for (size_t component_index = 0; component_index < dx_tensor.size();
               ++component_index) {
            const auto output_index =
                dx_tensor.get_tensor_index(component_index);
            // the 0 is because we want the x derivative
            const size_t input_index =
                dtensor.get_storage_index(prepend(output_index, size_t{0}));
            for (size_t i = 0; i < y_extents; ++i) {
              dx_tensor[component_index][i] =
                  dtensor[input_index][i * x_extents];
            }
          }

          const size_t start_index =
              quad_type == Spectral::Quadrature::SphericalSymmetry ? 1 : 2;
          for (size_t deriv_index = start_index; deriv_index < 3;
               ++deriv_index) {
            cartoon_contraction(make_not_null(&contracted_dx_tensor), dx_tensor,
                                deriv_index, quad_type);
            for (size_t component_index = 0;
                 component_index < contracted_dx_tensor.size();
                 ++component_index) {
              const auto input_index =
                  contracted_dx_tensor.get_tensor_index(component_index);
              const size_t output_index =
                  dtensor.get_storage_index(prepend(input_index, deriv_index));
              for (size_t i = 0; i < y_extents; ++i) {
                dtensor[output_index][i * x_extents] =
                    contracted_dx_tensor[component_index][i];
              }
            }
          }
        }
      });
}

template <size_t CompDim, typename ResultTags, typename VariableTags,
          size_t Dim, typename DerivativeFrame>
void cartoon_partial_derivatives_apply(
    const gsl::not_null<Variables<ResultTags>*> du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  using DerivativeTags =
      tmpl::front<tmpl::split_at<VariableTags, tmpl::size<ResultTags>>>;
  static_assert(Dim == 3);
  static_assert(
      std::is_same_v<
          tmpl::transform<ResultTags, tmpl::bind<tmpl::type_from, tmpl::_1>>,
          tmpl::transform<db::wrap_tags_in<Tags::deriv, DerivativeTags,
                                           tmpl::size_t<Dim>, DerivativeFrame>,
                          tmpl::bind<tmpl::type_from, tmpl::_1>>>);
  ASSERT((CompDim == 2 and
          mesh.quadrature(2) == Spectral::Quadrature::AxialSymmetry) or
             (CompDim == 1 and
              (mesh.quadrature(1) == Spectral::Quadrature::SphericalSymmetry and
               mesh.quadrature(2) == Spectral::Quadrature::SphericalSymmetry)),
         "Invalid Quadrature combinations: axial symmetry requires 2 "
         "non-Cartoon dimensions, spherical symmetry requires 1 non-Cartoon "
         "dimension. Got: "
             << mesh.quadrature());

  using ValueType = typename Variables<VariableTags>::value_type;
  auto& partial_derivatives_of_u = *du;

  if (UNLIKELY(partial_derivatives_of_u.number_of_grid_points() !=
               mesh.number_of_grid_points())) {
    partial_derivatives_of_u.initialize(mesh.number_of_grid_points());
  }

  const size_t vars_size =
      u.number_of_grid_points() *
      Variables<DerivativeTags>::number_of_independent_components;
  const auto logical_derivs_data =
      cpp20::make_unique_for_overwrite<ValueType[]>(
          (CompDim > 1 ? (CompDim + 1) : CompDim) * vars_size);

  std::array<ValueType*, CompDim> logical_derivs{};
  for (size_t i = 0; i < CompDim; ++i) {
    gsl::at(logical_derivs, i) = &(logical_derivs_data[i * vars_size]);
  }
  Variables<DerivativeTags> temp{};

  if constexpr (CompDim == 1) {
    partial_derivatives_detail::
        LogicalImpl<CompDim, VariableTags, DerivativeTags>::apply(
            make_not_null(&logical_derivs), &partial_derivatives_of_u, &temp, u,
            mesh.slice_through(0));
  } else {
    temp.set_data_ref(&logical_derivs_data[CompDim * vars_size], vars_size);
    partial_derivatives_detail::
        LogicalImpl<CompDim, VariableTags, DerivativeTags>::apply(
            make_not_null(&logical_derivs), &partial_derivatives_of_u, &temp, u,
            mesh.slice_through(0, 1));
  }

  std::array<const ValueType*, CompDim> const_logical_derivs{};
  for (size_t i = 0; i < CompDim; ++i) {
    gsl::at(const_logical_derivs, i) = gsl::at(logical_derivs, i);
  }

  partial_derivatives_with_cartoon_impl(du, u, const_logical_derivs, mesh,
                                        inverse_jacobian_3d, inertial_coords);
}

template <typename ResultTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame, Requires<Dim == 3>>
void cartoon_partial_derivatives(
    const gsl::not_null<Variables<ResultTags>*> du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  static_assert(Dim == 3);
  if (mesh.basis(0) != Spectral::Basis::Cartoon and
      mesh.basis(2) == Spectral::Basis::Cartoon) {
    // computational dimension needs to be a constexpr
    if (mesh.basis(1) == Spectral::Basis::Cartoon) {
      cartoon_partial_derivatives_apply<1, ResultTags, VariableTags, Dim,
                                        DerivativeFrame>(
          du, u, mesh, inverse_jacobian_3d, inertial_coords);
    } else {
      cartoon_partial_derivatives_apply<2, ResultTags, VariableTags, Dim,
                                        DerivativeFrame>(
          du, u, mesh, inverse_jacobian_3d, inertial_coords);
    }
  } else {
    ERROR("Bases do not match valid Cartoon pattern.");
  }
}

template <typename ResultTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
void partial_derivatives(
    const gsl::not_null<Variables<ResultTags>*> du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  using DerivativeTags =
      tmpl::front<tmpl::split_at<VariableTags, tmpl::size<ResultTags>>>;
  static_assert(
      std::is_same_v<
          tmpl::transform<ResultTags, tmpl::bind<tmpl::type_from, tmpl::_1>>,
          tmpl::transform<db::wrap_tags_in<Tags::deriv, DerivativeTags,
                                           tmpl::size_t<Dim>, DerivativeFrame>,
                          tmpl::bind<tmpl::type_from, tmpl::_1>>>);
  using ValueType = typename Variables<VariableTags>::value_type;
  auto& partial_derivatives_of_u = *du;
  // For mutating compute items we must set the size.
  if (UNLIKELY(partial_derivatives_of_u.number_of_grid_points() !=
               mesh.number_of_grid_points())) {
    partial_derivatives_of_u.initialize(mesh.number_of_grid_points());
  }

  const size_t vars_size =
      u.number_of_grid_points() *
      Variables<DerivativeTags>::number_of_independent_components;
  const auto logical_derivs_data =
      cpp20::make_unique_for_overwrite<ValueType[]>(
          (Dim > 1 ? (Dim + 1) : Dim) * vars_size);
  std::array<ValueType*, Dim> logical_derivs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(logical_derivs, i) = &(logical_derivs_data[i * vars_size]);
  }
  Variables<DerivativeTags> temp{};
  if constexpr (Dim > 1) {
    temp.set_data_ref(&logical_derivs_data[Dim * vars_size], vars_size);
  }
  partial_derivatives_detail::LogicalImpl<
      Dim, VariableTags, DerivativeTags>::apply(make_not_null(&logical_derivs),
                                                &partial_derivatives_of_u,
                                                &temp, u, mesh);

  std::array<const ValueType*, Dim> const_logical_derivs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(const_logical_derivs, i) = gsl::at(logical_derivs, i);
  }
  partial_derivatives_detail::partial_derivatives_impl(
      make_not_null(&partial_derivatives_of_u), const_logical_derivs,
      Variables<DerivativeTags>::number_of_independent_components,
      inverse_jacobian);
}

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags, tmpl::size_t<Dim>,
                           DerivativeFrame>>
partial_derivatives(
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags, tmpl::size_t<Dim>,
                             DerivativeFrame>>
      partial_derivatives_of_u(u.number_of_grid_points());
  partial_derivatives(make_not_null(&partial_derivatives_of_u), u, mesh,
                      inverse_jacobian);
  return partial_derivatives_of_u;
}

template <typename ResultTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
void partial_derivatives(
    const gsl::not_null<Variables<ResultTags>*> du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  if constexpr (Dim == 3) {
    if (mesh.basis(2) == Spectral::Basis::Cartoon) {
      cartoon_partial_derivatives(du, u, mesh, inverse_jacobian,
                                  inertial_coords);
    } else {
      partial_derivatives(du, u, mesh, inverse_jacobian);
    }
  } else {
    (void)inertial_coords;
    partial_derivatives(du, u, mesh, inverse_jacobian);
  }
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame, Requires<Dim == 3>>
void cartoon_partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame>*>
        du,
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  auto& du_tnsr = *du;
  using u_type = Tensor<DataVector, SymmList, IndexList>;
  using du_type =
      TensorMetafunctions::prepend_spatial_index<u_type, Dim, UpLo::Lo,
                                                 DerivativeFrame>;
  using vars_list = ::Tags::convert_to_temp_tensors<tmpl::list<u_type>, 0>;
  using d_vars_list = ::Tags::convert_to_temp_tensors<tmpl::list<du_type>, 0>;
  Variables<vars_list> u_vars{mesh.number_of_grid_points()};
  get<tmpl::front<vars_list>>(u_vars) = u;
  Variables<d_vars_list> du_vars{mesh.number_of_grid_points()};
  cartoon_partial_derivatives(make_not_null(&du_vars), u_vars, mesh,
                              inverse_jacobian, inertial_coords);
  du_tnsr = get<tmpl::front<d_vars_list>>(du_vars);
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame, Requires<Dim == 3>>
auto cartoon_partial_derivative(
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame> {
  using u_type = Tensor<DataVector, SymmList, IndexList>;
  using du_type =
      TensorMetafunctions::prepend_spatial_index<u_type, Dim, UpLo::Lo,
                                                 DerivativeFrame>;
  using vars_list = ::Tags::convert_to_temp_tensors<tmpl::list<u_type>, 0>;
  using d_vars_list = ::Tags::convert_to_temp_tensors<tmpl::list<du_type>, 0>;
  Variables<vars_list> u_vars{mesh.number_of_grid_points()};
  get<tmpl::front<vars_list>>(u_vars) = u;
  Variables<d_vars_list> du_vars{mesh.number_of_grid_points()};
  cartoon_partial_derivatives(make_not_null(&du_vars), u_vars, mesh,
                              inverse_jacobian, inertial_coords);
  return get<tmpl::front<d_vars_list>>(du_vars);
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame>*>
        du,
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  if constexpr (Dim == 3) {
    if (mesh.basis(2) == Spectral::Basis::Cartoon) {
      cartoon_partial_derivative(du, u, mesh, inverse_jacobian,
                                 inertial_coords);
    } else {
      partial_derivative(du, u, mesh, inverse_jacobian);
    }
  } else {
    (void)inertial_coords;
    partial_derivative(du, u, mesh, inverse_jacobian);
  }
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
auto partial_derivative(
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame> {
  if constexpr (Dim == 3) {
    if (mesh.basis(2) == Spectral::Basis::Cartoon) {
      return cartoon_partial_derivative(u, mesh, inverse_jacobian,
                                        inertial_coords);
    } else {
      return partial_derivative(u, mesh, inverse_jacobian);
    }
  } else {
    (void)inertial_coords;
    return partial_derivative(u, mesh, inverse_jacobian);
  }
}

namespace partial_derivatives_detail {
template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<1, VariableTags, DerivativeTags> {
  static constexpr const size_t Dim = 1;
  template <typename T, typename ValueType = typename Variables<T>::value_type>
  static void apply(
      const gsl::not_null<std::array<ValueType*, Dim>*> logical_du,
      Variables<T>* /*unused_in_1d*/,
      Variables<DerivativeTags>* const /*unused_in_1d*/,
      const Variables<VariableTags>& u, const Mesh<Dim>& mesh) {
    auto& logical_partial_derivatives_of_u = *logical_du;
    const size_t deriv_size =
        Variables<DerivativeTags>::number_of_independent_components *
        u.number_of_grid_points();
    if (UNLIKELY(mesh.basis(0) == Spectral::Basis::ZernikeB1)) {
      if constexpr (std::is_same_v<ValueType, double>) {
        zernikeb1_apply(make_not_null(logical_partial_derivatives_of_u[0]), u,
                        mesh.number_of_grid_points(), mesh.extents(0));
      } else {
        ERROR(
            "Support for complex numbers with ZernikeB1 is not yet implemented "
            "for logical_partial_derivative.");
      }
    } else {
      const Matrix& differentiation_matrix_xi =
          Spectral::differentiation_matrix(mesh.slice_through(0));
      apply_matrix_in_first_dim(logical_partial_derivatives_of_u[0], u.data(),
                                differentiation_matrix_xi, deriv_size);
    }
  }

  template <typename T, typename ValueType = typename Variables<T>::value_type>
  static void zernikeb1_apply(
      const gsl::not_null<ValueType*> logical_partial_derivative_of_u,
      const Variables<VariableTags>& u, const size_t num_grid_points,
      const size_t extents_xi, Variables<T>* const buffer1,
      Variables<DerivativeTags>* const buffer2) {
    // Zernike bases need a differentiation matrix based on their parity. For
    // ZernikeB1, that information is stored in the index structure.
    // We sort components to a buffer by parity, apply the appropriate
    // matrices, and copy the values to their destination

    const Matrix& even_diff_matrix =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB1,
                                         Spectral::Quadrature::GaussRadauUpper>(
            extents_xi, Spectral::Parity::Even);
    const Matrix& odd_diff_matrix =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB1,
                                         Spectral::Quadrature::GaussRadauUpper>(
            extents_xi, Spectral::Parity::Odd);

    ValueType* p_logical_derivs = logical_partial_derivative_of_u;
    const ValueType* p_data = u.data();
    bool even = true;

    constexpr auto parity_info =
        Spectral::compute_parity_list<DerivativeTags>();
    const auto parity_list = std::get<0>(parity_info);
    const auto num_even_comps = std::get<1>(parity_info);
    const auto num_odd_comps = std::get<2>(parity_info);

    const size_t required_buffer_size =
        (num_even_comps + num_odd_comps) * num_grid_points;
    ASSERT(buffer1->size() >= required_buffer_size,
           "Buffer1 size (" << buffer1->size()
                            << ") is insufficient for ZernikeB1 requirements ("
                            << required_buffer_size << ")");
    ASSERT(buffer2->size() >= required_buffer_size,
           "Buffer2 size (" << buffer2->size()
                            << ") is insufficient for ZernikeB1 requirements ("
                            << required_buffer_size << ")");

    using DataType = std::conditional_t<std::is_same_v<ValueType, double>,
                                        DataVector, ComplexDataVector>;
    DataType even_components{};
    even_components.set_data_ref(buffer1->data(),
                                 num_even_comps * num_grid_points);
    DataType odd_components{};
    odd_components.set_data_ref(
        buffer1->data() + num_even_comps * num_grid_points,
        num_odd_comps * num_grid_points);
    ValueType* p_even_components = even_components.data();
    ValueType* p_odd_components = odd_components.data();

    DataType even_deriv{};
    even_deriv.set_data_ref(buffer2->data(), num_even_comps * num_grid_points);
    DataType odd_deriv{};
    odd_deriv.set_data_ref(buffer2->data() + num_even_comps * num_grid_points,
                           num_odd_comps * num_grid_points);
    ValueType* p_even_deriv = even_deriv.data();
    ValueType* p_odd_deriv = odd_deriv.data();

    for (const auto seg_size : parity_list) {
      // seg_size is the number of contiguous components with the same parity
      // (based on the bool `even`). The pattern alternates even/odd, so if the
      // first element is odd, the list has a leading 0. However, after the
      // potential leading 0, any other zero in the list indicates all
      // components have been handled and all further values are guaranteed to
      // be zero (the list has its size set by the worst-case scenario). If
      // there are any neighboring components with the same parity (so not
      // worst-case size), the list will have the back padded with zeros, which
      // we don't need to loop over
      // See Spectral::compute_parity_list() for more info
      if (seg_size == 0) {
        if (even) {
          // Will have leading zero if first element is odd, so check next value
          even = false;
          continue;
        } else {
          // Iterated through all non-zero values
          break;
        }
      }
      if (even) {
        std::copy(p_data, p_data + seg_size * num_grid_points,
                  p_even_components);
        p_even_components += seg_size * num_grid_points;
      } else {
        std::copy(p_data, p_data + seg_size * num_grid_points,
                  p_odd_components);
        p_odd_components += seg_size * num_grid_points;
      }
      p_data += seg_size * num_grid_points;  // NOLINT
      even = not even;
    }
    apply_matrix_in_first_dim(p_even_deriv, even_components.data(),
                              even_diff_matrix,
                              num_even_comps * num_grid_points);
    apply_matrix_in_first_dim(p_odd_deriv, odd_components.data(),
                              odd_diff_matrix, num_odd_comps * num_grid_points);
    even = true;
    for (const auto seg_size : parity_list) {
      if (seg_size == 0) {
        if (even) {
          // Will have leading zero if first element is odd
          even = not even;
          continue;
        } else {
          // Iterated through all non-zero values
          break;
        }
      }
      // Now copy the data from the buffers to the correct location in
      // p_logical_derivs
      if (even) {
        std::copy(p_even_deriv, p_even_deriv + seg_size * num_grid_points,
                  p_logical_derivs);
        p_even_deriv += seg_size * num_grid_points;
      } else {
        std::copy(p_odd_deriv, p_odd_deriv + seg_size * num_grid_points,
                  p_logical_derivs);
        p_odd_deriv += seg_size * num_grid_points;
      }
      p_logical_derivs += seg_size * num_grid_points;  // NOLINT
      even = not even;
    }
  }

  template <typename ValueType>
  static void zernikeb1_apply(
      const gsl::not_null<ValueType*> logical_partial_derivative_of_u,
      const Variables<VariableTags>& u, const size_t num_grid_points,
      const size_t extents_0) {
    constexpr auto parity_info =
        Spectral::compute_parity_list<DerivativeTags>();
    const auto num_comps = std::get<1>(parity_info) + std::get<2>(parity_info);

    // Allocate buffers locally for this overload (use 2 buffers to match the
    // 2 buffers used in the LogicalImpl<2> call)
    const auto logical_derivs_data =
        cpp20::make_unique_for_overwrite<ValueType[]>(2 * num_comps *
                                                      num_grid_points);
    Variables<DerivativeTags> buffer1{};
    buffer1.set_data_ref(&logical_derivs_data[0], num_comps * num_grid_points);
    Variables<DerivativeTags> buffer2{};
    buffer2.set_data_ref(&logical_derivs_data[num_comps * num_grid_points],
                         num_comps * num_grid_points);

    zernikeb1_apply(logical_partial_derivative_of_u, u, num_grid_points,
                    extents_0, &buffer1, &buffer2);
  }
};

template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<2, VariableTags, DerivativeTags> {
  static constexpr size_t Dim = 2;
  template <typename T, typename ValueType = typename Variables<T>::value_type>
  static void apply(
      const gsl::not_null<std::array<ValueType*, Dim>*> logical_du,
      Variables<T>* const partial_u_wrt_eta,
      Variables<DerivativeTags>* const u_eta_fastest,
      const Variables<VariableTags>& u, const Mesh<2>& mesh) {
    static_assert(
        Variables<DerivativeTags>::number_of_independent_components <=
            Variables<T>::number_of_independent_components,
        "Temporary buffer in logical partial derivatives is too small");
    if (mesh.basis(0) == Spectral::Basis::ZernikeB2) {
      if constexpr (std::is_same_v<ValueType, double>) {
        disk_apply(logical_du, partial_u_wrt_eta, u_eta_fastest, u, mesh);
      } else {
        ERROR(
            "Support for complex numbers with a disk is not yet "
            "implemented for logical_partial_derivative.");
      }
    } else {
      auto& logical_partial_derivatives_of_u = *logical_du;
      const size_t deriv_size =
          Variables<DerivativeTags>::number_of_independent_components *
          u.number_of_grid_points();
      if (UNLIKELY(mesh.basis(0) == Spectral::Basis::ZernikeB1)) {
        if constexpr (std::is_same_v<ValueType, double>) {
          LogicalImpl<1, VariableTags, DerivativeTags>::zernikeb1_apply(
              make_not_null(logical_partial_derivatives_of_u[0]), u,
              mesh.number_of_grid_points(), mesh.extents(0), partial_u_wrt_eta,
              u_eta_fastest);
        } else {
          ERROR(
              "Support for complex numbers with ZernikeB1 is not yet "
              "implemented "
              "for logical_partial_derivative.");
        }
      } else {
        const Matrix& differentiation_matrix_xi =
            Spectral::differentiation_matrix(mesh.slice_through(0));
        apply_matrix_in_first_dim(logical_partial_derivatives_of_u[0], u.data(),
                                  differentiation_matrix_xi, deriv_size);
      }
      const size_t num_components_times_xi_slices =
          deriv_size / mesh.extents(0);
      transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
          make_not_null(u_eta_fastest), u, mesh.extents(0),
          num_components_times_xi_slices);
      const Matrix& differentiation_matrix_eta =
          Spectral::differentiation_matrix(mesh.slice_through(1));
      apply_matrix_in_first_dim(partial_u_wrt_eta->data(),
                                u_eta_fastest->data(),
                                differentiation_matrix_eta, deriv_size);
      raw_transpose(make_not_null(logical_partial_derivatives_of_u[1]),
                    partial_u_wrt_eta->data(), num_components_times_xi_slices,
                    mesh.extents(0));
    }
  }

  template <typename T, typename ValueType = typename Variables<T>::value_type>
  static void disk_apply(
      const gsl::not_null<std::array<ValueType*, Dim>*> logical_du,
      Variables<T>* const partial_u_wrt_eta,
      Variables<DerivativeTags>* const u_eta_fastest,
      const Variables<VariableTags>& u, const Mesh<2>& mesh) {
    ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
               mesh.basis(1) == Spectral::Basis::ZernikeB2,
           "Unexpected basis combination: " << mesh.basis());
    const auto [n_r, n_phi] = mesh.extents().indices();
    const size_t n_r_max = 2 * n_r - 2;
    ASSERT(n_phi % 2 == 1,
           "Fourier with an even number of grid points can be unstable due to "
           "the top derivative not being representable");
    ASSERT(n_phi / 2 <= n_r_max,
           "Zernike & Fourier on a disk have angular resolution limited by "
           "extents in both dimensions. We choose to enforce the restriction "
           "that the Fourier modal space is not larger than the Zernike "
           "angular capabilities (which would waste space and be physically "
           "ill-motivated).\nn_phi / 2 = "
               << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);

    const Matrix& diff_matrix_r_even =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB2,
                                         Spectral::Quadrature::GaussRadauUpper>(
            n_r, Spectral::Parity::Even);
    const Matrix& diff_matrix_r_odd =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB2,
                                         Spectral::Quadrature::GaussRadauUpper>(
            n_r, Spectral::Parity::Odd);
    const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

    const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
    const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

    const size_t local_buffer_size =
        Variables<DerivativeTags>::number_of_independent_components *
        u.number_of_grid_points();
    // These just point to the passed buffers but it gets confusing to read when
    // they are used for radial purposes that don't match their names
    Variables<DerivativeTags> local_buffer1{};
    Variables<DerivativeTags> local_buffer2{};
    local_buffer1.set_data_ref(partial_u_wrt_eta->data(), local_buffer_size);
    local_buffer2.set_data_ref(u_eta_fastest->data(), local_buffer_size);

    const size_t deriv_size =
        Variables<DerivativeTags>::number_of_independent_components *
        u.number_of_grid_points();
    const size_t num_components_times_xi_slices = deriv_size / mesh.extents(0);

    // phi derivative
    transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
        make_not_null(u_eta_fastest), u, mesh.extents(0),
        num_components_times_xi_slices);
    apply_matrix_in_first_dim(partial_u_wrt_eta->data(), u_eta_fastest->data(),
                              diff_matrix_phi, deriv_size);
    raw_transpose(make_not_null((*logical_du)[1]), partial_u_wrt_eta->data(),
                  num_components_times_xi_slices, mesh.extents(0));

    // There are a few ways to achieve the task of applying the appropriate even
    // or odd radial differentiation matrices.
    // Testing showed that manual looping to apply the matrices on contiguous
    // even/odd subsets was moderately slower than reworking the data to only
    // apply 2 BLAS calls on the full even/odd subsets.
    // One can rework the data two ways to get the subsets:
    //  (1) copy the even and odd columns so the different components are split
    // in two different buffers
    //  (2) copy the entire data and then mask with zeros where appropriate.
    // Option (1) uses less space for the BLAS calls as they act on truly dense
    // matrices, whereas option (2) maintains the structure of the columns by
    // adding zeros. Option (1) uses more copies but smaller BLAS operations,
    // which testing showed to be slightly faster.

    // reuse transposed data to go to angular spectral space
    apply_matrix_in_first_dim((*logical_du)[0], u_eta_fastest->data(),
                              nodal_to_modal_phi, deriv_size);
    // (*logical_du)[0] holds transposed angular modal rep

    raw_transpose(make_not_null(local_buffer1.data()), (*logical_du)[0],
                  num_components_times_xi_slices, mesh.extents(0));
    // local_buffer 1 holds angular modal rep

    // Copy even/odd subsets to compact regions
    const size_t n_even_cols =
        (1 + 2 * ((n_phi - 1) / 4)) *
        Variables<DerivativeTags>::number_of_independent_components;
    const size_t n_odd_cols =
        (2 * ((n_phi + 1) / 4)) *
        Variables<DerivativeTags>::number_of_independent_components;

    // Use local_buffer2 for initial compact storage
    const size_t even_size = n_even_cols * n_r;
    const size_t odd_size = n_odd_cols * n_r;
    ValueType* even_data = local_buffer2.data();
    ValueType* odd_data = local_buffer2.data() + even_size;

    size_t even_col = 0;
    size_t odd_col = 0;
    for (size_t i = 0;
         i <
         Variables<DerivativeTags>::number_of_independent_components * n_phi;
         ++i) {
      if (i % n_phi == 0) {
        // m = 0 (even) - copy single column
        std::copy(local_buffer1.data() + i * n_r,
                  local_buffer1.data() + (i + 1) * n_r,
                  even_data + even_col * n_r);
        ++even_col;
      } else if ((i % n_phi - 1) / 2 % 2 == 1) {
        // Even m with cos/sin pair - copy two columns
        std::copy(local_buffer1.data() + i * n_r,
                  local_buffer1.data() + (i + 2) * n_r,
                  even_data + even_col * n_r);
        even_col += 2;
        ++i;
      } else {
        // Odd m with cos/sin pair - copy two columns
        std::copy(local_buffer1.data() + i * n_r,
                  local_buffer1.data() + (i + 2) * n_r,
                  odd_data + odd_col * n_r);
        odd_col += 2;
        ++i;
      }
    }

    // Use local_buffer1 for differentiated compact storage
    ValueType* even_result = local_buffer1.data();
    ValueType* odd_result = local_buffer1.data() + even_size;

    apply_matrix_in_first_dim(even_result, even_data, diff_matrix_r_even,
                              even_size, false);

    // Copy even derivative components back to interleaved regions
    even_col = 0;
    for (size_t i = 0;
         i <
         Variables<DerivativeTags>::number_of_independent_components * n_phi;
         ++i) {
      if (i % n_phi == 0) {
        // m = 0 (even) - copy single column back
        std::copy(even_result + even_col * n_r,
                  even_result + (even_col + 1) * n_r,
                  (*logical_du)[0] + i * n_r);
        ++even_col;
      } else if ((i % n_phi - 1) / 2 % 2 == 1) {
        // Even m with cos/sin pair - copy two columns back
        std::copy(even_result + even_col * n_r,
                  even_result + (even_col + 2) * n_r,
                  (*logical_du)[0] + i * n_r);
        even_col += 2;
        ++i;
      } else {
        ++i;
      }
    }

    apply_matrix_in_first_dim(odd_result, odd_data, diff_matrix_r_odd, odd_size,
                              false);

    // Copy odd derivative components back to interleaved regions
    odd_col = 0;
    for (size_t i = 0;
         i <
         Variables<DerivativeTags>::number_of_independent_components * n_phi;
         ++i) {
      if (i % n_phi == 0) {
        // m = 0 (even) - skip
      } else if ((i % n_phi - 1) / 2 % 2 == 1) {
        // Even m - skip
        ++i;
      } else {
        // Odd m with cos/sin pair - copy both columns
        std::copy(odd_result + odd_col * n_r, odd_result + (odd_col + 2) * n_r,
                  (*logical_du)[0] + i * n_r);
        odd_col += 2;
        ++i;
      }
    }
    // (*logical_du)[0] holds the radial derivative in angular modal rep

    raw_transpose(make_not_null(local_buffer1.data()), (*logical_du)[0],
                  mesh.extents(0), num_components_times_xi_slices);
    // local_buffer1 holds transposed radial derivative in angular modal rep

    apply_matrix_in_first_dim(local_buffer2.data(), local_buffer1.data(),
                              modal_to_nodal_phi, deriv_size);
    // local_buffer2 holds transposed radial derivative in angular nodal rep

    raw_transpose(make_not_null((*logical_du)[0]), local_buffer2.data(),
                  num_components_times_xi_slices, mesh.extents(0));
  }
};

template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<3, VariableTags, DerivativeTags> {
  static constexpr size_t Dim = 3;
  template <class T, typename ValueType = typename Variables<T>::value_type>
  static void apply(
      const gsl::not_null<std::array<ValueType*, Dim>*> logical_du,
      Variables<T>* const partial_u_wrt_eta_or_zeta,
      Variables<DerivativeTags>* const u_eta_or_zeta_fastest,
      const Variables<VariableTags>& u, const Mesh<3>& mesh) {
    if (mesh.basis(1) == Spectral::Basis::SphericalHarmonic) {
      if constexpr (std::is_same_v<ValueType, double>) {
        spherical_apply(logical_du, u, mesh);
      } else {
        ERROR(
            "Support for complex numbers with spherical harmonics is not yet "
            "implemented for logical_partial_derivative.");
      }
    } else if (mesh.basis(0) == Spectral::Basis::ZernikeB3) {
      if constexpr (std::is_same_v<ValueType, double>) {
        filled_sphere_apply(logical_du, partial_u_wrt_eta_or_zeta,
                            u_eta_or_zeta_fastest, u, mesh);
      } else {
        ERROR(
            "Support for complex numbers with the filled sphere is not yet "
            "implemented for logical_partial_derivative.");
      }
    } else if (mesh.basis(0) == Spectral::Basis::ZernikeB2) {
      if constexpr (std::is_same_v<ValueType, double>) {
        cylinder_apply(logical_du, partial_u_wrt_eta_or_zeta,
                       u_eta_or_zeta_fastest, u, mesh);
      } else {
        ERROR(
            "Support for complex numbers with the cylinder is not yet "
            "implemented for logical_partial_derivative.");
      }
    } else {
      static_assert(
          Variables<DerivativeTags>::number_of_independent_components <=
              Variables<T>::number_of_independent_components,
          "Temporary buffer in logical partial derivatives is too small");
      auto& logical_partial_derivatives_of_u = *logical_du;
      const Matrix& differentiation_matrix_xi =
          Spectral::differentiation_matrix(mesh.slice_through(0));
      const size_t deriv_size =
          Variables<DerivativeTags>::number_of_independent_components *
          u.number_of_grid_points();
      const size_t num_components_times_xi_slices =
          deriv_size / mesh.extents(0);
      apply_matrix_in_first_dim(logical_partial_derivatives_of_u[0], u.data(),
                                differentiation_matrix_xi, deriv_size);
      transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
          make_not_null(u_eta_or_zeta_fastest), u, mesh.extents(0),
          num_components_times_xi_slices);
      const Matrix& differentiation_matrix_eta =
          Spectral::differentiation_matrix(mesh.slice_through(1));
      apply_matrix_in_first_dim(partial_u_wrt_eta_or_zeta->data(),
                                u_eta_or_zeta_fastest->data(),
                                differentiation_matrix_eta, deriv_size);
      raw_transpose(make_not_null(logical_partial_derivatives_of_u[1]),
                    partial_u_wrt_eta_or_zeta->data(),
                    num_components_times_xi_slices, mesh.extents(0));

      const size_t chunk_size = mesh.extents(0) * mesh.extents(1);
      const size_t number_of_chunks = deriv_size / chunk_size;
      transpose(make_not_null(u_eta_or_zeta_fastest), u, chunk_size,
                number_of_chunks);
      const Matrix& differentiation_matrix_zeta =
          Spectral::differentiation_matrix(mesh.slice_through(2));
      apply_matrix_in_first_dim(partial_u_wrt_eta_or_zeta->data(),
                                u_eta_or_zeta_fastest->data(),
                                differentiation_matrix_zeta, deriv_size);
      raw_transpose(make_not_null(logical_partial_derivatives_of_u[2]),
                    partial_u_wrt_eta_or_zeta->data(), number_of_chunks,
                    chunk_size);
    }
  }

  static void spherical_apply(
      const gsl::not_null<std::array<double*, Dim>*> logical_du,
      const Variables<VariableTags>& u, const Mesh<3>& mesh) {
    auto& logical_partial_derivatives_of_u = *logical_du;
    const Matrix& differentiation_matrix_xi =
        Spectral::differentiation_matrix(mesh.slice_through(0));
    const size_t deriv_size =
        Variables<DerivativeTags>::number_of_independent_components *
        u.number_of_grid_points();
    apply_matrix_in_first_dim(logical_partial_derivatives_of_u[0], u.data(),
                              differentiation_matrix_xi, deriv_size);
    const auto& ylm = ylm::get_spherepack_cache(mesh.extents(1) - 1);
    for (size_t n = 0;
         n < Variables<DerivativeTags>::number_of_independent_components; ++n) {
      const size_t offset = n * u.number_of_grid_points();
      const auto du = std::array{logical_partial_derivatives_of_u[1] + offset,
                                 logical_partial_derivatives_of_u[2] + offset};
      ylm.gradient_all_offsets(du, make_not_null(u.data() + offset),
                               mesh.extents(0));
    }
  }

  // Angular derivatives are computed via SPHEREPACK's gradient, identical to
  // the spherical-shell case.
  //
  // Radial derivative: functions on the ball with angular degree l behave as
  // r^l near the origin, so their radial profile has definite parity based on
  // l. We therefore:
  //   1. Decompose into spherical-harmonic spectral space (per radial shell).
  //   2. For each (l,m) mode apply D_even or D_odd depending on parity of l.
  //   3. Synthesize back to physical space.
  // Step 2 copies the components into definite-parity buffers such that the
  // differentiation matrices can be called with two BLAS calls.
  template <typename T>
  static void filled_sphere_apply(
      const gsl::not_null<std::array<double*, Dim>*> logical_du,
      Variables<T>* const partial_u_wrt_eta_or_zeta,
      Variables<DerivativeTags>* const u_eta_or_zeta_fastest,
      const Variables<VariableTags>& u, const Mesh<Dim>& mesh) {
    static_assert(Dim == 3);
    ASSERT(mesh.basis() == make_array<3>(Spectral::Basis::ZernikeB3),
           "filled_sphere_apply requires ZernikeB3 bases ZernikeB3, got "
               << mesh.basis());

    const size_t n_r = mesh.extents(0);
    const size_t l_max = mesh.extents(1) - 1;
    ASSERT(l_max < 2 * n_r - 1,
           "ZernikeB3 radial resolution is insufficient for the requested "
           "angular resolution. Need l_max < 2*n_r-1, but l_max="
               << l_max << " and n_r=" << n_r);
    const auto& ylm = ylm::get_spherepack_cache(l_max);
    const size_t n_spectral = ylm.spectral_size();
    const size_t num_grid_points = u.number_of_grid_points();

    constexpr size_t n_components =
        Variables<DerivativeTags>::number_of_independent_components;

    // Angular derivatives; same as spherical_apply()
    for (size_t n = 0; n < n_components; ++n) {
      const size_t offset = n * num_grid_points;
      const auto du_ang =
          std::array{(*logical_du)[1] + offset, (*logical_du)[2] + offset};
      ylm.gradient_all_offsets(du_ang, make_not_null(u.data() + offset), n_r);
    }

    // Radial derivative via parity-aware differentiation
    const Matrix& D_even =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB3,
                                         Spectral::Quadrature::GaussRadauUpper>(
            n_r, Spectral::Parity::Even);
    const Matrix& D_odd =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB3,
                                         Spectral::Quadrature::GaussRadauUpper>(
            n_r, Spectral::Parity::Odd);

    // n_spectral is the SPHEREPACK buffer size = 2*(l_max+1)^2, including
    // padding. Valid modes total (l_max+1)^2; get_spherepack_cache always
    // sets m_max = l_max so each degree l contributes 2l+1 valid modes.
    const size_t n_valid_modes = (l_max + 1) * (l_max + 1);
    const size_t n_even_modes = (l_max % 2 == 0) ? (l_max + 1) * (l_max + 2) / 2
                                                 : l_max * (l_max + 1) / 2;
    const size_t n_odd_modes = n_valid_modes - n_even_modes;

    // Five scratch segments needed. Buffer layout:
    //   A: spec_buf | even_data | odd_data   [3*(l_max+1)^2 * n_r * n_comp]
    //   B: even_result | odd_result          [  (l_max+1)^2 * n_r * n_comp]
    //
    // At the partial_derivatives() call site, A = *du (3 * n_ylm_phys * n_r *
    // n_comp), which always satisfies the size requirement. At the
    // logical_partial_derivatives() call site, A is explicitly over-allocated
    // to 3 * (l_max + 1)^2 * n_r * n_comp when ZernikeB3.
    const size_t spec_total = n_components * n_spectral * n_r;
    const size_t even_total = n_even_modes * n_components * n_r;
    const size_t odd_total = n_odd_modes * n_components * n_r;

    double* const spec_buf = partial_u_wrt_eta_or_zeta->data();
    double* const even_data = spec_buf + spec_total;  // NOLINT
    double* const odd_data = even_data + even_total;  // NOLINT
    double* const even_result = u_eta_or_zeta_fastest->data();
    double* const odd_result = even_result + even_total;  // NOLINT

    // SH analysis: physical -> spectral (all components)
    for (size_t n = 0; n < n_components; ++n) {
      ylm.phys_to_spec_all_offsets(
          make_not_null(spec_buf + n * n_spectral * n_r),  // NOLINT
          make_not_null(u.data() + n * num_grid_points), n_r);
    }

    // Gather even-l and odd-l radial profiles into compact buffers.
    // For mode at SPHEREPACK offset s and component n, the n_r values are
    // contiguous at spec_buf[n*n_spectral*n_r + s*n_r].
    ylm::SpherepackIterator spec_iter{l_max, ylm.m_max()};
    for (size_t n = 0; n < n_components; ++n) {
      const double* spec = spec_buf + n * n_spectral * n_r;    // NOLINT
      double* even_dest = even_data + n * n_even_modes * n_r;  // NOLINT
      double* odd_dest = odd_data + n * n_odd_modes * n_r;     // NOLINT
      size_t even_col = 0;
      size_t odd_col = 0;
      spec_iter.reset();
      while (spec_iter) {
        const size_t s = spec_iter();
        if (spec_iter.l() % 2 == 0) {
          std::copy(spec + s * n_r, spec + (s + 1) * n_r,  // NOLINT
                    even_dest + even_col * n_r);           // NOLINT
          ++even_col;
        } else {
          std::copy(spec + s * n_r, spec + (s + 1) * n_r,  // NOLINT
                    odd_dest + odd_col * n_r);             // NOLINT
          ++odd_col;
        }
        ++spec_iter;
      }
    }

    // Apply radial differentiation matrices, treating
    // n_components * n_{even,odd}_modes as the column count.
    if (n_even_modes > 0) {
      apply_matrix_in_first_dim(even_result, even_data, D_even, even_total);
    }
    if (n_odd_modes > 0) {
      apply_matrix_in_first_dim(odd_result, odd_data, D_odd, odd_total);
    }

    // Copy differentiated radial profiles back to spectral buffer
    for (size_t n = 0; n < n_components; ++n) {
      double* spec = spec_buf + n * n_spectral * n_r;                 // NOLINT
      const double* even_src = even_result + n * n_even_modes * n_r;  // NOLINT
      const double* odd_src = odd_result + n * n_odd_modes * n_r;     // NOLINT
      size_t even_col = 0;
      size_t odd_col = 0;
      spec_iter.reset();
      while (spec_iter) {
        const size_t s = spec_iter();
        if (spec_iter.l() % 2 == 0) {
          std::copy(even_src + even_col * n_r,        // NOLINT
                    even_src + (even_col + 1) * n_r,  // NOLINT
                    spec + s * n_r);                  // NOLINT
          ++even_col;
        } else {
          std::copy(odd_src + odd_col * n_r,        // NOLINT
                    odd_src + (odd_col + 1) * n_r,  // NOLINT
                    spec + s * n_r);                // NOLINT
          ++odd_col;
        }
        ++spec_iter;
      }
    }

    // SH spectral -> physical
    for (size_t n = 0; n < n_components; ++n) {
      ylm.spec_to_phys_all_offsets(
          make_not_null((*logical_du)[0] + n * num_grid_points),
          make_not_null(spec_buf + n * n_spectral * n_r), n_r);  // NOLINT
    }
  }

  template <typename T, typename ValueType = typename Variables<T>::value_type>
  static void cylinder_apply(
      const gsl::not_null<std::array<ValueType*, Dim>*> logical_du,
      Variables<T>* const partial_u_wrt_eta_or_zeta,
      Variables<DerivativeTags>* const u_eta_or_zeta_fastest,
      const Variables<VariableTags>& u, const Mesh<3>& mesh) {
    ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
               mesh.basis(1) == Spectral::Basis::ZernikeB2,
           "Unexpected basis combination: " << mesh.basis());
    const auto [n_r, n_phi, n_zeta] = mesh.extents().indices();
    ASSERT(n_phi % 2 == 1,
           "Fourier with an even number of grid points can be unstable due to "
           "the top derivative not being representable");
    ASSERT(n_phi / 2 <= 2 * n_r - 2,
           "Zernike & Fourier on a disk have angular resolution limited by "
           "extents in both dimensions. We choose to enforce the restriction "
           "that the Fourier modal space is not larger than the Zernike "
           "angular capabilities (which would waste space and be physically "
           "ill-motivated).\nn_phi / 2 = "
               << n_phi / 2 << ", Maximum from Zernike = " << 2 * n_r - 2);

    const Matrix& diff_matrix_r_even =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB2,
                                         Spectral::Quadrature::GaussRadauUpper>(
            n_r, Spectral::Parity::Even);
    const Matrix& diff_matrix_r_odd =
        Spectral::differentiation_matrix<Spectral::Basis::ZernikeB2,
                                         Spectral::Quadrature::GaussRadauUpper>(
            n_r, Spectral::Parity::Odd);
    const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
    const Matrix& differentiation_matrix_zeta =
        Spectral::differentiation_matrix(mesh.slice_through(2));

    const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
    const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

    const size_t local_buffer_size =
        Variables<DerivativeTags>::number_of_independent_components *
        u.number_of_grid_points();
    // These just point to the passed buffers but it gets confusing to read when
    // they are used for radial purposes that don't match their names
    Variables<DerivativeTags> local_buffer1{};
    Variables<DerivativeTags> local_buffer2{};
    local_buffer1.set_data_ref(partial_u_wrt_eta_or_zeta->data(),
                               local_buffer_size);
    local_buffer2.set_data_ref(u_eta_or_zeta_fastest->data(),
                               local_buffer_size);

    const size_t deriv_size =
        Variables<DerivativeTags>::number_of_independent_components *
        u.number_of_grid_points();
    const size_t num_components_times_xi_slices = deriv_size / mesh.extents(0);

    // phi derivative
    transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
        make_not_null(u_eta_or_zeta_fastest), u, mesh.extents(0),
        num_components_times_xi_slices);
    apply_matrix_in_first_dim(partial_u_wrt_eta_or_zeta->data(),
                              u_eta_or_zeta_fastest->data(), diff_matrix_phi,
                              deriv_size);
    raw_transpose(make_not_null((*logical_du)[1]),
                  partial_u_wrt_eta_or_zeta->data(),
                  num_components_times_xi_slices, mesh.extents(0));

    // See comments in disk_apply() for why the following choices were made

    // reuse transposed data to go to angular spectral space
    apply_matrix_in_first_dim((*logical_du)[0], u_eta_or_zeta_fastest->data(),
                              nodal_to_modal_phi, deriv_size);
    // (*logical_du)[0] holds transposed angular modal rep

    raw_transpose(make_not_null(local_buffer1.data()), (*logical_du)[0],
                  num_components_times_xi_slices, mesh.extents(0));
    std::copy(local_buffer1.data(), local_buffer1.data() + local_buffer_size,
              local_buffer2.data());
    // local_buffer 1 and local_buffer2 hold angular modal rep

    // Copy even/odd subsets to compact regions for 3D cylinder
    const size_t n_even_cols =
        (1 + 2 * ((n_phi - 1) / 4)) *
        Variables<DerivativeTags>::number_of_independent_components;
    const size_t n_odd_cols =
        (2 * ((n_phi + 1) / 4)) *
        Variables<DerivativeTags>::number_of_independent_components;

    // Use local_buffer2 for compact storage
    const size_t total_even_size = n_even_cols * n_r * n_zeta;
    const size_t total_odd_size = n_odd_cols * n_r * n_zeta;

    ValueType* even_data = local_buffer2.data();
    ValueType* odd_data = local_buffer2.data() + total_even_size;

    size_t even_col = 0;
    size_t odd_col = 0;
    for (size_t i = 0;
         i < Variables<DerivativeTags>::number_of_independent_components *
                 n_phi * n_zeta;
         ++i) {
      if (i % n_phi == 0) {
        // m = 0 (even) - copy single column
        std::copy(local_buffer1.data() + i * n_r,
                  local_buffer1.data() + (i + 1) * n_r,
                  even_data + even_col * n_r);
        ++even_col;
      } else if ((i % n_phi - 1) / 2 % 2 == 1) {
        // Even m with cos/sin pair - copy two columns
        std::copy(local_buffer1.data() + i * n_r,
                  local_buffer1.data() + (i + 2) * n_r,
                  even_data + even_col * n_r);
        even_col += 2;
        ++i;
      } else {
        // Odd m with cos/sin pair - copy two columns
        std::copy(local_buffer1.data() + i * n_r,
                  local_buffer1.data() + (i + 2) * n_r,
                  odd_data + odd_col * n_r);
        odd_col += 2;
        ++i;
      }
    }

    // Use local_buffer1 for temporary storage since it's no longer needed
    ValueType* even_result = local_buffer1.data();
    ValueType* odd_result = local_buffer1.data() + total_even_size;

    apply_matrix_in_first_dim(even_result, even_data, diff_matrix_r_even,
                              total_even_size, false);

    // Copy even derivative components back to interleaved regions
    even_col = 0;
    for (size_t i = 0;
         i < Variables<DerivativeTags>::number_of_independent_components *
                 n_phi * n_zeta;
         ++i) {
      if (i % n_phi == 0) {
        // m = 0 (even) - copy single column back
        std::copy(even_result + even_col * n_r,
                  even_result + (even_col + 1) * n_r,
                  (*logical_du)[0] + i * n_r);
        ++even_col;
      } else if ((i % n_phi - 1) / 2 % 2 == 1) {
        // Even m with cos/sin pair - copy two columns back
        std::copy(even_result + even_col * n_r,
                  even_result + (even_col + 2) * n_r,
                  (*logical_du)[0] + i * n_r);
        even_col += 2;
        ++i;
      } else {
        ++i;
      }
    }

    apply_matrix_in_first_dim(odd_result, odd_data, diff_matrix_r_odd,
                              total_odd_size, false);

    // Copy odd derivative components back to interleaved regions
    odd_col = 0;
    for (size_t i = 0;
         i < Variables<DerivativeTags>::number_of_independent_components *
                 n_phi * n_zeta;
         ++i) {
      if (i % n_phi == 0) {
        // m = 0 (even) - skip
      } else if ((i % n_phi - 1) / 2 % 2 == 1) {
        // Even m - skip
        ++i;
      } else {
        // Odd m with cos/sin pair - copy both columns
        std::copy(odd_result + odd_col * n_r, odd_result + (odd_col + 2) * n_r,
                  (*logical_du)[0] + i * n_r);
        odd_col += 2;
        ++i;
      }
    }
    // (*logical_du)[0] holds the radial derivative in angular modal rep

    raw_transpose(make_not_null(local_buffer1.data()), (*logical_du)[0],
                  mesh.extents(0), num_components_times_xi_slices);
    // local_buffer1 holds transposed radial derivative in angular modal rep

    apply_matrix_in_first_dim(local_buffer2.data(), local_buffer1.data(),
                              modal_to_nodal_phi, deriv_size);
    // local_buffer2 holds transposed radial derivative in angular nodal rep

    raw_transpose(make_not_null((*logical_du)[0]), local_buffer2.data(),
                  num_components_times_xi_slices, mesh.extents(0));

    // zeta derivative
    const size_t chunk_size = mesh.extents(0) * mesh.extents(1);
    const size_t number_of_chunks = deriv_size / chunk_size;
    transpose(make_not_null(u_eta_or_zeta_fastest), u, chunk_size,
              number_of_chunks);
    apply_matrix_in_first_dim(partial_u_wrt_eta_or_zeta->data(),
                              u_eta_or_zeta_fastest->data(),
                              differentiation_matrix_zeta, deriv_size);
    raw_transpose(make_not_null((*logical_du)[2]),
                  partial_u_wrt_eta_or_zeta->data(), number_of_chunks,
                  chunk_size);
  }
};
}  // namespace partial_derivatives_detail

// Macro to explicitly instantiate partial_derivatives()
// for a given system of equations
#define INSTANTIATE_PARTIAL_DERIVATIVES_WITH_SYSTEM(SYSTEM, DIM,             \
                                                    DERIVATIVE_FRAME)        \
  template void partial_derivatives(                                         \
      gsl::not_null<Variables<                                               \
          db::wrap_tags_in<Tags::deriv, typename SYSTEM::gradient_variables, \
                           tmpl::size_t<DIM>, DERIVATIVE_FRAME>>*>           \
          du,                                                                \
      const std::array<Variables<typename SYSTEM::gradient_variables>, DIM>& \
          logical_partial_derivatives_of_u,                                  \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,          \
                            DERIVATIVE_FRAME>& inverse_jacobian);            \
  template void partial_derivatives(                                         \
      gsl::not_null<Variables<                                               \
          db::wrap_tags_in<Tags::deriv, typename SYSTEM::gradient_variables, \
                           tmpl::size_t<DIM>, DERIVATIVE_FRAME>>*>           \
          du,                                                                \
      const Variables<typename SYSTEM::gradient_variables>& u,               \
      const Mesh<DIM>& mesh,                                                 \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,          \
                            DERIVATIVE_FRAME>& inverse_jacobian);            \
  template Variables<                                                        \
      db::wrap_tags_in<Tags::deriv, typename SYSTEM::gradient_variables,     \
                       tmpl::size_t<DIM>, DERIVATIVE_FRAME>>                 \
  partial_derivatives<typename SYSTEM::gradient_variables>(                  \
      const Variables<typename SYSTEM::gradient_variables>& u,               \
      const Mesh<DIM>& mesh,                                                 \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,          \
                            DERIVATIVE_FRAME>& inverse_jacobian);

// Macro to explicitly instantiate cartoon_partial_derivatives()
// for a given system of equations (distinct from non-cartoon call)
#define INSTANTIATE_CARTOON_PARTIAL_DERIVATIVES_WITH_SYSTEM(SYSTEM, DIM,      \
                                                            DERIVATIVE_FRAME) \
  template void partial_derivatives(                                          \
      gsl::not_null<Variables<                                                \
          db::wrap_tags_in<Tags::deriv, typename SYSTEM::gradient_variables,  \
                           tmpl::size_t<DIM>, DERIVATIVE_FRAME>>*>            \
          du,                                                                 \
      const Variables<typename SYSTEM::gradient_variables>& u,                \
      const Mesh<DIM>& mesh,                                                  \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,           \
                            DERIVATIVE_FRAME>& inverse_jacobian_3d,           \
      const tnsr::I<DataVector, DIM, Frame::Inertial>& inertial_coords);

// This instantiates the chooser and underlying cartoon_partial_derivative()
#define INSTANTIATE_CARTOON_PARTIAL_DERIVATIVE(DIM, DERIVATIVE_FRAME, TENSOR) \
  template void partial_derivative(                                           \
      const gsl::not_null<TensorMetafunctions::prepend_spatial_index<         \
          TENSOR<DataVector, DIM, DERIVATIVE_FRAME>, DIM, UpLo::Lo,           \
          DERIVATIVE_FRAME>*>                                                 \
          du,                                                                 \
      const TENSOR<DataVector, DIM, DERIVATIVE_FRAME>& u,                     \
      const Mesh<DIM>& mesh,                                                  \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,           \
                            DERIVATIVE_FRAME>& inverse_jacobian,              \
      const tnsr::I<DataVector, DIM, Frame::Inertial>& inertial_coords);      \
  template TensorMetafunctions::prepend_spatial_index<                        \
      TENSOR<DataVector, DIM, DERIVATIVE_FRAME>, DIM, UpLo::Lo,               \
      DERIVATIVE_FRAME>                                                       \
  partial_derivative(                                                         \
      const TENSOR<DataVector, DIM, DERIVATIVE_FRAME>& u,                     \
      const Mesh<DIM>& mesh,                                                  \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,           \
                            DERIVATIVE_FRAME>& inverse_jacobian,              \
      const tnsr::I<DataVector, DIM, Frame::Inertial>& inertial_coords);
