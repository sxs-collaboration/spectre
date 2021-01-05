// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/TagsTimeDependent.hpp"

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::Tags {
template <size_t Dim>
void InertialFromGridCoordinatesCompute<Dim>::function(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        target_coords,
    const tnsr::I<DataVector, Dim, Frame::Grid>& source_coords,
    const std::optional<std::tuple<
        tnsr::I<DataVector, Dim, Frame::Inertial>,
        ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
        ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
        tnsr::I<DataVector, Dim, Frame::Inertial>>>&
        grid_to_inertial_quantities) noexcept {
  if (not grid_to_inertial_quantities) {
    // We use a const_cast to point the data into the existing allocation
    // inside `source_coords` to avoid copying. This is safe because the output
    // of a compute tag is immutable.
    for (size_t i = 0; i < Dim; ++i) {
      target_coords->get(i).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          &const_cast<DataVector&>(source_coords.get(i)));
    }
  } else {
    // We use a const_cast to point the data into the existing allocation
    // inside `grid_to_inertial_quantities` to avoid copying. This is
    // effectively unpacking the tuple. This is safe because the output
    // of a compute tag is immutable.
    for (size_t i = 0; i < Dim; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      target_coords->get(i).set_data_ref(&const_cast<DataVector&>(
          std::get<0>(*grid_to_inertial_quantities).get(i)));
    }
  }
}

template <size_t Dim>
void ElementToInertialInverseJacobian<Dim>::function(
    const gsl::not_null<
        ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        inv_jac_logical_to_inertial,
    const ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Grid>&
        inv_jac_logical_to_grid,
    const std::optional<std::tuple<
        tnsr::I<DataVector, Dim, Frame::Inertial>,
        ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
        ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
        tnsr::I<DataVector, Dim, Frame::Inertial>>>&
        grid_to_inertial_quantities) noexcept {
  if (not grid_to_inertial_quantities) {
    // We use a const_cast to point the data into the existing allocation
    // inside `inv_jac_logical_to_grid` to avoid copying. This is safe because
    // the output of a compute tag is immutable.
    for (size_t i = 0; i < inv_jac_logical_to_inertial->size(); ++i) {
      inv_jac_logical_to_inertial->operator[](i).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          &const_cast<DataVector&>(inv_jac_logical_to_grid[i]));
    }
  } else {
    const auto& inv_jac_grid_to_inertial =
        std::get<1>(*grid_to_inertial_quantities);
    for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < Dim; ++inertial_i) {
        inv_jac_logical_to_inertial->get(logical_i, inertial_i) =
            inv_jac_logical_to_grid.get(logical_i, 0) *
            inv_jac_grid_to_inertial.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < Dim; ++grid_i) {
          inv_jac_logical_to_inertial->get(logical_i, inertial_i) +=
              inv_jac_logical_to_grid.get(logical_i, grid_i) *
              inv_jac_grid_to_inertial.get(grid_i, inertial_i);
        }
      }
    }
  }
}

template <size_t Dim>
void InertialMeshVelocityCompute<Dim>::function(
    const gsl::not_null<return_type*> mesh_velocity,
    const std::optional<std::tuple<
        tnsr::I<DataVector, Dim, Frame::Inertial>,
        ::InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
        ::Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>,
        tnsr::I<DataVector, Dim, Frame::Inertial>>>&
        grid_to_inertial_quantities) noexcept {
  if (not grid_to_inertial_quantities) {
    *mesh_velocity = std::nullopt;
  } else {
    if (not*mesh_velocity) {
      *mesh_velocity = typename return_type::value_type{};
    }
    // We use a const_cast to point the data into the existing allocation
    // inside `grid_to_inertial_quantities` to avoid copying. This is
    // effectively unpacking the tuple. This is safe because the output
    // of a compute tag is immutable.
    for (size_t i = 0; i < Dim; ++i) {
      mesh_velocity->value().operator[](i).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          &const_cast<DataVector&>(
              std::get<3>(*grid_to_inertial_quantities).get(i)));
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                     \
  template struct InertialFromGridCoordinatesCompute<DIM(data)>; \
  template struct ElementToInertialInverseJacobian<DIM(data)>;   \
  template struct InertialMeshVelocityCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace domain::Tags
