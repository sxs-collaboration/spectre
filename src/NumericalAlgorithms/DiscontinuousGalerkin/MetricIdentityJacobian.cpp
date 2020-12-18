// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"

#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t Dim>
void metric_identity_jacobian_impl(
    const gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept {
  static_assert(Dim == 1 or Dim == 2, "Generic impl handles only 1d and 2d.");
  destructive_resize_components(det_jac_times_inverse_jacobian,
                                mesh.number_of_grid_points());
  if constexpr (Dim == 1) {
    (void)jacobian;
    get<0, 0>(*det_jac_times_inverse_jacobian) = 1.0;
  } else if constexpr (Dim == 2) {
    (void)jacobian;
    const Mesh<1>& mesh0 = mesh.slice_through(0);
    const Mesh<1>& mesh1 = mesh.slice_through(1);
    const Matrix identity{};
    auto diff_matrices = make_array<Dim>(std::cref(identity));

    diff_matrices[1] = Spectral::differentiation_matrix(mesh1);
    apply_matrices(make_not_null(&get<0, 0>(*det_jac_times_inverse_jacobian)),
                   diff_matrices, get<1>(inertial_coords), mesh.extents());

    apply_matrices(make_not_null(&get<0, 1>(*det_jac_times_inverse_jacobian)),
                   diff_matrices, get<0>(inertial_coords), mesh.extents());
    get<0, 1>(*det_jac_times_inverse_jacobian) *= -1.0;

    diff_matrices[1] = std::cref(identity);
    diff_matrices[0] = Spectral::differentiation_matrix(mesh0);
    apply_matrices(make_not_null(&get<1, 0>(*det_jac_times_inverse_jacobian)),
                   diff_matrices, get<1>(inertial_coords), mesh.extents());
    get<1, 0>(*det_jac_times_inverse_jacobian) *= -1.0;

    apply_matrices(make_not_null(&get<1, 1>(*det_jac_times_inverse_jacobian)),
                   diff_matrices, get<0>(inertial_coords), mesh.extents());
  }
}

void metric_identity_jacobian_impl(
    const gsl::not_null<
        InverseJacobian<DataVector, 3, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const gsl::not_null<DataVector*> buffer,
    const gsl::not_null<DataVector*> buffer_component, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, 3, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept {
  // The 3d case is handled separately because this actually requires a buffer.
  // Basically, in 2d you can get into the situation where you have 2 unused
  // components (and thus 2 buffers) and so taking the eta derivatives you have
  // plenty of space for transposing. In 3d all components of the Jacobian
  // depend on either eta or zeta derivatives and so you can't use one of them
  // as a buffer to do a transpose.
  //
  // The 3d implementation is what should be generalized to higher dimensions if
  // the need arises.
  destructive_resize_components(det_jac_times_inverse_jacobian,
                                mesh.number_of_grid_points());
  const Mesh<1>& mesh0 = mesh.slice_through(0);
  const Mesh<1>& mesh1 = mesh.slice_through(1);
  const Mesh<1>& mesh2 = mesh.slice_through(2);
  const Matrix identity{};
  auto diff_matrices = make_array<3>(std::cref(identity));

  ASSERT(buffer->size() == mesh.number_of_grid_points(),
         "The size of buffer must be " << mesh.number_of_grid_points()
                                       << " but is " << buffer->size());
  ASSERT(buffer_component->size() == mesh.number_of_grid_points(),
         "The size of buffer_component must be " << mesh.number_of_grid_points()
                                                 << " but is "
                                                 << buffer_component->size());

  // Note that we will multiply everything by 0.5 once we have all terms
  // contributing to each component computed.

  // Compute all terms with logical derivatives in xi direction
  diff_matrices[0] = Spectral::differentiation_matrix(mesh0);
  *buffer = get<2>(inertial_coords) * get<1, 2>(jacobian) -
            get<1>(inertial_coords) * get<2, 2>(jacobian);
  apply_matrices(make_not_null(&get<1, 0>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<1>(inertial_coords) * get<2, 1>(jacobian) -
            get<2>(inertial_coords) * get<1, 1>(jacobian);
  apply_matrices(make_not_null(&get<2, 0>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<0>(inertial_coords) * get<2, 2>(jacobian) -
            get<2>(inertial_coords) * get<0, 2>(jacobian);
  apply_matrices(make_not_null(&get<1, 1>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<2>(inertial_coords) * get<0, 1>(jacobian) -
            get<0>(inertial_coords) * get<2, 1>(jacobian);
  apply_matrices(make_not_null(&get<2, 1>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<1>(inertial_coords) * get<0, 2>(jacobian) -
            get<0>(inertial_coords) * get<1, 2>(jacobian);
  apply_matrices(make_not_null(&get<1, 2>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<0>(inertial_coords) * get<1, 1>(jacobian) -
            get<1>(inertial_coords) * get<0, 1>(jacobian);
  apply_matrices(make_not_null(&get<2, 2>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  // Compute all terms with logical derivatives in eta direction
  diff_matrices[0] = identity;
  diff_matrices[1] = Spectral::differentiation_matrix(mesh1);

  // First do terms where the derivative can be directly written into the
  // output buffer.
  *buffer = get<1>(inertial_coords) * get<2, 2>(jacobian) -
            get<2>(inertial_coords) * get<1, 2>(jacobian);
  apply_matrices(make_not_null(&get<0, 0>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<2>(inertial_coords) * get<0, 2>(jacobian) -
            get<0>(inertial_coords) * get<2, 2>(jacobian);
  apply_matrices(make_not_null(&get<0, 1>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  *buffer = get<0>(inertial_coords) * get<1, 2>(jacobian) -
            get<1>(inertial_coords) * get<0, 2>(jacobian);
  apply_matrices(make_not_null(&get<0, 2>(*det_jac_times_inverse_jacobian)),
                 diff_matrices, *buffer, mesh.extents());

  // Now do terms that need to be added to existing output.
  // These we can multiply by 0.5
  *buffer = get<2>(inertial_coords) * get<1, 0>(jacobian) -
            get<1>(inertial_coords) * get<2, 0>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<2, 0>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<2, 0>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<0>(inertial_coords) * get<2, 0>(jacobian) -
            get<2>(inertial_coords) * get<0, 0>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<2, 1>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<2, 1>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<1>(inertial_coords) * get<0, 0>(jacobian) -
            get<0>(inertial_coords) * get<1, 0>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<2, 2>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<2, 2>(*det_jac_times_inverse_jacobian) *= 0.5;

  // Compute all terms with logical derivatives in eta direction
  diff_matrices[1] = identity;
  diff_matrices[2] = Spectral::differentiation_matrix(mesh2);

  // All terms need to be added to existing output.
  // These we multiply by 0.5
  *buffer = get<2>(inertial_coords) * get<1, 1>(jacobian) -
            get<1>(inertial_coords) * get<2, 1>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<0, 0>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<0, 0>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<1>(inertial_coords) * get<2, 0>(jacobian) -
            get<2>(inertial_coords) * get<1, 0>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<1, 0>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<1, 0>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<0>(inertial_coords) * get<2, 1>(jacobian) -
            get<2>(inertial_coords) * get<0, 1>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<0, 1>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<0, 1>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<2>(inertial_coords) * get<0, 0>(jacobian) -
            get<0>(inertial_coords) * get<2, 0>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<1, 1>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<1, 1>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<1>(inertial_coords) * get<0, 1>(jacobian) -
            get<0>(inertial_coords) * get<1, 1>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<0, 2>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<0, 2>(*det_jac_times_inverse_jacobian) *= 0.5;

  *buffer = get<0>(inertial_coords) * get<1, 0>(jacobian) -
            get<1>(inertial_coords) * get<0, 0>(jacobian);
  apply_matrices(buffer_component, diff_matrices, *buffer, mesh.extents());
  get<1, 2>(*det_jac_times_inverse_jacobian) += *buffer_component;
  get<1, 2>(*det_jac_times_inverse_jacobian) *= 0.5;
}
}  // namespace

namespace dg {
template <size_t Dim>
void metric_identity_jacobian(
    const gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept {
  if constexpr (Dim == 3) {
    const size_t num_grid_points = mesh.number_of_grid_points();
    DataVector buffers{2 * num_grid_points};
    DataVector buffer{buffers.data(), num_grid_points};
    // clang-tidy: no pointer math
    DataVector buffer_component{buffers.data() + num_grid_points,  // NOLINT
                                num_grid_points};

    metric_identity_jacobian_impl(
        det_jac_times_inverse_jacobian, make_not_null(&buffer),
        make_not_null(&buffer_component), mesh, inertial_coords, jacobian);
  } else {
    metric_identity_jacobian_impl(det_jac_times_inverse_jacobian, mesh,
                                  inertial_coords, jacobian);
  }
}

template void metric_identity_jacobian(
    gsl::not_null<
        InverseJacobian<DataVector, 1, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const Mesh<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, 1, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept;
template void metric_identity_jacobian(
    gsl::not_null<
        InverseJacobian<DataVector, 2, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const Mesh<2>& mesh,
    const tnsr::I<DataVector, 2, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, 2, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept;
template void metric_identity_jacobian(
    gsl::not_null<
        InverseJacobian<DataVector, 3, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, 3, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept;
}  // namespace dg
