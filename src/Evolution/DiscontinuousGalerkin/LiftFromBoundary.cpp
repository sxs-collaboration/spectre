// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/LiftFromBoundary.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::detail {
// We use a separate function in the  xi direction to avoid the expensive
// SliceIterator
void lift_boundary_terms_gauss_points_impl_xi_dir(
    const gsl::not_null<double*> volume_dt_vars,
    const size_t num_independent_components, const size_t num_volume_pts,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const double>& boundary_corrections,
    const DataVector& boundary_lifting_term,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& face_det_jacobian) noexcept {
  DataVector volume_dt_vars_view{};
  DataVector volume_inv_det_jacobian_view{};
  for (size_t component_index = 0; component_index < num_independent_components;
       ++component_index) {
    const size_t stripe_size = num_volume_pts / num_boundary_pts;
    for (size_t boundary_index = 0; boundary_index < num_boundary_pts;
         ++boundary_index) {
      volume_dt_vars_view.set_data_ref(volume_dt_vars.get() +
                                           component_index * num_volume_pts +
                                           boundary_index * stripe_size,
                                       stripe_size);
      // safe const_cast since used as a view
      volume_inv_det_jacobian_view.set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(get(volume_det_inv_jacobian).data()) +
              boundary_index * stripe_size,
          stripe_size);
      // Minus sign because we brought this from the LHS to the RHS.
      volume_dt_vars_view -=
          volume_inv_det_jacobian_view * boundary_lifting_term *
          get(face_det_jacobian)[boundary_index] *
          get(magnitude_of_face_normal)[boundary_index] *
          boundary_corrections[component_index * num_boundary_pts +
                               boundary_index];
    }
  }
}

// We use a separate function in the  xi direction to avoid the expensive
// SliceIterator
void lift_boundary_terms_gauss_points_impl_xi_dir(
    const gsl::not_null<double*> volume_dt_vars,
    const size_t num_independent_components, const size_t num_volume_pts,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const double>& upper_boundary_corrections,
    const DataVector& upper_boundary_lifting_term,
    const Scalar<DataVector>& upper_magnitude_of_face_normal,
    const Scalar<DataVector>& upper_face_det_jacobian,
    const gsl::span<const double>& lower_boundary_corrections,
    const DataVector& lower_boundary_lifting_term,
    const Scalar<DataVector>& lower_magnitude_of_face_normal,
    const Scalar<DataVector>& lower_face_det_jacobian) noexcept {
  DataVector volume_dt_vars_view{};
  DataVector volume_inv_det_jacobian_view{};
  for (size_t component_index = 0; component_index < num_independent_components;
       ++component_index) {
    const size_t stripe_size = num_volume_pts / num_boundary_pts;
    for (size_t boundary_index = 0; boundary_index < num_boundary_pts;
         ++boundary_index) {
      volume_dt_vars_view.set_data_ref(volume_dt_vars.get() +
                                           component_index * num_volume_pts +
                                           boundary_index * stripe_size,
                                       stripe_size);
      // safe const_cast since used as a view
      volume_inv_det_jacobian_view.set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(get(volume_det_inv_jacobian).data()) +
              boundary_index * stripe_size,
          stripe_size);
      // Minus sign because we brought this from the LHS to the RHS.
      volume_dt_vars_view -=
          volume_inv_det_jacobian_view *
          (upper_boundary_lifting_term *
               get(upper_face_det_jacobian)[boundary_index] *
               get(upper_magnitude_of_face_normal)[boundary_index] *
               upper_boundary_corrections[component_index * num_boundary_pts +
                                          boundary_index] +
           lower_boundary_lifting_term *
               get(lower_face_det_jacobian)[boundary_index] *
               get(lower_magnitude_of_face_normal)[boundary_index] *
               lower_boundary_corrections[component_index * num_boundary_pts +
                                          boundary_index]);
    }
  }
}

template <size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    const gsl::not_null<double*> volume_dt_vars,
    const size_t num_independent_components, const Mesh<Dim>& volume_mesh,
    const size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const double>& boundary_corrections,
    const DataVector& boundary_lifting_term,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& face_det_jacobian) noexcept {
  const size_t num_volume_pts = volume_mesh.number_of_grid_points();
  if (dimension == 0) {
    lift_boundary_terms_gauss_points_impl_xi_dir(
        volume_dt_vars, num_independent_components, num_volume_pts,
        volume_det_inv_jacobian, num_boundary_pts, boundary_corrections,
        boundary_lifting_term, magnitude_of_face_normal, face_det_jacobian);
    return;
  }

  // Developer note: A potential optimization is to re-order (not transpose!)
  // the volume time derivative for all variables before lifting, so that the
  // lifting can be done using vectorized math and DataVectors as views. It
  // would need to be tested whether that actually increases performance.
  // Another alternative would be to use a SIMD library, such as nsimd or xsimd.

  const size_t stripe_size = volume_mesh.extents(dimension);
  size_t boundary_index = 0;
  for (StripeIterator si{volume_mesh.extents(), dimension}; si;
       (void)++si, (void)++boundary_index) {
    // Loop over each stripe in this logical direction. This is effectively
    // looping over each boundary grid point.
    for (size_t component_index = 0;
         component_index < num_independent_components; ++component_index) {
      for (size_t index_on_stripe = 0; index_on_stripe < stripe_size;
           ++index_on_stripe) {
        const size_t volume_index = si.offset() + si.stride() * index_on_stripe;
        volume_dt_vars.get()[component_index * num_volume_pts + volume_index] -=
            get(volume_det_inv_jacobian)[volume_index] *
            boundary_lifting_term[index_on_stripe] *
            get(face_det_jacobian)[boundary_index] *
            get(magnitude_of_face_normal)[boundary_index] *
            boundary_corrections[component_index * num_boundary_pts +
                                 boundary_index];
      }
    }
  }
}

template <size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    const gsl::not_null<double*> volume_dt_vars,
    const size_t num_independent_components, const Mesh<Dim>& volume_mesh,
    const size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const double>& upper_boundary_corrections,
    const DataVector& upper_boundary_lifting_term,
    const Scalar<DataVector>& upper_magnitude_of_face_normal,
    const Scalar<DataVector>& upper_face_det_jacobian,
    const gsl::span<const double>& lower_boundary_corrections,
    const DataVector& lower_boundary_lifting_term,
    const Scalar<DataVector>& lower_magnitude_of_face_normal,
    const Scalar<DataVector>& lower_face_det_jacobian) noexcept {
  const size_t num_volume_pts = volume_mesh.number_of_grid_points();
  if (dimension == 0) {
    lift_boundary_terms_gauss_points_impl_xi_dir(
        volume_dt_vars, num_independent_components, num_volume_pts,
        volume_det_inv_jacobian, num_boundary_pts, upper_boundary_corrections,
        upper_boundary_lifting_term, upper_magnitude_of_face_normal,
        upper_face_det_jacobian, lower_boundary_corrections,
        lower_boundary_lifting_term, lower_magnitude_of_face_normal,
        lower_face_det_jacobian);
    return;
  }
  // Developer note: A potential optimization is to re-order (not transpose!)
  // the volume time derivative for all variables before lifting, so that the
  // lifting can be done using vectorized math and DataVectors as views. It
  // would need to be tested whether that actually increases performance.
  // Another alternative would be to use a SIMD library, such as nsimd or xsimd.

  const size_t stripe_size = volume_mesh.extents(dimension);
  size_t boundary_index = 0;
  for (StripeIterator si{volume_mesh.extents(), dimension}; si;
       (void)++si, (void)++boundary_index) {
    // Loop over each stripe in this logical direction. This is effectively
    // looping over each boundary grid point.
    for (size_t component_index = 0;
         component_index < num_independent_components; ++component_index) {
      for (size_t index_on_stripe = 0; index_on_stripe < stripe_size;
           ++index_on_stripe) {
        const size_t volume_index = si.offset() + si.stride() * index_on_stripe;
        volume_dt_vars.get()[component_index * num_volume_pts + volume_index] -=
            get(volume_det_inv_jacobian)[volume_index] *
            (upper_boundary_lifting_term[index_on_stripe] *
                 get(upper_face_det_jacobian)[boundary_index] *
                 get(upper_magnitude_of_face_normal)[boundary_index] *
                 upper_boundary_corrections[component_index * num_boundary_pts +
                                            boundary_index] +
             lower_boundary_lifting_term[index_on_stripe] *
                 get(lower_face_det_jacobian)[boundary_index] *
                 get(lower_magnitude_of_face_normal)[boundary_index] *
                 lower_boundary_corrections[component_index * num_boundary_pts +
                                            boundary_index]);
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                                 \
  template void lift_boundary_terms_gauss_points_impl(                       \
      gsl::not_null<double*> volume_dt_vars,                                 \
      size_t num_independent_components, const Mesh<DIM(data)>& volume_mesh, \
      size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,   \
      size_t num_boundary_pts,                                               \
      const gsl::span<const double>& boundary_corrections,                   \
      const DataVector& boundary_lifting_term,                               \
      const Scalar<DataVector>& magnitude_of_face_normal,                    \
      const Scalar<DataVector>& face_det_jacobian) noexcept;                 \
  template void lift_boundary_terms_gauss_points_impl(                       \
      gsl::not_null<double*> volume_dt_vars,                                 \
      size_t num_independent_components, const Mesh<DIM(data)>& volume_mesh, \
      size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,   \
      size_t num_boundary_pts,                                               \
      const gsl::span<const double>& upper_boundary_corrections,             \
      const DataVector& upper_boundary_lifting_term,                         \
      const Scalar<DataVector>& upper_magnitude_of_face_normal,              \
      const Scalar<DataVector>& upper_face_det_jacobian,                     \
      const gsl::span<const double>& lower_boundary_corrections,             \
      const DataVector& lower_boundary_lifting_term,                         \
      const Scalar<DataVector>& lower_magnitude_of_face_normal,              \
      const Scalar<DataVector>& lower_face_det_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::detail
