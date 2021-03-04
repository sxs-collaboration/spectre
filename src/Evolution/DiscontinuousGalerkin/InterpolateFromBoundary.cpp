// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InterpolateFromBoundary.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::detail {
namespace {
// We use a separate function in the  xi direction to avoid the expensive
// SliceIterator
void interpolate_dt_terms_gauss_points_impl_xi_dir(
    const gsl::not_null<double*> volume_dt_vars,
    const size_t num_independent_components, const size_t num_volume_pts,
    const size_t num_boundary_pts,
    const gsl::span<const double>& dt_corrections,
    const DataVector& boundary_interpolation_term) noexcept {
  DataVector volume_dt_vars_view{};
  for (size_t component_index = 0; component_index < num_independent_components;
       ++component_index) {
    const size_t stripe_size = num_volume_pts / num_boundary_pts;
    for (size_t boundary_index = 0; boundary_index < num_boundary_pts;
         ++boundary_index) {
      volume_dt_vars_view.set_data_ref(volume_dt_vars.get() +
                                           component_index * num_volume_pts +
                                           boundary_index * stripe_size,
                                       stripe_size);
      volume_dt_vars_view +=
          boundary_interpolation_term *
          dt_corrections[component_index * num_boundary_pts + boundary_index];
    }
  }
}
}  // namespace

template <size_t Dim>
void interpolate_dt_terms_gauss_points_impl(
    const gsl::not_null<double*> volume_dt_vars,
    const size_t num_independent_components, const Mesh<Dim>& volume_mesh,
    const size_t dimension, const size_t num_boundary_pts,
    const gsl::span<const double>& dt_corrections,
    const DataVector& boundary_interpolation_term) noexcept {
  const size_t num_volume_pts = volume_mesh.number_of_grid_points();
  if (dimension == 0) {
    interpolate_dt_terms_gauss_points_impl_xi_dir(
        volume_dt_vars, num_independent_components, num_volume_pts,
        num_boundary_pts, dt_corrections, boundary_interpolation_term);
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
        volume_dt_vars.get()[component_index * num_volume_pts + volume_index] +=
            boundary_interpolation_term[index_on_stripe] *
            dt_corrections[component_index * num_boundary_pts + boundary_index];
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                                 \
  template void interpolate_dt_terms_gauss_points_impl(                      \
      gsl::not_null<double*> volume_dt_vars,                                 \
      size_t num_independent_components, const Mesh<DIM(data)>& volume_mesh, \
      size_t dimension, size_t num_boundary_pts,                             \
      const gsl::span<const double>& dt_corrections,                         \
      const DataVector& boundary_interpolation_term) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::detail
