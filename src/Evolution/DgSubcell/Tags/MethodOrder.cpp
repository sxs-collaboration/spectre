// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/MethodOrder.hpp"

#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::Tags {
template <size_t Dim>
void MethodOrderCompute<Dim>::function(
    const gsl::not_null<return_type*> method_order, const ::Mesh<Dim>& dg_mesh,
    const ::Mesh<Dim>& subcell_mesh, const subcell::ActiveGrid active_grid,
    const std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>&
        reconstruction_order,
    const subcell::SubcellOptions& subcell_options) {
  if (not method_order->has_value()) {
    *method_order = typename return_type::value_type{};
  }
  if (active_grid == subcell::ActiveGrid::Dg) {
    destructive_resize_components(make_not_null(&method_order->value()),
                                  dg_mesh.number_of_grid_points());
    for (size_t i = 0; i < Dim; ++i) {
      method_order->value()[i] = dg_mesh.extents(i);
    }
  } else {
    destructive_resize_components(make_not_null(&method_order->value()),
                                  subcell_mesh.number_of_grid_points());
    if (reconstruction_order.has_value()) {
      *method_order = reconstruction_order;
    } else {
      if (UNLIKELY(static_cast<int>(
                       subcell_options.finite_difference_derivative_order()) <
                   0)) {
        ERROR(
            "Using adaptive order reconstruction but the reconstruction order "
            "was not set.");
      }
      for (size_t i = 0; i < Dim; ++i) {
        method_order->value()[i] = static_cast<int>(
            subcell_options.finite_difference_derivative_order());
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct MethodOrderCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::Tags
