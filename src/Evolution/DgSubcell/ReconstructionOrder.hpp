// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
/// Copies the `reconstruction_order` into the DataBox tag
/// `evolution::dg::subcell::Tags::ReconstructionOrder` if
/// `reconstruction_order` has a value.
template <size_t Dim, typename DbTagsList>
void store_reconstruction_order_in_databox(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const std::optional<std::array<gsl::span<std::uint8_t>, Dim>>&
        reconstruction_order) {
  if (reconstruction_order.has_value()) {
    db::mutate<evolution::dg::subcell::Tags::ReconstructionOrder<Dim>>(
        [&reconstruction_order](const auto recons_order_ptr,
                                const ::Mesh<Dim>& subcell_mesh) {
          if (UNLIKELY(not recons_order_ptr->has_value())) {
            (*recons_order_ptr) = typename std::decay_t<
                decltype(*recons_order_ptr)>::value_type{};
          }
          destructive_resize_components(
              make_not_null(&(recons_order_ptr->value())),
              subcell_mesh.number_of_grid_points());
          for (size_t d = Dim - 1; d < Dim; --d) {
            // The data is always in d-varies fastest ordering
            //
            // We always copy into the `d=0` buffer, then do transposes later.
            for (size_t j = 0;
                 j < subcell_mesh.slice_away(d).number_of_grid_points(); ++j) {
              for (size_t i = 0; i < subcell_mesh.extents(d); ++i) {
                get<0>(recons_order_ptr
                           ->value())[j * subcell_mesh.extents(d) + i] =
                    gsl::at(reconstruction_order.value(),
                            d)[j * (subcell_mesh.extents(d) + 2) + i + 1];
              }
            }
            if (d == 1) {
              // Order is (y, z, x) -> (x, y, z)
              transpose(make_not_null(&recons_order_ptr->value().get(d)),
                        get<0>(recons_order_ptr->value()),
                        subcell_mesh.extents(1) *
                            (Dim > 2 ? subcell_mesh.extents(2) : 1),
                        subcell_mesh.number_of_grid_points() /
                            (subcell_mesh.extents(1) *
                             (Dim > 2 ? subcell_mesh.extents(2) : 1)));
            } else if (d == 2) {
              // Order is (z, x, y) -> (x, y, z)
              transpose(make_not_null(&recons_order_ptr->value().get(d)),
                        get<0>(recons_order_ptr->value()),
                        subcell_mesh.extents(2),
                        subcell_mesh.number_of_grid_points() /
                            subcell_mesh.extents(2));
            }
            // }
          }
        },
        box, db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box));
  }
}
}  // namespace evolution::dg::subcell
