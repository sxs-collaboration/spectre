// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::Tags {
/*!
 * \brief "Computes" the active coordinates by setting the `DataVector`s to
 * point into the coordinates of either the DG or subcell grid.
 */
template <size_t Dim, typename Fr>
struct ObserverCoordinatesCompute
    : db::ComputeTag,
      ::Events::Tags::ObserverCoordinates<Dim, Fr> {
  using base = ::Events::Tags::ObserverCoordinates<Dim, Fr>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<ActiveGrid, Coordinates<Dim, Fr>,
                                   ::domain::Tags::Coordinates<Dim, Fr>>;
  static void function(const gsl::not_null<return_type*> active_coords,
                       const subcell::ActiveGrid active_grid,
                       const tnsr::I<DataVector, Dim, Fr>& subcell_coords,
                       const tnsr::I<DataVector, Dim, Fr>& dg_coords) {
    const auto set_to_refs =
        [&active_coords](const tnsr::I<DataVector, Dim, Fr>& coords) {
          for (size_t i = 0; i < Dim; ++i) {
            active_coords->get(i).set_data_ref(
                make_not_null(&const_cast<DataVector&>(coords.get(i))));
          }
        };
    if (active_grid == subcell::ActiveGrid::Dg) {
      set_to_refs(dg_coords);
    } else {
      ASSERT(active_grid == subcell::ActiveGrid::Subcell,
             "ActiveGrid should be subcell if it isn't DG. Maybe an extra enum "
             "entry was added?");
      set_to_refs(subcell_coords);
    }
  }
};
}  // namespace evolution::dg::subcell::Tags
