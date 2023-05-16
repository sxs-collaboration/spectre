// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/TciOnDgGrid.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::subcell {

std::tuple<int, evolution::dg::subcell::RdmpTciData> TciOnDgGrid::apply(
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_q, const Mesh<3>& dg_mesh,
    const Mesh<3>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const double persson_exponent, bool /*element_stays_on_dg*/) {
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};

  const size_t num_dg_pts = dg_mesh.number_of_grid_points();
  const size_t num_subcell_pts = subcell_mesh.number_of_grid_points();

  DataVector temp_buffer{2 * num_dg_pts + 3 * num_subcell_pts};
  size_t offset_into_temp_buffer = 0;
  const auto assign_data =
      [&temp_buffer, &offset_into_temp_buffer](
          const gsl::not_null<Scalar<DataVector>*> to_assign,
          const size_t size) {
        ASSERT(offset_into_temp_buffer + size <= temp_buffer.size(),
               "Trying to assign data out of allocated memory size");
        get(*to_assign)
            .set_data_ref(temp_buffer.data() + offset_into_temp_buffer, size);
        offset_into_temp_buffer += size;
      };

  // Note : for RDMP TCI data, we compute mag(E), mag(B) on DG grid and then
  // project their magnitude to FD grid, NOT projecting E^i and B^i on FD grid
  // then compute magnitudes of them.

  Scalar<DataVector> dg_mag_tilde_e{};
  Scalar<DataVector> subcell_mag_tilde_e{};
  assign_data(make_not_null(&dg_mag_tilde_e), num_dg_pts);
  magnitude(make_not_null(&dg_mag_tilde_e), tilde_e);
  assign_data(make_not_null(&subcell_mag_tilde_e), num_subcell_pts);
  evolution::dg::subcell::fd::project(make_not_null(&get(subcell_mag_tilde_e)),
                                      get(dg_mag_tilde_e), dg_mesh,
                                      subcell_mesh.extents());

  Scalar<DataVector> dg_mag_tilde_b{};
  Scalar<DataVector> subcell_mag_tilde_b{};
  assign_data(make_not_null(&dg_mag_tilde_b), num_dg_pts);
  magnitude(make_not_null(&dg_mag_tilde_b), tilde_b);
  assign_data(make_not_null(&subcell_mag_tilde_b), num_subcell_pts);
  evolution::dg::subcell::fd::project(make_not_null(&get(subcell_mag_tilde_b)),
                                      get(dg_mag_tilde_b), dg_mesh,
                                      subcell_mesh.extents());

  Scalar<DataVector> subcell_tilde_q{};
  assign_data(make_not_null(&subcell_tilde_q), num_subcell_pts);
  evolution::dg::subcell::fd::project(make_not_null(&get(subcell_tilde_q)),
                                      get(tilde_q), dg_mesh,
                                      subcell_mesh.extents());

  using std::max;
  using std::min;
  rdmp_tci_data.max_variables_values =
      DataVector{max(max(get(subcell_mag_tilde_e)), max(get(dg_mag_tilde_e))),
                 max(max(get(subcell_mag_tilde_b)), max(get(dg_mag_tilde_b))),
                 max(max(get(subcell_tilde_q)), max(get(tilde_q)))};
  rdmp_tci_data.min_variables_values =
      DataVector{min(min(get(subcell_mag_tilde_e)), min(get(dg_mag_tilde_e))),
                 min(min(get(subcell_mag_tilde_b)), min(get(dg_mag_tilde_b))),
                 min(min(get(subcell_tilde_q)), min(get(tilde_q)))};

  // Perform the TCI checks
  if (evolution::dg::subcell::persson_tci(dg_mag_tilde_e, dg_mesh,
                                          persson_exponent)) {
    return {-1, std::move(rdmp_tci_data)};
  }
  if (evolution::dg::subcell::persson_tci(dg_mag_tilde_b, dg_mesh,
                                          persson_exponent)) {
    return {-2, std::move(rdmp_tci_data)};
  }
  if (evolution::dg::subcell::persson_tci(tilde_q, dg_mesh, persson_exponent)) {
    return {-3, std::move(rdmp_tci_data)};
  }
  if (const int rdmp_tci_status = evolution::dg::subcell::rdmp_tci(
          rdmp_tci_data.max_variables_values,
          rdmp_tci_data.min_variables_values,
          past_rdmp_tci_data.max_variables_values,
          past_rdmp_tci_data.min_variables_values,
          subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon())) {
    return {-(3 + rdmp_tci_status), std::move(rdmp_tci_data)};
  }

  return {0, std::move(rdmp_tci_data)};
}

}  // namespace ForceFree::subcell
