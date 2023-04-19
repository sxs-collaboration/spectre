// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/TciOnFdGrid.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace ForceFree::subcell {

std::tuple<int, evolution::dg::subcell::RdmpTciData> TciOnFdGrid::apply(
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
    const Scalar<DataVector>& subcell_tilde_q, const Mesh<3>& dg_mesh,
    const Mesh<3>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const double persson_exponent, const bool need_rdmp_data_only) {
  const size_t num_dg_pts = dg_mesh.number_of_grid_points();
  const size_t num_subcell_pts = subcell_mesh.number_of_grid_points();

  DataVector temp_buffer{3 * num_dg_pts + 2 * num_subcell_pts};
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

  Scalar<DataVector> subcell_mag_tilde_e{};
  assign_data(make_not_null(&subcell_mag_tilde_e), num_subcell_pts);
  magnitude(make_not_null(&subcell_mag_tilde_e), subcell_tilde_e);

  Scalar<DataVector> subcell_mag_tilde_b{};
  assign_data(make_not_null(&subcell_mag_tilde_b), num_subcell_pts);
  magnitude(make_not_null(&subcell_mag_tilde_b), subcell_tilde_b);

  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  rdmp_tci_data.max_variables_values =
      DataVector{max(get(subcell_mag_tilde_e)), max(get(subcell_mag_tilde_b)),
                 max(get(subcell_tilde_q))};
  rdmp_tci_data.min_variables_values =
      DataVector{min(get(subcell_mag_tilde_e)), min(get(subcell_mag_tilde_b)),
                 min(get(subcell_tilde_q))};

  if (need_rdmp_data_only) {
    return {false, rdmp_tci_data};
  }

  // Note : we compute mag(E), mag(B) on FD grid first and then project their
  // magnitude to DG grid, NOT projecting E^i and B^i on DG grid then compute
  // magnitudes of them.

  Scalar<DataVector> dg_mag_tilde_e{};
  assign_data(make_not_null(&dg_mag_tilde_e), num_dg_pts);
  evolution::dg::subcell::fd::reconstruct(
      make_not_null(&get(dg_mag_tilde_e)), get(subcell_mag_tilde_e), dg_mesh,
      subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

  Scalar<DataVector> dg_mag_tilde_b{};
  assign_data(make_not_null(&dg_mag_tilde_b), num_dg_pts);
  evolution::dg::subcell::fd::reconstruct(
      make_not_null(&get(dg_mag_tilde_b)), get(subcell_mag_tilde_b), dg_mesh,
      subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

  Scalar<DataVector> dg_tilde_q{};
  assign_data(make_not_null(&dg_tilde_q), num_dg_pts);
  evolution::dg::subcell::fd::reconstruct(
      make_not_null(&get(dg_tilde_q)), get(subcell_tilde_q), dg_mesh,
      subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

  if (evolution::dg::subcell::persson_tci(dg_mag_tilde_e, dg_mesh,
                                          persson_exponent)) {
    return {+1, rdmp_tci_data};
  }
  if (evolution::dg::subcell::persson_tci(dg_mag_tilde_b, dg_mesh,
                                          persson_exponent)) {
    return {+2, rdmp_tci_data};
  }
  if (evolution::dg::subcell::persson_tci(dg_tilde_q, dg_mesh,
                                          persson_exponent)) {
    return {+3, rdmp_tci_data};
  }

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data_for_check{};
  rdmp_tci_data_for_check.max_variables_values = DataVector{
      max(max(get(dg_mag_tilde_e)), rdmp_tci_data.max_variables_values[0]),
      max(max(get(dg_mag_tilde_b)), rdmp_tci_data.max_variables_values[1]),
      max(max(get(dg_tilde_q)), rdmp_tci_data.max_variables_values[2])};
  rdmp_tci_data_for_check.min_variables_values = DataVector{
      min(min(get(dg_mag_tilde_e)), rdmp_tci_data.min_variables_values[0]),
      min(min(get(dg_mag_tilde_b)), rdmp_tci_data.min_variables_values[1]),
      min(min(get(dg_tilde_q)), rdmp_tci_data.min_variables_values[2])};

  if (const int rdmp_tci_status = evolution::dg::subcell::rdmp_tci(
          rdmp_tci_data_for_check.max_variables_values,
          rdmp_tci_data_for_check.min_variables_values,
          past_rdmp_tci_data.max_variables_values,
          past_rdmp_tci_data.min_variables_values,
          subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon())) {
    return {+3 + rdmp_tci_status, rdmp_tci_data};
  }

  return {0, std::move(rdmp_tci_data)};
}

}  // namespace ForceFree::subcell
