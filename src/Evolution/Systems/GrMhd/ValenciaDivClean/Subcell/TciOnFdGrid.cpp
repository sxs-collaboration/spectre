// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnFdGrid.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace grmhd::ValenciaDivClean::subcell {
std::tuple<bool, evolution::dg::subcell::RdmpTciData> TciOnFdGrid::apply(
    const Scalar<DataVector>& subcell_tilde_d,
    const Scalar<DataVector>& subcell_tilde_tau,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
    const bool vars_needed_fixing, const Mesh<3>& dg_mesh,
    const Mesh<3>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const TciOptions& tci_options,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const double persson_exponent) {
  const size_t num_dg_pts = dg_mesh.number_of_grid_points();
  const size_t num_subcell_pts = subcell_mesh.number_of_grid_points();
  DataVector temp_buffer{num_subcell_pts + 3 * num_dg_pts};
  size_t offset_into_temp_buffer = 0;
  const auto assign_data =
      [&temp_buffer, &offset_into_temp_buffer](
          const gsl::not_null<Scalar<DataVector>*> to_assign,
          const size_t size) {
        get(*to_assign)
            .set_data_ref(temp_buffer.data() + offset_into_temp_buffer, size);
        offset_into_temp_buffer += size;
      };

  Scalar<DataVector> subcell_mag_tilde_b{};
  assign_data(make_not_null(&subcell_mag_tilde_b), num_subcell_pts);
  magnitude(make_not_null(&subcell_mag_tilde_b), subcell_tilde_b);

  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  rdmp_tci_data.max_variables_values =
      std::vector{max(get(subcell_tilde_d)), max(get(subcell_tilde_tau)),
                  max(get(subcell_mag_tilde_b))};
  rdmp_tci_data.min_variables_values =
      std::vector{min(get(subcell_tilde_d)), min(get(subcell_tilde_tau)),
                  min(get(subcell_mag_tilde_b))};

  Scalar<DataVector> dg_tilde_d{};
  assign_data(make_not_null(&dg_tilde_d), num_dg_pts);
  evolution::dg::subcell::fd::reconstruct(
      make_not_null(&get(dg_tilde_d)), get(subcell_tilde_d), dg_mesh,
      subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
  Scalar<DataVector> dg_tilde_tau{};
  assign_data(make_not_null(&dg_tilde_tau), num_dg_pts);
  evolution::dg::subcell::fd::reconstruct(
      make_not_null(&get(dg_tilde_tau)), get(subcell_tilde_tau), dg_mesh,
      subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

  // Note: `or` is short-circuiting, so if any of the first tests fail, we won't
  // do the more expensive Persson TCI test.
  bool cell_is_troubled =
      vars_needed_fixing or
      min(get(dg_tilde_d)) <
          tci_options.minimum_rest_mass_density_times_lorentz_factor or
      min(get(dg_tilde_tau)) < tci_options.minimum_tilde_tau or
      evolution::dg::subcell::persson_tci(dg_tilde_d, dg_mesh,
                                          persson_exponent) or
      evolution::dg::subcell::persson_tci(dg_tilde_tau, dg_mesh,
                                          persson_exponent);

  if (cell_is_troubled) {
    return {cell_is_troubled, rdmp_tci_data};
  }

  Scalar<DataVector> dg_mag_tilde_b{};
  assign_data(make_not_null(&dg_mag_tilde_b), num_dg_pts);
  evolution::dg::subcell::fd::reconstruct(
      make_not_null(&get(dg_mag_tilde_b)), get(subcell_mag_tilde_b), dg_mesh,
      subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
  // Add the reconstructed DG solution to the check. This is done so that we
  // wouldn't violate RDMP if we switch back. However, we don't want to return
  // max/mins from a bad DG solution.
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data_for_check{};
  rdmp_tci_data_for_check.max_variables_values = std::vector{
      max(max(get(dg_tilde_d)), rdmp_tci_data.max_variables_values[0]),
      max(max(get(dg_tilde_tau)), rdmp_tci_data.max_variables_values[1]),
      max(max(get(dg_mag_tilde_b)), rdmp_tci_data.max_variables_values[2])};
  rdmp_tci_data_for_check.min_variables_values = std::vector{
      min(min(get(dg_tilde_d)), rdmp_tci_data.min_variables_values[0]),
      min(min(get(dg_tilde_tau)), rdmp_tci_data.min_variables_values[1]),
      min(min(get(dg_mag_tilde_b)), rdmp_tci_data.min_variables_values[2])};

  cell_is_troubled |= evolution::dg::subcell::rdmp_tci(
      rdmp_tci_data_for_check.max_variables_values,
      rdmp_tci_data_for_check.min_variables_values,
      past_rdmp_tci_data.max_variables_values,
      past_rdmp_tci_data.min_variables_values, subcell_options.rdmp_delta0(),
      subcell_options.rdmp_epsilon());

  if (tci_options.magnetic_field_cutoff.has_value() and not cell_is_troubled) {
    cell_is_troubled = (max(get(dg_mag_tilde_b)) >
                            tci_options.magnetic_field_cutoff.value() and
                        evolution::dg::subcell::persson_tci(
                            dg_mag_tilde_b, dg_mesh, persson_exponent));
  }

  return {cell_is_troubled, rdmp_tci_data};
}
}  // namespace grmhd::ValenciaDivClean::subcell
