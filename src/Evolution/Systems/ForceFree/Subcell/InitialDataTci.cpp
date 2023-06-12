// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/InitialDataTci.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace ForceFree::subcell {

namespace detail {
std::tuple<int, evolution::dg::subcell::RdmpTciData> initial_data_tci_work(
    const Scalar<DataVector>& dg_tilde_e_mag,
    const Scalar<DataVector>& dg_tilde_b_mag,
    const Scalar<DataVector>& dg_tilde_q,
    const Scalar<DataVector>& subcell_tilde_e_mag,
    const Scalar<DataVector>& subcell_tilde_b_mag,
    const Scalar<DataVector>& subcell_tilde_q, const double persson_exponent,
    const Mesh<3>& dg_mesh) {
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  using std::max;
  using std::min;
  rdmp_tci_data.max_variables_values =
      DataVector{max(max(get(dg_tilde_e_mag)), max(get(subcell_tilde_e_mag))),
                 max(max(get(dg_tilde_b_mag)), max(get(subcell_tilde_b_mag))),
                 max(max(get(dg_tilde_q)), max(get(subcell_tilde_q)))};
  rdmp_tci_data.min_variables_values =
      DataVector{min(min(get(dg_tilde_e_mag)), min(get(subcell_tilde_e_mag))),
                 min(min(get(dg_tilde_b_mag)), min(get(subcell_tilde_b_mag))),
                 min(min(get(dg_tilde_q)), min(get(subcell_tilde_q)))};

  if (evolution::dg::subcell::persson_tci(dg_tilde_e_mag, dg_mesh,
                                          persson_exponent)) {
    return {-1, std::move(rdmp_tci_data)};
  }
  if (evolution::dg::subcell::persson_tci(dg_tilde_b_mag, dg_mesh,
                                          persson_exponent)) {
    return {-2, std::move(rdmp_tci_data)};
  }
  if (evolution::dg::subcell::persson_tci(dg_tilde_q, dg_mesh,
                                          persson_exponent)) {
    return {-3, std::move(rdmp_tci_data)};
  }
  return {0, std::move(rdmp_tci_data)};
}

}  // namespace detail

std::tuple<int, evolution::dg::subcell::RdmpTciData> DgInitialDataTci::apply(
    const Variables<tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                               Tags::TildePhi, Tags::TildeQ>>& dg_vars,
    const double rdmp_delta0, const double rdmp_epsilon,
    const double persson_exponent, const Mesh<3>& dg_mesh,
    const Mesh<3>& subcell_mesh) {
  const Scalar<DataVector> dg_tilde_e_magnitude =
      magnitude(get<Tags::TildeE>(dg_vars));
  const Scalar<DataVector> dg_tilde_b_magnitude =
      magnitude(get<Tags::TildeB>(dg_vars));

  const auto subcell_vars = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());
  const Scalar<DataVector> subcell_tilde_e_magnitude =
      magnitude(get<Tags::TildeE>(subcell_vars));
  const Scalar<DataVector> subcell_tilde_b_magnitude =
      magnitude(get<Tags::TildeB>(subcell_vars));

  auto result = detail::initial_data_tci_work(
      dg_tilde_e_magnitude, dg_tilde_b_magnitude, get<Tags::TildeQ>(dg_vars),
      subcell_tilde_e_magnitude, subcell_tilde_b_magnitude,
      get<Tags::TildeQ>(subcell_vars), persson_exponent, dg_mesh);

  const int tci_status = std::get<0>(result);

  if (static_cast<bool>(tci_status)) {
    return {tci_status, std::move(std::get<1>(result))};
  }
  if (static_cast<bool>(evolution::dg::subcell::two_mesh_rdmp_tci(
          dg_vars, subcell_vars, rdmp_delta0, rdmp_epsilon))) {
    return {-4, std::move(std::get<1>(result))};
  }
  return {0, std::move(std::get<1>(result))};
}

void SetInitialRdmpData::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
    const Scalar<DataVector>& subcell_tilde_q,
    const evolution::dg::subcell::ActiveGrid active_grid) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    const Scalar<DataVector> subcell_tilde_e_magnitude =
        magnitude(subcell_tilde_e);
    const Scalar<DataVector> subcell_tilde_b_magnitude =
        magnitude(subcell_tilde_b);

    rdmp_tci_data->max_variables_values = DataVector{
        max(get(subcell_tilde_e_magnitude)),
        max(get(subcell_tilde_b_magnitude)), max(get(subcell_tilde_q))};
    rdmp_tci_data->min_variables_values = DataVector{
        min(get(subcell_tilde_e_magnitude)),
        min(get(subcell_tilde_b_magnitude)), min(get(subcell_tilde_q))};
  }
}

}  // namespace ForceFree::subcell
