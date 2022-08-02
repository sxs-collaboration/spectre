// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/TciOnFdGrid.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::subcell {
template <size_t Dim>
std::tuple<bool, evolution::dg::subcell::RdmpTciData> TciOnFdGrid<Dim>::apply(
    const Scalar<DataVector>& subcell_u, const Mesh<Dim>& dg_mesh,
    const Mesh<Dim>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const TciOptions& tci_options, const double persson_exponent) {
  const Scalar<DataVector> dg_u{evolution::dg::subcell::fd::reconstruct(
      get(subcell_u), dg_mesh, subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)};

  using std::max;
  using std::min;
  const evolution::dg::subcell::RdmpTciData rdmp_data_for_tci{
      {max(max(get(dg_u)), max(get(subcell_u)))},
      {min(min(get(dg_u)), min(get(subcell_u)))}};

  const double max_abs_u = max(abs(get(dg_u)));

  const bool cell_is_troubled = evolution::dg::subcell::rdmp_tci(
      rdmp_data_for_tci.max_variables_values,
      rdmp_data_for_tci.min_variables_values,
      past_rdmp_tci_data.max_variables_values,
      past_rdmp_tci_data.min_variables_values, subcell_options.rdmp_delta0(),
      subcell_options.rdmp_epsilon());

  return {cell_is_troubled or ((max_abs_u > tci_options.u_cutoff) and
                               ::evolution::dg::subcell::persson_tci(
                                   dg_u, dg_mesh, persson_exponent)),
          {{max(get(subcell_u))}, {min(get(subcell_u))}}};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct TciOnFdGrid<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
