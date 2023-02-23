// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/TciOnDgGrid.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::subcell {
template <size_t Dim>
std::tuple<bool, evolution::dg::subcell::RdmpTciData> TciOnDgGrid<Dim>::apply(
    const Scalar<DataVector>& dg_u, const Mesh<Dim>& dg_mesh,
    const Mesh<Dim>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const TciOptions& tci_options, const double persson_exponent,
    [[maybe_unused]] const bool element_stays_on_dg) {
  // Don't use buffer since we have only one memory allocation right now (until
  // persson_tci can use a buffer)
  const Scalar<DataVector> subcell_u{::evolution::dg::subcell::fd::project(
      get(dg_u), dg_mesh, subcell_mesh.extents())};

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(max(get(dg_u)), max(get(subcell_u)))},
      {min(min(get(dg_u)), min(get(subcell_u)))}};

  const double max_abs_u = max(abs(get(dg_u)));

  const bool cell_is_troubled =
      evolution::dg::subcell::rdmp_tci(rdmp_tci_data.max_variables_values,
                                       rdmp_tci_data.min_variables_values,
                                       past_rdmp_tci_data.max_variables_values,
                                       past_rdmp_tci_data.min_variables_values,
                                       subcell_options.rdmp_delta0(),
                                       subcell_options.rdmp_epsilon()) or
      ((max_abs_u > tci_options.u_cutoff) and
       ::evolution::dg::subcell::persson_tci(dg_u, dg_mesh, persson_exponent));

  return {cell_is_troubled, std::move(rdmp_tci_data)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct TciOnDgGrid<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
