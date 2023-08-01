// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianEuler::subcell {
/// \brief Sets the initial RDMP data.
///
/// Used on the subcells after the TCI marked the DG solution as inadmissible.
template <size_t Dim>
struct SetInitialRdmpData {
 private:
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;

 public:
  using argument_tags = tmpl::list<
      ::Tags::Variables<
          tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>,
      evolution::dg::subcell::Tags::ActiveGrid, ::domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>>;
  using return_tags = tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;

  static void apply(
      gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
      const Variables<tmpl::list<MassDensityCons, MomentumDensity,
                                 EnergyDensity>>& subcell_vars,
      evolution::dg::subcell::ActiveGrid active_grid, const Mesh<Dim>& dg_mesh,
      const Mesh<Dim>& subcell_mesh);
};
}  // namespace NewtonianEuler::subcell
