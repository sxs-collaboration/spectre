// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Monotised central reconstruction. See
 * ::fd::reconstruction::monotised_central() for details.
 */
class MonotisedCentralPrim : public Reconstructor {
 private:
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

 public:
  static constexpr size_t dim = 3;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotised central reconstruction scheme using primitive variables."};

  MonotisedCentralPrim() = default;
  MonotisedCentralPrim(MonotisedCentralPrim&&) = default;
  MonotisedCentralPrim& operator=(MonotisedCentralPrim&&) = default;
  MonotisedCentralPrim(const MonotisedCentralPrim&) = default;
  MonotisedCentralPrim& operator=(const MonotisedCentralPrim&) = default;
  ~MonotisedCentralPrim() override = default;

  explicit MonotisedCentralPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, MonotisedCentralPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                 hydro::Tags::EquationOfStateBase, domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::
                     NeighborDataForReconstructionAndRdmpTci<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>>;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, dim>*> vars_on_upper_face,
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(dim) + 1,
          std::pair<Direction<dim>, ElementId<dim>>,
          evolution::dg::subcell::NeighborData,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
          neighbor_data,
      const Mesh<dim>& subcell_mesh) const;

  /// Called by an element doing DG when the neighbor is doing subcell.
  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(dim) + 1,
          std::pair<Direction<dim>, ElementId<dim>>,
          evolution::dg::subcell::NeighborData,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
          neighbor_data,
      const Mesh<dim>& subcell_mesh,
      const Direction<dim> direction_to_reconstruct) const;
};

bool operator==(const MonotisedCentralPrim& /*lhs*/,
                const MonotisedCentralPrim& /*rhs*/);

bool operator!=(const MonotisedCentralPrim& lhs,
                const MonotisedCentralPrim& rhs);
}  // namespace grmhd::ValenciaDivClean::fd
