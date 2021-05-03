// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
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
namespace evolution::dg::subcell {
struct NeighborData;
}  // namespace evolution::dg::subcell
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianEuler::fd {
/*!
 * \brief Monotised central reconstruction. See
 * `::fd::reconstruction::monotised_central()` for details.
 */
template <size_t Dim>
class MonotisedCentralPrim : public Reconstructor<Dim> {
 private:
  // Conservative vars tags
  using MassDensityCons = Tags::MassDensityCons;
  using EnergyDensity = Tags::EnergyDensity;
  using MomentumDensity = Tags::MomentumDensity<Dim>;

  // Primitive vars tags
  using MassDensity = Tags::MassDensity<DataVector>;
  using Velocity = Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy = Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = Tags::Pressure<DataVector>;

  using prims_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<Dim>,
                                     Frame::Inertial>;
  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotised central reconstruction scheme using primitive variables."};

  MonotisedCentralPrim() = default;
  MonotisedCentralPrim(MonotisedCentralPrim&&) noexcept = default;
  MonotisedCentralPrim& operator=(MonotisedCentralPrim&&) noexcept = default;
  MonotisedCentralPrim(const MonotisedCentralPrim&) = default;
  MonotisedCentralPrim& operator=(const MonotisedCentralPrim&) = default;
  ~MonotisedCentralPrim() override = default;

  explicit MonotisedCentralPrim(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>, MonotisedCentralPrim);

  auto get_clone() const noexcept
      -> std::unique_ptr<Reconstructor<Dim>> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const noexcept override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<prims_tags>,
                 hydro::Tags::EquationOfStateBase, domain::Tags::Element<Dim>,
                 evolution::dg::subcell::Tags::
                     NeighborDataForReconstructionAndRdmpTci<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>>;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
      const Variables<prims_tags>& volume_prims,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim) + 1,
          std::pair<Direction<Dim>, ElementId<Dim>>,
          evolution::dg::subcell::NeighborData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data,
      const Mesh<Dim>& subcell_mesh) const noexcept;

  /// Called by an element doing DG when the neighbor is doing subcell.
  ///
  /// This is used to reconstruct the fluxes on the mortar that the subcell
  /// neighbor would have sent had we instead used a two a two-communication
  /// subcell solver (first communication for reconstruction, second for
  /// fluxes).
  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<prims_tags>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim) + 1,
          std::pair<Direction<Dim>, ElementId<Dim>>,
          evolution::dg::subcell::NeighborData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data,
      const Mesh<Dim>& subcell_mesh,
      const Direction<Dim> direction_to_reconstruct) const noexcept;
};
}  // namespace NewtonianEuler::fd
