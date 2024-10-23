// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Monotonised central reconstruction. See
 * ::fd::reconstruction::monotonised_central() for details.
 */
class MonotonisedCentralPrim : public Reconstructor {
 private:
  // pressure -> temperature
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

 public:
  static constexpr size_t dim = 3;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotonised central reconstruction scheme using primitive variables."};

  MonotonisedCentralPrim() = default;
  MonotonisedCentralPrim(MonotonisedCentralPrim&&) = default;
  MonotonisedCentralPrim& operator=(MonotonisedCentralPrim&&) = default;
  MonotonisedCentralPrim(const MonotonisedCentralPrim&) = default;
  MonotonisedCentralPrim& operator=(const MonotonisedCentralPrim&) = default;
  ~MonotonisedCentralPrim() override = default;

  explicit MonotonisedCentralPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, MonotonisedCentralPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  static constexpr bool use_adaptive_order = false;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                 hydro::Tags::GrmhdEquationOfState, domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>,
                 ::Tags::VariableFixer<VariableFixing::FixToAtmosphere<dim>>>;

  template <size_t ThermodynamicDim>
  void reconstruct(
      gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, dim>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, dim>*>
          vars_on_upper_face,
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<dim>& subcell_mesh,
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const;

  /// Called by an element doing DG when the neighbor is doing subcell.
  template <size_t ThermodynamicDim>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<tags_list_for_reconstruct>*> vars_on_face,
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<dim>& subcell_mesh,
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,
      const Direction<dim> direction_to_reconstruct) const;
};

bool operator==(const MonotonisedCentralPrim& /*lhs*/,
                const MonotonisedCentralPrim& /*rhs*/);

bool operator!=(const MonotonisedCentralPrim& lhs,
                const MonotonisedCentralPrim& rhs);
}  // namespace grmhd::ValenciaDivClean::fd
