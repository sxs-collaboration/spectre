// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
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

namespace Burgers::fd {
/*!
 * \brief Monotised central reconstruction. See
 * `::fd::reconstruction::monotised_central()` for details.
 */
class MonotisedCentral : public Reconstructor {
  using face_vars_tags =
      tmpl::list<Tags::U,
                 ::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>;
  using volume_vars_tags = tmpl::list<Tags::U>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotised central reconstruction scheme"};

  MonotisedCentral() = default;
  MonotisedCentral(MonotisedCentral&&) noexcept = default;
  MonotisedCentral& operator=(MonotisedCentral&&) noexcept = default;
  MonotisedCentral(const MonotisedCentral&) = default;
  MonotisedCentral& operator=(const MonotisedCentral&) = default;
  ~MonotisedCentral() override = default;

  explicit MonotisedCentral(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl(MonotisedCentral);

  auto get_clone() const noexcept -> std::unique_ptr<Reconstructor> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const noexcept override { return 2; }

  using reconstruction_argument_tags = tmpl::list<
      ::Tags::Variables<volume_vars_tags>, domain::Tags::Element<1>,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<1>,
      evolution::dg::subcell::Tags::Mesh<1>>;

  void reconstruct(
      gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
          vars_on_upper_face,
      const Variables<volume_vars_tags>& volume_vars, const Element<1>& element,
      const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                         std::pair<Direction<1>, ElementId<1>>,
                         evolution::dg::subcell::NeighborData,
                         boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
          neighbor_data,
      const Mesh<1>& subcell_mesh) const noexcept;

  /// Called by an element doing DG when the neighbor is doing subcell.
  ///
  /// This is used to reconstruct the fluxes on the mortar that the subcell
  /// neighbor would have sent had we instead used a two a two-communication
  /// subcell solver (first communication for reconstruction, second for
  /// fluxes).
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<face_vars_tags>*> vars_on_face,
      const Variables<volume_vars_tags>& subcell_volume_vars,
      const Element<1>& element,
      const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                         std::pair<Direction<1>, ElementId<1>>,
                         evolution::dg::subcell::NeighborData,
                         boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
          neighbor_data,
      const Mesh<1>& subcell_mesh,
      const Direction<1> direction_to_reconstruct) const noexcept;
};

bool operator==(const MonotisedCentral& /*lhs*/,
                const MonotisedCentral& /*rhs*/) noexcept;

bool operator!=(const MonotisedCentral& lhs,
                const MonotisedCentral& rhs) noexcept;
}  // namespace Burgers::fd
