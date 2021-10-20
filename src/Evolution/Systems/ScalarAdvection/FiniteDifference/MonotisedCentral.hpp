// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace evolution::dg::subcell {
class NeighborData;
}  // namespace evolution::dg::subcell
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarAdvection::fd {
/*!
 * \brief Monotised central reconstruction. See
 * ::fd::reconstruction::monotised_central() for details.
 */
template <size_t Dim>
class MonotisedCentral : public Reconstructor<Dim> {
 private:
  using face_vars_tags =
      tmpl::list<Tags::U,
                 ::Tags::Flux<Tags::U, tmpl::size_t<Dim>, Frame::Inertial>>;
  using volume_vars_tags = tmpl::list<Tags::U>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotised central reconstruction scheme."};

  MonotisedCentral() = default;
  MonotisedCentral(MonotisedCentral&&) = default;
  MonotisedCentral& operator=(MonotisedCentral&&) = default;
  MonotisedCentral(const MonotisedCentral&) = default;
  MonotisedCentral& operator=(const MonotisedCentral&) = default;
  ~MonotisedCentral() override = default;

  void pup(PUP::er& p) override;

  /// \cond
  explicit MonotisedCentral(CkMigrateMessage* msg);
  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>, MonotisedCentral);
  /// \endcond

  auto get_clone() const -> std::unique_ptr<Reconstructor<Dim>> override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags = tmpl::list<
      ::Tags::Variables<volume_vars_tags>, domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
          Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>>;

  template <typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
      const Variables<tmpl::list<Tags::U>>& volume_vars,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim) + 1,
          std::pair<Direction<Dim>, ElementId<Dim>>,
          evolution::dg::subcell::NeighborData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data,
      const Mesh<Dim>& subcell_mesh) const;

  template <typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<tmpl::list<Tags::U>>& volume_vars,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim) + 1,
          std::pair<Direction<Dim>, ElementId<Dim>>,
          evolution::dg::subcell::NeighborData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
          neighbor_data,
      const Mesh<Dim>& subcell_mesh,
      const Direction<Dim> direction_to_reconstruct) const;
};

template <size_t Dim>
bool operator==(const MonotisedCentral<Dim>& /*lhs*/,
                const MonotisedCentral<Dim>& /*rhs*/);

template <size_t Dim>
bool operator!=(const MonotisedCentral<Dim>& lhs,
                const MonotisedCentral<Dim>& rhs);
}  // namespace ScalarAdvection::fd
