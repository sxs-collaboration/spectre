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

namespace Burgers::fd {
/*!
 * \brief Monotised central reconstruction. See
 * ::fd::reconstruction::monotised_central() for details.
 */
class MonotisedCentral : public Reconstructor {
 private:
  using face_vars_tags = tmpl::list<
      Burgers::Tags::U,
      ::Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>;
  using volume_vars_tags = tmpl::list<Burgers::Tags::U>;

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
  WRAPPED_PUPable_decl_base_template(Reconstructor, MonotisedCentral);
  /// \endcond

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags = tmpl::list<
      ::Tags::Variables<volume_vars_tags>, domain::Tags::Element<1>,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<1>,
      evolution::dg::subcell::Tags::Mesh<1>>;

  void reconstruct(
      gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
          vars_on_upper_face,
      const Variables<tmpl::list<Burgers::Tags::U>>& volume_vars,
      const Element<1>& element,
      const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                         std::pair<Direction<1>, ElementId<1>>,
                         evolution::dg::subcell::NeighborData,
                         boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
          neighbor_data,
      const Mesh<1>& subcell_mesh) const;

  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<face_vars_tags>*> vars_on_face,
      const Variables<volume_vars_tags>& volume_vars, const Element<1>& element,
      const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                         std::pair<Direction<1>, ElementId<1>>,
                         evolution::dg::subcell::NeighborData,
                         boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
          neighbor_data,
      const Mesh<1>& subcell_mesh,
      const Direction<1> direction_to_reconstruct) const;
};

bool operator==(const MonotisedCentral& /*lhs*/,
                const MonotisedCentral& /*rhs*/);

bool operator!=(const MonotisedCentral& lhs, const MonotisedCentral& rhs);
}  // namespace Burgers::fd
