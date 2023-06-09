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
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
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
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace ScalarAdvection::fd {
/*!
 * \brief Monotonised central reconstruction. See
 * ::fd::reconstruction::monotonised_central() for details.
 */
template <size_t Dim>
class MonotonisedCentral : public Reconstructor<Dim> {
 private:
  using volume_vars_tags = tmpl::list<Tags::U>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotonised central reconstruction scheme."};

  MonotonisedCentral() = default;
  MonotonisedCentral(MonotonisedCentral&&) = default;
  MonotonisedCentral& operator=(MonotonisedCentral&&) = default;
  MonotonisedCentral(const MonotonisedCentral&) = default;
  MonotonisedCentral& operator=(const MonotonisedCentral&) = default;
  ~MonotonisedCentral() override = default;

  void pup(PUP::er& p) override;

  /// \cond
  explicit MonotonisedCentral(CkMigrateMessage* msg);
  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>, MonotonisedCentral);
  /// \endcond

  auto get_clone() const -> std::unique_ptr<Reconstructor<Dim>> override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<volume_vars_tags>,
                 domain::Tags::Element<Dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>>;

  template <typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
      const Variables<tmpl::list<Tags::U>>& volume_vars,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim),
          std::pair<Direction<Dim>, ElementId<Dim>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& ghost_data,
      const Mesh<Dim>& subcell_mesh) const;

  template <typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<tmpl::list<Tags::U>>& volume_vars,
      const Element<Dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(Dim),
          std::pair<Direction<Dim>, ElementId<Dim>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& ghost_data,
      const Mesh<Dim>& subcell_mesh,
      const Direction<Dim> direction_to_reconstruct) const;
};

template <size_t Dim>
bool operator==(const MonotonisedCentral<Dim>& /*lhs*/,
                const MonotonisedCentral<Dim>& /*rhs*/);

template <size_t Dim>
bool operator!=(const MonotonisedCentral<Dim>& lhs,
                const MonotonisedCentral<Dim>& rhs);
}  // namespace ScalarAdvection::fd
