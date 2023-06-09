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
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
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

namespace Burgers::fd {
/*!
 * \brief Monotonised central reconstruction. See
 * ::fd::reconstruction::monotonised_central() for details.
 */
class MonotonisedCentral : public Reconstructor {
 private:
  using face_vars_tags = tmpl::list<
      Burgers::Tags::U,
      ::Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>;
  using volume_vars_tags = tmpl::list<Burgers::Tags::U>;

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
  WRAPPED_PUPable_decl_base_template(Reconstructor, MonotonisedCentral);
  /// \endcond

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<volume_vars_tags>, domain::Tags::Element<1>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<1>,
                 evolution::dg::subcell::Tags::Mesh<1>>;

  void reconstruct(
      gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
          vars_on_upper_face,
      const Variables<tmpl::list<Burgers::Tags::U>>& volume_vars,
      const Element<1>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(1), std::pair<Direction<1>, ElementId<1>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<1>, ElementId<1>>>>& ghost_data,
      const Mesh<1>& subcell_mesh) const;

  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<face_vars_tags>*> vars_on_face,
      const Variables<volume_vars_tags>& volume_vars, const Element<1>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(1), std::pair<Direction<1>, ElementId<1>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<1>, ElementId<1>>>>& ghost_data,
      const Mesh<1>& subcell_mesh,
      const Direction<1> direction_to_reconstruct) const;
};

bool operator==(const MonotonisedCentral& /*lhs*/,
                const MonotonisedCentral& /*rhs*/);

bool operator!=(const MonotonisedCentral& lhs, const MonotonisedCentral& rhs);
}  // namespace Burgers::fd
