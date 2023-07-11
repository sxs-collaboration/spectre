// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t dim>
class Direction;
template <size_t dim>
class Element;
template <size_t dim>
class ElementId;
template <size_t dim>
class Mesh;
template <typename face_vars_tags>
class Variables;
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::fd {

/*!
 * \brief Monotonised central reconstruction. See
 * ::fd::reconstruction::monotonised_central() for details.
 */

class MonotonisedCentral : public Reconstructor {
 private:
  using TildeE = ForceFree::Tags::TildeE;
  using TildeB = ForceFree::Tags::TildeB;
  using TildePsi = ForceFree::Tags::TildePsi;
  using TildePhi = ForceFree::Tags::TildePhi;
  using TildeQ = ForceFree::Tags::TildeQ;
  using TildeJ = ForceFree::Tags::TildeJ;

  using volume_vars_tags =
      tmpl::list<TildeE, TildeB, TildePsi, TildePhi, TildeQ>;

  using face_vars_tags =
      tmpl::list<TildeJ,  // we reconstruct TildeJ, not computing it from values
                          // of reconstructed evolved variables
                 TildeE, TildeB, TildePsi, TildePhi, TildeQ,
                 ::Tags::Flux<TildeE, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<TildeB, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<TildePsi, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<TildePhi, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<TildeQ, tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
                 gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
                 evolution::dg::Actions::detail::NormalVector<3>>;

 public:
  static constexpr size_t dim = 3;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotonised central reconstruction scheme."};

  MonotonisedCentral() = default;
  MonotonisedCentral(MonotonisedCentral&&) = default;
  MonotonisedCentral& operator=(MonotonisedCentral&&) = default;
  MonotonisedCentral(const MonotonisedCentral&) = default;
  MonotonisedCentral& operator=(const MonotonisedCentral&) = default;
  ~MonotonisedCentral() override = default;

  explicit MonotonisedCentral(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, MonotonisedCentral);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  static constexpr bool use_adaptive_order = false;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<volume_vars_tags>, TildeJ,
                 domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>>;

  void reconstruct(
      gsl::not_null<std::array<Variables<face_vars_tags>, dim>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<face_vars_tags>, dim>*>
          vars_on_upper_face,
      const Variables<volume_vars_tags>& volume_vars,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
      const Element<dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(dim),
          std::pair<Direction<dim>, ElementId<dim>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>& ghost_data,
      const Mesh<dim>& subcell_mesh) const;

  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<face_vars_tags>*> vars_on_face,
      const Variables<volume_vars_tags>& volume_vars,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
      const Element<dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(dim),
          std::pair<Direction<dim>, ElementId<dim>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>& ghost_data,
      const Mesh<dim>& subcell_mesh,
      const Direction<dim> direction_to_reconstruct) const;
};

bool operator==(const MonotonisedCentral& /*lhs*/,
                const MonotonisedCentral& /*rhs*/);

bool operator!=(const MonotonisedCentral& lhs, const MonotonisedCentral& rhs);

}  // namespace ForceFree::fd
