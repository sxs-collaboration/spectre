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
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
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
template <typename recons_tags>
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
 * \brief Fifth order weighted nonlinear compact scheme reconstruction using the
 * Z oscillation indicator. See ::fd::reconstruction::wcns5z() for details.
 */
class Wcns5z : public Reconstructor {
 private:
  using TildeE = ForceFree::Tags::TildeE;
  using TildeB = ForceFree::Tags::TildeB;
  using TildePsi = ForceFree::Tags::TildePsi;
  using TildePhi = ForceFree::Tags::TildePhi;
  using TildeQ = ForceFree::Tags::TildeQ;
  using TildeJ = ForceFree::Tags::TildeJ;

  using volume_vars_tags =
      tmpl::list<TildeE, TildeB, TildePsi, TildePhi, TildeQ>;

  using recons_tags = ForceFree::fd::tags_list_for_reconstruction;

  using FallbackReconstructorType =
      ::fd::reconstruction::FallbackReconstructorType;

 public:
  static constexpr size_t dim = 3;

  struct NonlinearWeightExponent {
    using type = size_t;
    static constexpr Options::String help = {
        "The exponent q to which the oscillation indicator term is raised"};
  };
  struct Epsilon {
    using type = double;
    static constexpr Options::String help = {
        "The parameter added to the oscillation indicators to avoid division "
        "by zero"};
  };
  struct FallbackReconstructor {
    using type = FallbackReconstructorType;
    static constexpr Options::String help = {
        "A reconstruction scheme to fallback to adaptively. Finite difference "
        "will switch to this reconstruction scheme if there are more extrema "
        "in a FD stencil than a specified number. See also the option "
        "'MaxNumberOfExtrema' below. Adaptive fallback is disabled if 'None'."};
  };
  struct MaxNumberOfExtrema {
    using type = size_t;
    static constexpr Options::String help = {
        "The maximum allowed number of extrema in FD stencil for using Wcns5z "
        "reconstruction before switching to a low-order reconstruction. If "
        "FallbackReconstructor=None, this option is ignored"};
  };

  using options = tmpl::list<NonlinearWeightExponent, Epsilon,
                             FallbackReconstructor, MaxNumberOfExtrema>;

  static constexpr Options::String help{"WCNS 5Z reconstruction scheme."};

  Wcns5z() = default;
  Wcns5z(Wcns5z&&) = default;
  Wcns5z& operator=(Wcns5z&&) = default;
  Wcns5z(const Wcns5z&) = default;
  Wcns5z& operator=(const Wcns5z&) = default;
  ~Wcns5z() override = default;

  Wcns5z(size_t nonlinear_weight_exponent, double epsilon,
         FallbackReconstructorType fallback_reconstructor,
         size_t max_number_of_extrema);

  explicit Wcns5z(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, Wcns5z);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  static constexpr bool use_adaptive_order = false;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<volume_vars_tags>, TildeJ,
                 domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>>;

  void reconstruct(
      gsl::not_null<std::array<Variables<recons_tags>, dim>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<recons_tags>, dim>*>
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
      gsl::not_null<Variables<recons_tags>*> vars_on_face,
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

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Wcns5z& lhs, const Wcns5z& rhs);
  friend bool operator!=(const Wcns5z& lhs, const Wcns5z& rhs);

  size_t nonlinear_weight_exponent_ = 0;
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();
  FallbackReconstructorType fallback_reconstructor_ =
      FallbackReconstructorType::None;
  size_t max_number_of_extrema_ = 0;

  void (*reconstruct_)(gsl::not_null<std::array<gsl::span<double>, dim>*>,
                       gsl::not_null<std::array<gsl::span<double>, dim>*>,
                       const gsl::span<const double>&,
                       const DirectionMap<dim, gsl::span<const double>>&,
                       const Index<dim>&, size_t, double, size_t) = nullptr;
  void (*reconstruct_lower_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<dim>&, const Index<dim>&,
                                      const Direction<dim>&, const double&,
                                      const size_t&) = nullptr;
  void (*reconstruct_upper_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<dim>&, const Index<dim>&,
                                      const Direction<dim>&, const double&,
                                      const size_t&) = nullptr;
};

}  // namespace ForceFree::fd
