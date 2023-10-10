// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
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
 * \brief Adaptive order FD reconstruction. See
 * ::fd::reconstruction::positivity_preserving_adaptive_order() for details.
 * Note that in the ForceFree evolution system no variable needs to be kept
 * positive.
 *
 */
class AdaptiveOrder : public Reconstructor {
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

  struct Alpha5 {
    using type = double;
    static constexpr Options::String help = {
        "The alpha parameter in the Persson convergence measurement. 4 is the "
        "right value, but anything in the range of 3-5 is 'reasonable'. "
        "Smaller values allow for more oscillations."};
  };
  struct Alpha7 {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "The alpha parameter in the Persson convergence measurement. 4 is the "
        "right value, but anything in the range of 3-5 is 'reasonable'. "
        "Smaller values allow for more oscillations. If not specified then "
        "7th-order reconstruction is not used."};
  };
  struct Alpha9 {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "The alpha parameter in the Persson convergence measurement. 4 is the "
        "right value, but anything in the range of 3-5 is 'reasonable'. "
        "Smaller values allow for more oscillations. If not specified then "
        "9th-order reconstruction is not used."};
  };
  struct LowOrderReconstructor {
    using type = FallbackReconstructorType;
    static constexpr Options::String help = {
        "The 2nd/3rd-order reconstruction scheme to use if unlimited 5th-order "
        "isn't okay."};
  };

  using options = tmpl::list<Alpha5, Alpha7, Alpha9, LowOrderReconstructor>;

  static constexpr Options::String help{"Adaptive-order reconstruction."};
  AdaptiveOrder() = default;
  AdaptiveOrder(AdaptiveOrder&&) = default;
  AdaptiveOrder& operator=(AdaptiveOrder&&) = default;
  AdaptiveOrder(const AdaptiveOrder&) = default;
  AdaptiveOrder& operator=(const AdaptiveOrder&) = default;
  ~AdaptiveOrder() override = default;

  AdaptiveOrder(double alpha_5, std::optional<double> alpha_7,
                std::optional<double> alpha_9,
                FallbackReconstructorType low_order_reconstructor,
                const Options::Context& context = {});

  explicit AdaptiveOrder(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, AdaptiveOrder);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  static constexpr bool use_adaptive_order = true;
  bool supports_adaptive_order() const override { return use_adaptive_order; }

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override {
    return eight_to_the_alpha_9_.has_value()
               ? 5
               : (six_to_the_alpha_7_.has_value() ? 4 : 3);
  }

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
  friend bool operator==(const AdaptiveOrder& lhs, const AdaptiveOrder& rhs);
  friend bool operator!=(const AdaptiveOrder& lhs, const AdaptiveOrder& rhs);
  void set_function_pointers();

  double four_to_the_alpha_5_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<double> six_to_the_alpha_7_{};
  std::optional<double> eight_to_the_alpha_9_{};
  FallbackReconstructorType low_order_reconstructor_ =
      FallbackReconstructorType::None;

  using PointerReconsOrder = void (*)(
      gsl::not_null<std::array<gsl::span<double>, dim>*>,
      gsl::not_null<std::array<gsl::span<double>, dim>*>,
      gsl::not_null<std::optional<std::array<gsl::span<std::uint8_t>, dim>>*>,
      const gsl::span<const double>&,
      const DirectionMap<dim, gsl::span<const double>>&, const Index<dim>&,
      size_t, double, double, double);
  using PointerRecons =
      void (*)(gsl::not_null<std::array<gsl::span<double>, dim>*>,
               gsl::not_null<std::array<gsl::span<double>, dim>*>,
               const gsl::span<const double>&,
               const DirectionMap<dim, gsl::span<const double>>&,
               const Index<dim>&, size_t, double, double, double);
  PointerRecons reconstruct_ = nullptr;
  PointerReconsOrder pp_reconstruct_ = nullptr;

  using PointerNeighbor = void (*)(gsl::not_null<DataVector*>,
                                   const DataVector&, const DataVector&,
                                   const Index<dim>&, const Index<dim>&,
                                   const Direction<dim>&, const double&,
                                   const double&, const double&);
  PointerNeighbor reconstruct_lower_neighbor_ = nullptr;
  PointerNeighbor reconstruct_upper_neighbor_ = nullptr;
  PointerNeighbor pp_reconstruct_lower_neighbor_ = nullptr;
  PointerNeighbor pp_reconstruct_upper_neighbor_ = nullptr;
};

}  // namespace ForceFree::fd
