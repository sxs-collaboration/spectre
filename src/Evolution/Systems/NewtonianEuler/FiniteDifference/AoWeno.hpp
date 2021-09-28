// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace NewtonianEuler::fd {
/*!
 * \brief Adaptive-order WENO reconstruction hybridizing orders 5 and 3. See
 * ::fd::reconstruction::aoweno_53() for details.
 */
template <size_t Dim>
class AoWeno53Prim : public Reconstructor<Dim> {
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
  struct GammaHi {
    using type = double;
    static constexpr Options::String help = {
        "The linear weight for the 5th-order stencil."};
  };
  struct GammaLo {
    using type = double;
    static constexpr Options::String help = {
        "The linear weight for the central 3rd-order stencil."};
  };
  struct Epsilon {
    using type = double;
    static constexpr Options::String help = {
        "The parameter added to the oscillation indicators to avoid division "
        "by zero"};
  };
  struct NonlinearWeightExponent {
    using type = size_t;
    static constexpr Options::String help = {
        "The exponent q to which the oscillation indicators are raised"};
  };

  using options =
      tmpl::list<GammaHi, GammaLo, Epsilon, NonlinearWeightExponent>;
  static constexpr Options::String help{
      "Monotised central reconstruction scheme using primitive variables."};

  AoWeno53Prim() = default;
  AoWeno53Prim(AoWeno53Prim&&) = default;
  AoWeno53Prim& operator=(AoWeno53Prim&&) = default;
  AoWeno53Prim(const AoWeno53Prim&) = default;
  AoWeno53Prim& operator=(const AoWeno53Prim&) = default;
  ~AoWeno53Prim() override = default;

  AoWeno53Prim(double gamma_hi, double gamma_lo, double epsilon,
               size_t nonlinear_weight_exponent);

  explicit AoWeno53Prim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>, AoWeno53Prim);

  auto get_clone() const -> std::unique_ptr<Reconstructor<Dim>> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

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
      const Mesh<Dim>& subcell_mesh) const;

  /// Called by an element doing DG when the neighbor is doing subcell.
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
      const Direction<Dim> direction_to_reconstruct) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const AoWeno53Prim<LocalDim>& lhs,
                         const AoWeno53Prim<LocalDim>& rhs);

  double gamma_hi_ = std::numeric_limits<double>::signaling_NaN();
  double gamma_lo_ = std::numeric_limits<double>::signaling_NaN();
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();
  size_t nonlinear_weight_exponent_ = 0;

  void (*reconstruct_)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                       gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                       const gsl::span<const double>&,
                       const DirectionMap<Dim, gsl::span<const double>>&,
                       const Index<Dim>&, size_t, double, double, double);
  void (*reconstruct_lower_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<Dim>&, const Index<Dim>&,
                                      const Direction<Dim>&, const double&,
                                      const double&, const double&);
  void (*reconstruct_upper_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<Dim>&, const Index<Dim>&,
                                      const Direction<Dim>&, const double&,
                                      const double&, const double&);
};

template <size_t Dim>
bool operator!=(const AoWeno53Prim<Dim>& lhs, const AoWeno53Prim<Dim>& rhs) {
  return not(lhs == rhs);
}
}  // namespace NewtonianEuler::fd
