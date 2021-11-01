// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
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
namespace Tags {
template <typename TagsList>
class Variables;
}  // namespace Tags
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarAdvection::fd {
/*!
 * \brief Adaptive-order WENO reconstruction hybridizing orders 5 and 3. See
 * ::fd::reconstruction::aoweno_53() for details.
 */
template <size_t Dim>
class AoWeno53 : public Reconstructor<Dim> {
 private:
  using face_vars_tags =
      tmpl::list<Tags::U,
                 ::Tags::Flux<Tags::U, tmpl::size_t<Dim>, Frame::Inertial>>;
  using volume_vars_tags = tmpl::list<Tags::U>;

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
      "Adaptive-order WENO reconstruction hybridizing orders 5 and 3."};

  AoWeno53() = default;
  AoWeno53(AoWeno53&&) = default;
  AoWeno53& operator=(AoWeno53&&) = default;
  AoWeno53(const AoWeno53&) = default;
  AoWeno53& operator=(const AoWeno53&) = default;
  ~AoWeno53() override = default;

  AoWeno53(double gamma_hi, double gamma_lo, double epsilon,
           size_t nonlinear_weight_exponent);

  explicit AoWeno53(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>, AoWeno53);

  auto get_clone() const -> std::unique_ptr<Reconstructor<Dim>> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

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

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const AoWeno53<LocalDim>& lhs,
                         const AoWeno53<LocalDim>& rhs);

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
bool operator==(const AoWeno53<Dim>& lhs, const AoWeno53<Dim>& rhs);

template <size_t Dim>
bool operator!=(const AoWeno53<Dim>& lhs, const AoWeno53<Dim>& rhs);
}  // namespace ScalarAdvection::fd
