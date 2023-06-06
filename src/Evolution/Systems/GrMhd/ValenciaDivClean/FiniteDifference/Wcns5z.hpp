// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
template <typename TagsList>
class Variables;
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Fifth order weighted nonlinear compact scheme reconstruction using the
 * Z oscillation indicator. See ::fd::reconstruction::wcns5z() for details.
 *
 */
class Wcns5zPrim : public Reconstructor {
 private:
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

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

  static constexpr Options::String help{
      "WCNS 5Z reconstruction scheme using primitive variables."};

  Wcns5zPrim() = default;
  Wcns5zPrim(Wcns5zPrim&&) = default;
  Wcns5zPrim& operator=(Wcns5zPrim&&) = default;
  Wcns5zPrim(const Wcns5zPrim&) = default;
  Wcns5zPrim& operator=(const Wcns5zPrim&) = default;
  ~Wcns5zPrim() override = default;

  Wcns5zPrim(size_t nonlinear_weight_exponent, double epsilon,
             FallbackReconstructorType fallback_reconstructor,
             size_t max_number_of_extrema);

  explicit Wcns5zPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, Wcns5zPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  static constexpr bool use_adaptive_order = false;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                 hydro::Tags::EquationOfStateBase, domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>>;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, dim>*> vars_on_upper_face,
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(dim),
          std::pair<Direction<dim>, ElementId<dim>>,
          evolution::dg::subcell::GhostData,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>& ghost_data,
      const Mesh<dim>& subcell_mesh) const;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
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
  friend bool operator==(const Wcns5zPrim& lhs, const Wcns5zPrim& rhs);
  friend bool operator!=(const Wcns5zPrim& lhs, const Wcns5zPrim& rhs);

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

}  // namespace grmhd::ValenciaDivClean::fd
