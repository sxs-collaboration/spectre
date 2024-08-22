// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
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
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief Positivity-preserving adaptive order reconstruction. See
 * ::fd::reconstruction::positivity_preserving_adaptive_order() for details.
 * The rest mass density, electron fraction, and the pressure are kept positive.
 * Use unlimited 5th order (degree 4 polynomial) reconstruction on the
 * metric variables.
 *
 * Only the spacetime metric is reconstructed when we and the neighboring
 * element in the direction are doing FD. If we are doing DG and a neighboring
 * element is doing FD, then the spacetime metric, \f$\Phi_{iab}\f$, and
 * \f$\Pi_{ab}\f$ are all reconstructed since the Riemann solver on the DG
 * element also needs to solve for the metric variables.
 */
class PositivityPreservingAdaptiveOrderPrim : public Reconstructor {
 private:
  using positivity_preserving_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;
  using non_positive_tags =
      tmpl::list<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

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

  static constexpr Options::String help{
      "Positivity-preserving adaptive-order reconstruction."};

  PositivityPreservingAdaptiveOrderPrim() = default;
  PositivityPreservingAdaptiveOrderPrim(
      PositivityPreservingAdaptiveOrderPrim&&) = default;
  PositivityPreservingAdaptiveOrderPrim& operator=(
      PositivityPreservingAdaptiveOrderPrim&&) = default;
  PositivityPreservingAdaptiveOrderPrim(
      const PositivityPreservingAdaptiveOrderPrim&) = default;
  PositivityPreservingAdaptiveOrderPrim& operator=(
      const PositivityPreservingAdaptiveOrderPrim&) = default;
  ~PositivityPreservingAdaptiveOrderPrim() override = default;

  PositivityPreservingAdaptiveOrderPrim(
      double alpha_5, std::optional<double> alpha_7,
      std::optional<double> alpha_9,
      FallbackReconstructorType low_order_reconstructor,
      const Options::Context& context = {});

  explicit PositivityPreservingAdaptiveOrderPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor,
                                     PositivityPreservingAdaptiveOrderPrim);

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
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                 typename System::variables_tag,
                 hydro::Tags::GrmhdEquationOfState, domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>,
                 ::Tags::VariableFixer<VariableFixing::FixToAtmosphere<dim>>>;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, dim>*> vars_on_upper_face,
      gsl::not_null<std::optional<std::array<gsl::span<std::uint8_t>, dim>>*>
          reconstruction_order,
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
      const Variables<typename System::variables_tag::type::tags_list>&
          volume_spacetime_and_cons_vars,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<dim>& subcell_mesh,
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const;

  /// Called by an element doing DG when the neighbor is doing subcell.
  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
      const Variables<
          grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>&
          subcell_volume_spacetime_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<dim>& subcell_mesh,
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,
      const Direction<dim>& direction_to_reconstruct) const;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const PositivityPreservingAdaptiveOrderPrim& lhs,
                         const PositivityPreservingAdaptiveOrderPrim& rhs);

  friend bool operator!=(const PositivityPreservingAdaptiveOrderPrim& lhs,
                         const PositivityPreservingAdaptiveOrderPrim& rhs);

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

}  // namespace grmhd::GhValenciaDivClean::fd
