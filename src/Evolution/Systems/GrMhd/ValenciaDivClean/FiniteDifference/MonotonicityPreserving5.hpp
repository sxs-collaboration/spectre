// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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
 * \brief Fifth order monotonicity-preserving (MP5) reconstruction. See
 * ::fd::reconstruction::monotonicity_preserving_5() for details.
 *
 */
class MonotonicityPreserving5Prim : public Reconstructor {
 private:
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

 public:
  static constexpr size_t dim = 3;

  struct Alpha {
    using type = double;
    static constexpr Options::String help = {
        "The parameter used in an intermediate reconstruction step to impose "
        "monotonicity; typically Alpha=4.0 is used. Note that in principle the "
        "CFL number must be not bigger than 1/(1+Alpha). See the original text "
        "Suresh & Huynh (1997) for the details"};
  };
  struct Epsilon {
    using type = double;
    static constexpr Options::String help = {
        "A small tolerance value by which limiting process is turned on and "
        "off. Suresh & Huynh (1997) suggests 1e-10, but for hydro simulations "
        "with atmosphere treatment setting Epsilon=0.0 would be safe."};
  };

  using options = tmpl::list<Alpha, Epsilon>;
  static constexpr Options::String help{
      "MP5 reconstruction scheme using primitive variables."};

  MonotonicityPreserving5Prim() = default;
  MonotonicityPreserving5Prim(MonotonicityPreserving5Prim&&) = default;
  MonotonicityPreserving5Prim& operator=(MonotonicityPreserving5Prim&&) =
      default;
  MonotonicityPreserving5Prim(const MonotonicityPreserving5Prim&) = default;
  MonotonicityPreserving5Prim& operator=(const MonotonicityPreserving5Prim&) =
      default;
  ~MonotonicityPreserving5Prim() override = default;

  MonotonicityPreserving5Prim(double alpha, double epsilon);

  explicit MonotonicityPreserving5Prim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor,
                                     MonotonicityPreserving5Prim);

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
  friend bool operator==(const MonotonicityPreserving5Prim& lhs,
                         const MonotonicityPreserving5Prim& rhs);
  friend bool operator!=(const MonotonicityPreserving5Prim& lhs,
                         const MonotonicityPreserving5Prim& rhs);

  double alpha_ = std::numeric_limits<double>::signaling_NaN();
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();
};

}  // namespace grmhd::ValenciaDivClean::fd
