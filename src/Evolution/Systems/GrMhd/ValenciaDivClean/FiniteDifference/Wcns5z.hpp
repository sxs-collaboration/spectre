// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Options/Options.hpp"
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
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

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

  using options = tmpl::list<NonlinearWeightExponent, Epsilon>;

  static constexpr Options::String help{
      "WCNS 5Z reconstruction scheme using primitive variables."};

  Wcns5zPrim() = default;
  Wcns5zPrim(Wcns5zPrim&&) = default;
  Wcns5zPrim& operator=(Wcns5zPrim&&) = default;
  Wcns5zPrim(const Wcns5zPrim&) = default;
  Wcns5zPrim& operator=(const Wcns5zPrim&) = default;
  ~Wcns5zPrim() override = default;

  Wcns5zPrim(size_t nonlinear_weight_exponent, double epsilon);

  explicit Wcns5zPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, Wcns5zPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

  using reconstruction_argument_tags = tmpl::list<
      ::Tags::Variables<hydro::grmhd_tags<DataVector>>,
      hydro::Tags::EquationOfStateBase, domain::Tags::Element<dim>,
      evolution::dg::subcell::Tags::NeighborDataForReconstruction<dim>,
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
          std::pair<Direction<dim>, ElementId<dim>>, std::vector<double>,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
          neighbor_data,
      const Mesh<dim>& subcell_mesh) const;

  template <size_t ThermodynamicDim, typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Element<dim>& element,
      const FixedHashMap<
          maximum_number_of_neighbors(dim),
          std::pair<Direction<dim>, ElementId<dim>>, std::vector<double>,
          boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
          neighbor_data,
      const Mesh<dim>& subcell_mesh,
      const Direction<dim> direction_to_reconstruct) const;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Wcns5zPrim& lhs, const Wcns5zPrim& rhs);
  friend bool operator!=(const Wcns5zPrim& lhs, const Wcns5zPrim& rhs);

  size_t nonlinear_weight_exponent_ = 0;
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();

  void (*reconstruct_)(gsl::not_null<std::array<gsl::span<double>, dim>*>,
                       gsl::not_null<std::array<gsl::span<double>, dim>*>,
                       const gsl::span<const double>&,
                       const DirectionMap<dim, gsl::span<const double>>&,
                       const Index<dim>&, size_t, double) = nullptr;
  void (*reconstruct_lower_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<dim>&, const Index<dim>&,
                                      const Direction<dim>&,
                                      const double&) = nullptr;
  void (*reconstruct_upper_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<dim>&, const Index<dim>&,
                                      const Direction<dim>&,
                                      const double&) = nullptr;
};

}  // namespace grmhd::ValenciaDivClean::fd
