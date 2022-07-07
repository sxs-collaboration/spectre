// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/// A `BoundaryCondition` that only verifies that all characteristic speeds are
/// directed out of the domain; no boundary data is altered by this boundary
/// condition.
class Outflow final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Outflow boundary condition that only verifies the characteristic speeds "
      "are all directed out of the domain."};

  Outflow() = default;
  Outflow(Outflow&&) = default;
  Outflow& operator=(Outflow&&) = default;
  Outflow(const Outflow&) = default;
  Outflow& operator=(const Outflow&) = default;
  ~Outflow() override = default;

  explicit Outflow(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Outflow);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Outflow;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::Lapse<DataVector>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
      /*outward_directed_normal_vector*/,

      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const Scalar<DataVector>& lapse);

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>,
                 gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::Lapse<DataVector>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;
  using fd_gridless_tags = tmpl::list<fd::Tags::Reconstructor>;

  static void fd_outflow(
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      const gsl::not_null<Scalar<DataVector>*> electron_fraction,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          magnetic_field,
      const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      const Direction<3>& direction,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,

      // fd_interior_temporary_tags
      const Mesh<3>& subcell_mesh,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const Scalar<DataVector>& lapse,

      // fd_interior_primitive_variables_tags
      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_divergence_cleaning_field,

      // fd_gridless_tags
      const fd::Reconstructor& reconstructor);
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
