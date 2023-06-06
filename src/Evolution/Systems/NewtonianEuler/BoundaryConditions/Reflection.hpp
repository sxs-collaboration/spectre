// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler::BoundaryConditions {
/*!
 * \brief Reflecting boundary conditions for Newtonian hydrodynamics.
 *
 * Ghost (exterior) data 'mirrors' interior volume data with respect to the
 * boundary interface. i.e. reverses the normal component of velocity while
 * using same values for other scalar quantities.
 *
 * In the frame instantaneously moving with the same velocity as face mesh, the
 * reflection condition reads
 *
 * \f{align*}
 * \vec{u}_\text{ghost} = \vec{u}_\text{int} - 2 (\vec{u}_\text{int} \cdot
 * \hat{n}) \hat{n}
 * \f}
 *
 * where \f$\vec{u}\f$ is the fluid velocity in the moving frame, "int" stands
 * for interior, and \f$\hat{n}\f$ is the outward normal vector on the boundary
 * interface.
 *
 * Substituting \f$\vec{u} = \vec{v} - \vec{v}_m\f$, we get
 *
 * \f{align*}
 * v_\text{ghost}^i &= v_\text{int}^i - 2[(v_\text{int}^j-v_m^j)n_j]n^i
 * \f}
 *
 * where \f$v\f$ is the fluid velocity and \f$v_m\f$ is face mesh velocity.
 *
 */
template <size_t Dim>
class Reflection final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Reflecting boundary conditions for Newtonian hydrodynamics."};

  Reflection() = default;
  Reflection(Reflection&&) = default;
  Reflection& operator=(Reflection&&) = default;
  Reflection(const Reflection&) = default;
  Reflection& operator=(const Reflection&) = default;
  ~Reflection() override = default;

  explicit Reflection(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Reflection);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, Dim>,
                 Tags::SpecificInternalEnergy<DataVector>,
                 Tags::Pressure<DataVector>>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          momentum_density,
      const gsl::not_null<Scalar<DataVector>*> energy_density,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_mass_density,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_momentum_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_energy_density,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const Scalar<DataVector>& interior_mass_desity,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_pressure) const;
};

}  // namespace NewtonianEuler::BoundaryConditions
