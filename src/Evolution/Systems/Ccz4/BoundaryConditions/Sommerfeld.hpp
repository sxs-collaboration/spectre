// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace Ccz4::BoundaryConditions {
/*!
 * \brief Sets Sommerfeld boundary conditions per tensor component.
 *
 * Unlike time-independent subcell external boundary conditions,
 * the Sommerfeld boundary condition is applied at the outermost
 * interior points in the volume instead of the ghost zone.
 * This is because the Sommerfeld condition requires radial derivatives
 * of the fields and modify their time derivatives, which then need to
 * be time integrated. Under current infrastructure, it is therefore
 * the simplest to apply this boundary condition in the interior
 * rather than the ghost zone. This file extrapolates the interior
 * evolved variables into the ghost zone. Their radial derivatives and
 * time derivatives are then computed and applied in SoTimeDerivative.hpp for
 * the outermost interior points. All tensor components (including lapse and
 * shift) are treated as independent fields when applying this boundary
 * condition.
 *
 * \warning This boundary condition assumes a complete sphere domain
 * (all wedges), as we only apply it on the upper_zeta
 * direction in blocks with external boundaries.
 */
class Sommerfeld final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Sommerfeld boundary conditions applied per tensor component."};

  Sommerfeld() = default;
  Sommerfeld(Sommerfeld&&) = default;
  Sommerfeld& operator=(Sommerfeld&&) = default;
  Sommerfeld(const Sommerfeld&);
  Sommerfeld& operator=(const Sommerfeld&);
  ~Sommerfeld() override = default;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Sommerfeld);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using fd_interior_evolved_variables_tags =
      ::Ccz4::fd::System::variables_tag_list;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags = tmpl::list<::Ccz4::fd::Tags::Reconstructor>;
  static void fd_ghost(
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> conformal_metric,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> auxiliary_shift_b,
      const Direction<3>& direction,

      // fd_interior_evolved_variables_tags (variables_tag_list order)
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
      const Scalar<DataVector>& interior_conformal_factor,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
      const Scalar<DataVector>& interior_trace_extrinsic_curvature,
      const Scalar<DataVector>& interior_theta,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,

      // fd_interior_temporary_tags
      const Mesh<3>& subcell_mesh,

      // fd_gridless_tags
      const fd::Reconstructor& reconstructor);
};
}  // namespace Ccz4::BoundaryConditions
