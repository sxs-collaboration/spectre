// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems//BnsInitialData/Tags.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace BnsInitialData::BoundaryConditions {

/*!
 * Impose StarSurface boundary conditions:
 * \f[
 *   n_i F^i = \frac{C}{\alpha^2} B^i n_i.
 * \f]
 * The boundary condition results from requiring the conservation
 * equations be regular at the surface of the neutron star.  See
 * \cite BaumgarteShapiro 15.79.
 */
class StarSurface : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help =
      "StarSurface boundary conditions  n_i F^i = C/square(alpha) B^i n_i .";

  using options = tmpl::list<>;

  StarSurface() = default;
  StarSurface(const StarSurface&) = default;
  StarSurface& operator=(const StarSurface&) = default;
  StarSurface(StarSurface&&) = default;
  StarSurface& operator=(StarSurface&&) = default;
  ~StarSurface() override = default;

  /// \cond
  explicit StarSurface(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(StarSurface);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<StarSurface>(*this);
  }

  explicit StarSurface(const Options::Context& context);

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 BnsInitialData::Tags::RotationalShift<DataVector>,
                 BnsInitialData::Tags::EulerEnthalpyConstant,
                 domain::Tags::FaceNormal<3>>;
  using volume_tags = tmpl::list<BnsInitialData::Tags::EulerEnthalpyConstant>;

  static void apply(gsl::not_null<Scalar<DataVector>*> velocity_potential,
                    gsl::not_null<Scalar<DataVector>*> n_dot_flux_for_potential,
                    const tnsr::i<DataVector, 3>& deriv_velocity_potential,
                    const Scalar<DataVector>& lapse,
                    const tnsr::I<DataVector, 3>& rotational_shift,
                    double euler_enthalpy_constant,
                    const tnsr::i<DataVector, 3>& normal);

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> velocity_potential_correction,
      gsl::not_null<Scalar<DataVector>*> n_dot_flux_for_potential_correction,
      const tnsr::i<DataVector, 3>& deriv_velocity_potential);

  void pup(PUP::er& p) override;
};

bool operator==(const StarSurface& lhs, const StarSurface& rhs);

bool operator!=(const StarSurface& lhs, const StarSurface& rhs);

}  // namespace BnsInitialData::BoundaryConditions
