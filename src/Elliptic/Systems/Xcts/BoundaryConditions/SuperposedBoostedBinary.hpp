// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <brigand/brigand.hpp>

#include <array>
#include <memory>
#include <optional>
#include <pup.h>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {

/*!
 * \brief Impose supperposed boosted binary system on the boundary.
 *
 * This takes two isolated objects and after applying a boost to each of them,
 * superposes them. The superposed system is then imposed on the boundary, with
 * Dirichlet boundary conditions.
 *
 */

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
class SuperposedBoostedBinary
    : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  struct XCoords {
    static constexpr Options::String help =
        "The coordinates on the x-axis where the two objects are placed.";
    using type = std::array<double, 2>;
  };
  struct Masses {
    static constexpr Options::String help =
        "The masses of each object, first left and second right";
    using type = std::array<double, 2>;
  };
  struct MomentumLeft {
    static constexpr Options::String help =
        "The momentum assigned to the left object.";
    using type = std::array<double, 3>;
  };
  struct MomentumRight {
    static constexpr Options::String help =
        "The momentum assigned to the right object.";
    using type = std::array<double, 3>;
  };
  struct CenterOfMassOffset {
    static constexpr Options::String help = {
        "Offset in the y and z axes applied to both objects in order to "
        "control the center of mass."};
    using type = std::array<double, 2>;
  };
  struct ObjectLeft {
    static constexpr Options::String help =
        "The object placed on the negative x-axis.";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct ObjectRight {
    static constexpr Options::String help =
        "The object placed on the positive x-axis.";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };

  using options = tmpl::list<XCoords, Masses, MomentumLeft, MomentumRight,
                             CenterOfMassOffset, ObjectLeft, ObjectRight>;
  static constexpr Options::String help =
      "Impose supperposed boosted binary system on the boundary.";

  SuperposedBoostedBinary() = default;
  SuperposedBoostedBinary(const SuperposedBoostedBinary&) = delete;
  SuperposedBoostedBinary& operator=(const SuperposedBoostedBinary&) = delete;
  SuperposedBoostedBinary(SuperposedBoostedBinary&&) = default;
  SuperposedBoostedBinary& operator=(SuperposedBoostedBinary&&) = default;
  ~SuperposedBoostedBinary() override = default;

  /// \cond
  explicit SuperposedBoostedBinary(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SuperposedBoostedBinary);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    const std::array<double, 2> center_of_mass_offset = {
        {y_offset_, z_offset_}};
    return std::make_unique<SuperposedBoostedBinary>(
        xcoords_, masses_, momentum_left_, momentum_right_,
        center_of_mass_offset,
        superposed_objects_[0].has_value()
            ? std::make_optional(
                  deserialize<std::unique_ptr<IsolatedObjectBase>>(
                      serialize(superposed_objects_[0].value()).data()))
            : std::nullopt,
        superposed_objects_[1].has_value()
            ? std::make_optional(
                  deserialize<std::unique_ptr<IsolatedObjectBase>>(
                      serialize(superposed_objects_[1].value()).data()))
            : std::nullopt);
  }

  SuperposedBoostedBinary(
      const std::array<double, 2> xcoords, const std::array<double, 2> masses,
      const std::array<double, 3> momentum_left,
      const std::array<double, 3> momentum_right,
      const std::array<double, 2> center_of_mass_offset,
      std::optional<std::unique_ptr<IsolatedObjectBase>> object_left,
      std::optional<std::unique_ptr<IsolatedObjectBase>> object_right,
      const Options::Context& context = {})
      : xcoords_(xcoords),
        masses_(masses),
        momentum_left_(momentum_left),
        momentum_right_(momentum_right),
        y_offset_(center_of_mass_offset[0]),
        z_offset_(center_of_mass_offset[1]),
        superposed_objects_({std::move(object_left), std::move(object_right)}) {
    if (masses_[0] <= 0. || masses_[1] <= 0.) {
      PARSE_ERROR(context, "The masses must be positive.");
    }
    if (xcoords_[0] >= xcoords_[1]) {
      PARSE_ERROR(context, "Specify 'XCoords' ascending from left to right.");
    }
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {// Conformal factor
            elliptic::BoundaryConditionType::Dirichlet,
            // Lapse times conformal factor
            elliptic::BoundaryConditionType::Dirichlet,
            // Shift
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags =
      tmpl::flatten<tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::I<DataVector, 3>& x) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Base::pup(p);
    p | xcoords_;
    p | masses_;
    p | y_offset_;
    p | z_offset_;
    p | momentum_left_;
    p | momentum_right_;
    p | superposed_objects_;
  }

 private:
  std::array<double, 2> xcoords_{};
  std::array<double, 2> masses_{};
  std::array<double, 3> momentum_left_{};
  std::array<double, 3> momentum_right_{};
  double y_offset_{};
  double z_offset_{};
  std::array<std::optional<std::unique_ptr<IsolatedObjectBase>>, 2>
      superposed_objects_{};
};

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
bool operator==(const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& lhs,
                const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& rhs);

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
bool operator!=(const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& lhs,
                const SuperposedBoostedBinary<IsolatedObjectBase,
                                              IsolatedObjectClasses>& rhs);

/// \cond
template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
PUP::able::PUP_ID
    SuperposedBoostedBinary<IsolatedObjectBase,
                            IsolatedObjectClasses>::my_PUP_ID =  // NOLINT
    0;                                                           // NOLINT
/// \endcond

}  // namespace Xcts::BoundaryConditions
