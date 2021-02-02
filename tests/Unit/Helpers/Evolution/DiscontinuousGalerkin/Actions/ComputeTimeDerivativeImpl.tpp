// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeImpl.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::evolution::dg::Actions {
inline std::ostream& operator<<(std::ostream& os,
                                const UseBoundaryCorrection t) noexcept {
  return os << (t == UseBoundaryCorrection::Yes ? std::string{"Yes"}
                                                : std::string{"No"});
}

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

// Var3 is used as an extra quantity in the DataBox that the time derivative
// computation depends on. It could be loosely interpreted as a "primitive"
// variable, or a compute tag retrieved for the time derivative computation.
struct Var3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PrimVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct PrimVar2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct PrimVarsCompute : db::ComputeTag,
                         Tags::Variables<tmpl::list<PrimVar1, PrimVar2<Dim>>> {
  using base = Tags::Variables<tmpl::list<PrimVar1, PrimVar2<Dim>>>;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> result,
                       const Mesh<Dim>& mesh) noexcept {
    result->initialize(mesh.number_of_grid_points());
    get(get<PrimVar1>(*result)) = 5.0;
    for (size_t i = 0; i < Dim; ++i) {
      get<PrimVar2<Dim>>(*result).get(i) = 7.0 + i;
    }
  }
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;
};

struct Var3Squared : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PackagedVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct PackagedVar2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct SimpleUnnormalizedFaceNormal
    : db::ComputeTag,
      domain::Tags::UnnormalizedFaceNormal<Dim> {
  using base = domain::Tags::UnnormalizedFaceNormal<Dim>;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> result,
                       const Mesh<Dim - 1>& face_mesh,
                       const Direction<Dim>& direction) noexcept {
    for (size_t i = 0; i < Dim; ++i) {
      result->get(i) =
          DataVector{face_mesh.number_of_grid_points(),
                     i == direction.dimension() ? 1.0 * i + 0.25 : 0.0};
    }
    const auto mag = magnitude(*result);
    for (size_t i = 0; i < Dim; ++i) {
      result->get(i) /= get(mag);
    }
  }
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim - 1>, domain::Tags::Direction<Dim>>;
};

template <size_t Dim, bool UseMovingMesh>
struct FaceMeshVelocity : db::ComputeTag, domain::Tags::MeshVelocity<Dim> {
  using base = domain::Tags::MeshVelocity<Dim>;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> result,
                       const Mesh<Dim - 1>& face_mesh,
                       const Direction<Dim>& direction) noexcept {
    if constexpr (UseMovingMesh) {
      tnsr::I<DataVector, Dim> mesh_velocity{face_mesh.number_of_grid_points()};
      for (size_t i = 0; i < Dim; ++i) {
        mesh_velocity.get(i) =
            (i == direction.dimension() ? 2.0 * i + 0.5 : 0.0);
      }
      *result = std::move(mesh_velocity);
    } else {
      *result = return_type{};
    }
  }
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim - 1>, domain::Tags::Direction<Dim>>;
};

template <size_t Dim, SystemType system_type, bool HasPrimitiveVars>
struct TimeDerivativeTerms {
  /// [dt_ta]
  using temporary_tags = tmpl::list<Var3Squared>;
  using common_argument_tags = tmpl::list<Var1, Var2<Dim>, Var3>;
  using argument_tags =
      tmpl::conditional_t<HasPrimitiveVars,
                          tmpl::push_back<common_argument_tags, PrimVar1>,
                          common_argument_tags>;
  /// [dt_ta]

  // Conservative system
  /// [dt_con]
  static void apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> flux_var1,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_var2,

      // Temporaries returned by reference. Listed in temporary_tags above.
      const gsl::not_null<Scalar<DataVector>*> square_var3,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3) noexcept {
    get(*square_var3) = square(get(var3));

    // Set source terms
    get(*dt_var1) = get(*square_var3);
    for (size_t d = 0; d < Dim; ++d) {
      dt_var2->get(d) = get(var3) * d;
    }

    // Set fluxes
    for (size_t i = 0; i < Dim; ++i) {
      flux_var1->get(i) = square(get(var1)) * var2.get(i);
      for (size_t j = 0; j < Dim; ++j) {
        flux_var2->get(i, j) = var2.get(i) * var2.get(j) * get(var1);
        if (i == j) {
          flux_var2->get(i, j) += cube(get(var1));
        }
      }
    }
  }
  /// [dt_con]
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> flux_var1,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_var2,

      const gsl::not_null<Scalar<DataVector>*> square_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3,
      const Scalar<DataVector>& prim_var1) noexcept {
    apply(dt_var1, dt_var2, flux_var1, flux_var2, square_var3, var1, var2,
          var3);
    get(*dt_var1) += get(prim_var1);
  }

  // Nonconservative system
  /// [dt_nc]
  static void apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      // Temporaries returned by reference. Listed in temporary_tags above.
      const gsl::not_null<Scalar<DataVector>*> square_var3,

      // Partial derivative arguments. Listed in the system struct as
      // gradient_variables.
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_var1,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_var2,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3) noexcept {
    get(*square_var3) = square(get(var3));

    // Set source terms and nonconservative products
    get(*dt_var1) = get(*square_var3);
    for (size_t d = 0; d < Dim; ++d) {
      get(*dt_var1) -= var2.get(d) * d_var1.get(d);
      dt_var2->get(d) = get(var3) * d;
      for (size_t i = 0; i < Dim; ++i) {
        dt_var2->get(d) -= get(var1) * var2.get(i) * d_var2.get(i, d);
      }
    }
  }
  /// [dt_nc]

  // Mixed system
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_var2,

      const gsl::not_null<Scalar<DataVector>*> square_var3,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_var1,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3) noexcept {
    get(*square_var3) = square(get(var3));

    // Set source terms and nonconservative products
    get(*dt_var1) = get(*square_var3);
    for (size_t d = 0; d < Dim; ++d) {
      get(*dt_var1) -= var2.get(d) * d_var1.get(d);
      dt_var2->get(d) = get(var3) * d;
    }

    // Set fluxes
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        flux_var2->get(i, j) = var2.get(i) * var2.get(j) * get(var1);
        if (i == j) {
          flux_var2->get(i, j) += cube(get(var1));
        }
      }
    }
  }
  /// [dt_mp]
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_var2,

      const gsl::not_null<Scalar<DataVector>*> square_var3,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_var1,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3,
      const Scalar<DataVector>& prim_var1) noexcept {
    apply(dt_var1, dt_var2, flux_var2, square_var3, d_var1, var1, var2, var3);
    get(*dt_var1) += get(prim_var1);
  }
  /// [dt_mp]
};

template <size_t Dim>
struct NonconservativeNormalDotFlux {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> var1_normal_dot_flux,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          var2_normal_dot_flux) noexcept {
    get(*var1_normal_dot_flux) = 1.1;
    for (size_t i = 0; i < Dim; ++i) {
      var2_normal_dot_flux->get(i) = 1.3 + i;
    }
  }
};

template <size_t Dim, bool HasPrims>
struct BoundaryTerms;

template <size_t Dim, bool HasPrims>
class BoundaryCorrection : public PUP::able {
 public:
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;

  ~BoundaryCorrection() override = default;

  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT

  using creatable_classes = tmpl::list<BoundaryTerms<Dim, HasPrims>>;
};

template <size_t Dim, bool HasPrims>
struct BoundaryTerms final : tt::ConformsTo<::dg::protocols::NumericalFlux>,
                             public BoundaryCorrection<Dim, HasPrims> {
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  /// \cond
  explicit BoundaryTerms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoundaryTerms);  // NOLINT
  /// \endcond
  BoundaryTerms() = default;
  BoundaryTerms(const BoundaryTerms&) = default;
  BoundaryTerms& operator=(const BoundaryTerms&) = default;
  BoundaryTerms(BoundaryTerms&&) = default;
  BoundaryTerms& operator=(BoundaryTerms&&) = default;
  ~BoundaryTerms() override = default;

  using variables_tags = tmpl::list<Var1, Var2<Dim>>;
  using variables_tag = Tags::Variables<variables_tags>;

  using package_field_tags = tmpl::push_back<
      tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                   variables_tags>,
      MaxAbsCharSpeed>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags = tmpl::push_back<tmpl::append<
      db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>, variables_tags>>;

  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection<Dim, HasPrims>::pup(p);
  }

  void package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const Scalar<DataVector>& normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_dot_flux_var2,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2) const noexcept {
    *out_normal_dot_flux_var1 = normal_dot_flux_var1;
    *out_normal_dot_flux_var2 = normal_dot_flux_var2;
    *out_var1 = var1;
    *out_var2 = var2;
    get(*max_abs_char_speed) = 2.0 * max(get(var1));
  }

  /// [bt_nnv]
  static constexpr bool need_normal_vector = false;
  /// [bt_nnv]

  /// [bt_ta]
  using dg_package_field_tags = tmpl::push_back<
      tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                   variables_tags>,
      MaxAbsCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<Var3Squared>;
  using dg_package_data_primitive_tags =
      tmpl::conditional_t<HasPrims, tmpl::list<PrimVar1>, tmpl::list<>>;
  using dg_package_data_volume_tags =
      tmpl::conditional_t<HasPrims, tmpl::list<Tags::TimeStepId>, tmpl::list<>>;
  /// [bt_ta]

  // Conservative system
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    *out_normal_dot_flux_var1 = dot_product(flux_var1, normal_covector);
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          get(var1) * get(dot_product(*mesh_velocity, normal_covector));
    }
    for (size_t i = 0; i < Dim; ++i) {
      out_normal_dot_flux_var2->get(i) =
          flux_var2.get(i, 0) * normal_covector.get(0);
      if (mesh_velocity.has_value()) {
        out_normal_dot_flux_var2->get(i) -=
            var2.get(i) * get<0>(*mesh_velocity) * normal_covector.get(0);
      }
      for (size_t j = 1; j < Dim; ++j) {
        out_normal_dot_flux_var2->get(i) +=
            flux_var2.get(i, j) * normal_covector.get(j);
        if (mesh_velocity.has_value()) {
          out_normal_dot_flux_var2->get(i) -=
              var2.get(i) * mesh_velocity->get(j) * normal_covector.get(j);
        }
      }
    }
    *out_var1 = var1;
    *out_var2 = var2;

    get(*max_abs_char_speed) = 2.0 * max(get(var3_squared));

    if (normal_dot_mesh_velocity.has_value()) {
      get(*max_abs_char_speed) += get(*normal_dot_mesh_velocity);
    }
    return max(get(*max_abs_char_speed));
  }

  // Conservative system with prim vars
  /// [bt_cp]
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,
      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const TimeStepId& time_step_id) const noexcept {
    dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                    out_var1, out_var2, max_abs_char_speed, var1, var2,
                    flux_var1, flux_var2, var3_squared, normal_covector,
                    mesh_velocity, normal_dot_mesh_velocity);
    get(*out_var1) += get(prim_var1) + time_step_id.step_time().value();
    return max(get(*max_abs_char_speed));
  }
  /// [bt_cp]

  // Nonconservative system
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    get(*out_normal_dot_flux_var1) =
        get(var1) + get(dot_product(var2, normal_covector));
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          get(dot_product(*mesh_velocity, normal_covector));
    }
    for (size_t i = 0; i < Dim; ++i) {
      out_normal_dot_flux_var2->get(i) =
          normal_covector.get(i) * normal_covector.get(0) * var2.get(0);
      if (mesh_velocity.has_value()) {
        out_normal_dot_flux_var2->get(i) -=
            var2.get(i) * get<0>(*mesh_velocity) * normal_covector.get(0);
      }
      for (size_t j = 1; j < Dim; ++j) {
        out_normal_dot_flux_var2->get(i) +=
            normal_covector.get(i) * normal_covector.get(j) * var2.get(j);
        if (mesh_velocity.has_value()) {
          out_normal_dot_flux_var2->get(i) -=
              var2.get(i) * mesh_velocity->get(j) * normal_covector.get(j);
        }
      }
    }
    *out_var1 = var1;
    *out_var2 = var2;

    get(*max_abs_char_speed) = 2.0 * max(get(var3_squared));

    if (normal_dot_mesh_velocity.has_value()) {
      get(*max_abs_char_speed) += get(*normal_dot_mesh_velocity);
    }
    return max(get(*max_abs_char_speed));
  }

  // Mixed system
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    get(*out_normal_dot_flux_var1) =
        get(var1) + get(dot_product(var2, normal_covector));
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          get(dot_product(*mesh_velocity, normal_covector));
    }
    for (size_t i = 0; i < Dim; ++i) {
      out_normal_dot_flux_var2->get(i) =
          flux_var2.get(i, 0) * normal_covector.get(0);
      if (mesh_velocity.has_value()) {
        out_normal_dot_flux_var2->get(i) -=
            var2.get(i) * get<0>(*mesh_velocity) * normal_covector.get(0);
      }
      for (size_t j = 1; j < Dim; ++j) {
        out_normal_dot_flux_var2->get(i) +=
            flux_var2.get(i, j) * normal_covector.get(j);
        if (mesh_velocity.has_value()) {
          out_normal_dot_flux_var2->get(i) -=
              var2.get(i) * mesh_velocity->get(j) * normal_covector.get(j);
        }
      }
    }
    *out_var1 = var1;
    *out_var2 = var2;

    get(*max_abs_char_speed) = 2.0 * max(get(var3_squared));

    if (normal_dot_mesh_velocity.has_value()) {
      get(*max_abs_char_speed) += get(*normal_dot_mesh_velocity);
    }
    return max(get(*max_abs_char_speed));
  }

  // Mixed system with prims
  /// [bt_mp]
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const TimeStepId& time_step_id) const noexcept {
    dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                    out_var1, out_var2, max_abs_char_speed, var1, var2,
                    flux_var2, var3_squared, normal_covector, mesh_velocity,
                    normal_dot_mesh_velocity);
    get(*out_var1) += get(prim_var1) + time_step_id.step_time().value();
    return max(get(*max_abs_char_speed));
  }
  /// [bt_mp]
};

template <size_t Dim, bool HasPrims>
PUP::able::PUP_ID BoundaryTerms<Dim, HasPrims>::my_PUP_ID = 0;

// We only use an Outflow boundary condition with a static member variable that
// we set to verify the call went through. The actual boundary condition
// implementation is tested elsewhere.
template <size_t Dim>
class Outflow;

template <size_t Dim>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes = tmpl::list<Outflow<Dim>>;

  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};

template <size_t Dim>
class Outflow : public BoundaryCondition<Dim> {
 public:
  Outflow() = default;
  Outflow(Outflow&&) noexcept = default;
  Outflow& operator=(Outflow&&) noexcept = default;
  Outflow(const Outflow&) = default;
  Outflow& operator=(const Outflow&) = default;
  ~Outflow() override = default;

  explicit Outflow(CkMigrateMessage* msg) noexcept
      : BoundaryCondition<Dim>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Outflow);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<Outflow<Dim>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::Outflow;

  void pup(PUP::er& p) override { BoundaryCondition<Dim>::pup(p); }

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
      /*outward_directed_normal_covector*/) noexcept {
    Outflow<Dim>::number_of_times_called += 1;
    return std::nullopt;
  }

  static size_t number_of_times_called;
};

template <size_t Dim>
PUP::able::PUP_ID Outflow<Dim>::my_PUP_ID = 0;

template <size_t Dim>
size_t Outflow<Dim>::number_of_times_called = 0;

struct NoBoundaryCorrection {};
template <size_t Dim, bool HasPrims>
struct HaveBoundaryCorrection {
  using boundary_correction = BoundaryCorrection<Dim, HasPrims>;
};

template <size_t Dim, SystemType system_type,
          UseBoundaryCorrection use_boundary_correction,
          bool HasPrimitiveVariables>
struct System : public tmpl::conditional_t<
                    use_boundary_correction == UseBoundaryCorrection::Yes,
                    HaveBoundaryCorrection<Dim, HasPrimitiveVariables>,
                    NoBoundaryCorrection> {
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveVariables;
  static constexpr size_t volume_dim = Dim;

  using boundary_conditions_base = BoundaryCondition<Dim>;

  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2<Dim>>>;
  using flux_variables = tmpl::conditional_t<
      system_type == SystemType::Conservative, tmpl::list<Var1, Var2<Dim>>,
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<>, tmpl::list<Var2<Dim>>>>;
  using gradient_variables = tmpl::conditional_t<
      system_type == SystemType::Conservative, tmpl::list<>,
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<Var1, Var2<Dim>>, tmpl::list<Var1>>>;
  using primitive_variables_tag =
      Tags::Variables<tmpl::list<PrimVar1, PrimVar2<Dim>>>;

  using compute_volume_time_derivative_terms =
      TimeDerivativeTerms<Dim, system_type,
                          has_primitive_and_conservative_vars>;

  using normal_dot_fluxes = NonconservativeNormalDotFlux<Dim>;
};

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;

  using internal_directions =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;
  using boundary_directions_interior =
      domain::Tags::BoundaryDirectionsInterior<Metavariables::volume_dim>;

  using common_simple_tags = tmpl::list<
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
      ::Tags::Next<::Tags::TimeStep>, ::Tags::Time,
      ::evolution::dg::Tags::Quadrature,
      typename Metavariables::system::variables_tag,
      db::add_tag_prefix<::Tags::dt,
                         typename Metavariables::system::variables_tag>,
      ::Tags::HistoryEvolvedVariables<
          typename Metavariables::system::variables_tag>,
      Var3, domain::Tags::Mesh<Metavariables::volume_dim>,
      ::domain::Tags::FunctionsOfTime,
      domain::CoordinateMaps::Tags::CoordinateMap<Metavariables::volume_dim,
                                                  Frame::Grid, Frame::Inertial>,
      domain::Tags::Interface<
          internal_directions,
          db::add_tag_prefix<
              ::Tags::NormalDotFlux,
              typename metavariables::boundary_scheme::variables_tag>>,
      domain::Tags::Element<Metavariables::volume_dim>,
      domain::Tags::Coordinates<Metavariables::volume_dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Metavariables::volume_dim, Frame::Logical,
                                    Frame::Inertial>,
      domain::Tags::MeshVelocity<Metavariables::volume_dim>,
      domain::Tags::DivMeshVelocity,
      domain::Tags::ElementMap<Metavariables::volume_dim, Frame::Grid>>;
  using simple_tags = tmpl::conditional_t<
      Metavariables::system_type == SystemType::Nonconservative or
          Metavariables::use_boundary_correction == UseBoundaryCorrection::Yes,
      common_simple_tags,
      tmpl::conditional_t<
          Metavariables::system_type == SystemType::Conservative,
          tmpl::push_back<
              common_simple_tags,
              db::add_tag_prefix<
                  ::Tags::Flux, typename Metavariables::system::variables_tag,
                  tmpl::size_t<Metavariables::volume_dim>, Frame::Inertial>>,
          tmpl::push_back<
              common_simple_tags,
              db::add_tag_prefix<
                  ::Tags::Flux,
                  Tags::Variables<tmpl::list<Var2<Metavariables::volume_dim>>>,
                  tmpl::size_t<Metavariables::volume_dim>, Frame::Inertial>>>>;
  using common_compute_tags = tmpl::list<
      domain::Tags::JacobianCompute<Metavariables::volume_dim, Frame::Logical,
                                    Frame::Inertial>,
      domain::Tags::DetInvJacobianCompute<Metavariables::volume_dim,
                                          Frame::Logical, Frame::Inertial>,
      domain::Tags::InternalDirectionsCompute<Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          SimpleUnnormalizedFaceNormal<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          FaceMeshVelocity<Metavariables::volume_dim,
                           Metavariables::use_moving_mesh>>,

      domain::Tags::BoundaryDirectionsInteriorCompute<
          Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::Slice<internal_directions, Var1>,
      domain::Tags::Slice<internal_directions,
                          Var2<Metavariables::volume_dim>>>;
  using compute_tags = tmpl::conditional_t<
      Metavariables::system::has_primitive_and_conservative_vars,
      tmpl::push_front<common_compute_tags,
                       PrimVarsCompute<Metavariables::volume_dim>>,
      common_compute_tags>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::flatten<tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              ::Actions::SetupDataBox,
              tmpl::conditional_t<Metavariables::use_boundary_correction ==
                                      UseBoundaryCorrection::No,
                                  ::dg::Actions::InitializeMortars<
                                      typename Metavariables::boundary_scheme>,
                                  tmpl::list<>>,
              ::evolution::dg::Initialization::Mortars<
                  Metavariables::volume_dim, typename Metavariables::system>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              ::evolution::dg::Actions::ComputeTimeDerivative<Metavariables>>>>;
};

template <size_t Dim, SystemType SystemTypeIn,
          UseBoundaryCorrection UseBoundaryCorrectionIn, bool LocalTimeStepping,
          bool UseMovingMesh, bool HasPrimitiveVariables>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr SystemType system_type = SystemTypeIn;
  static constexpr UseBoundaryCorrection use_boundary_correction =
      UseBoundaryCorrectionIn;
  static constexpr bool use_moving_mesh = UseMovingMesh;
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using system =
      System<Dim, system_type, use_boundary_correction, HasPrimitiveVariables>;
  using boundary_scheme = ::dg::FirstOrderScheme::FirstOrderScheme<
      Dim, typename system::variables_tag,
      db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
      Tags::NumericalFlux<BoundaryTerms<Dim, HasPrimitiveVariables>>,
      Tags::TimeStepId>;
  using normal_dot_numerical_flux =
      Tags::NumericalFlux<BoundaryTerms<Dim, HasPrimitiveVariables>>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>, normal_dot_numerical_flux,
                 domain::Tags::Domain<Dim>>;
  using step_choosers = tmpl::list<StepChoosers::Registrars::Constant>;

  using component_list = tmpl::list<component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename BoundaryCorrection, typename... PackagedFieldTags,
          typename... ProjectedFieldTags, size_t Dim, typename TagRetriever,
          typename... VolumeTags>
double dg_package_data(
    const gsl::not_null<Variables<tmpl::list<PackagedFieldTags...>>*>
        packaged_data,
    const BoundaryCorrection& boundary_correction,
    const Variables<tmpl::list<ProjectedFieldTags...>>& projected_fields,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const TagRetriever& get_tag, tmpl::list<VolumeTags...> /*meta*/) noexcept {
  return boundary_correction.dg_package_data(
      make_not_null(&get<PackagedFieldTags>(*packaged_data))...,
      get<ProjectedFieldTags>(projected_fields)..., normal_covector,
      mesh_velocity, normal_dot_mesh_velocity, get_tag(VolumeTags{})...);
}

template <bool LocalTimeStepping, bool UseMovingMesh, size_t Dim,
          SystemType system_type, UseBoundaryCorrection use_boundary_correction,
          bool HasPrims>
void test_impl(const Spectral::Quadrature quadrature,
               const ::dg::Formulation dg_formulation) noexcept {
  CAPTURE(LocalTimeStepping);
  CAPTURE(UseMovingMesh);
  CAPTURE(Dim);
  CAPTURE(system_type);
  CAPTURE(use_boundary_correction);
  CAPTURE(quadrature);
  CAPTURE(dg_formulation);
  using metavars = Metavariables<Dim, system_type, use_boundary_correction,
                                 LocalTimeStepping, UseMovingMesh, HasPrims>;
  Parallel::register_derived_classes_with_charm<
      StepChooser<typename metavars::step_choosers>>();
  Parallel::register_derived_classes_with_charm<StepController>();
  Parallel::register_derived_classes_with_charm<TimeStepper>();

  using system = typename metavars::system;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using variables_tag = typename system::variables_tag;
  using flux_variables = typename system::flux_variables;
  using flux_variables_tag = ::Tags::Variables<flux_variables>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, flux_variables_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  // The reference element in 2d denoted by X below:
  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  //
  // The "self_id" for the element that we are considering is marked by an X in
  // the diagram. We consider a configuration with one neighbor in the +xi
  // direction (east_id), and (in 2d and 3d) one in the -eta (south_id)
  // direction.
  //
  // In 1d there aren't any projections to test, and in 3d we only have 1
  // element in the z-direction.
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
  } else if constexpr (Dim == 2) {
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{
        {south_id},
        OrientationMap<Dim>{std::array{Direction<Dim>::lower_xi(),
                                       Direction<Dim>::lower_eta()}}};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{
        {south_id},
        OrientationMap<Dim>{std::array{Direction<Dim>::lower_xi(),
                                       Direction<Dim>::lower_eta(),
                                       Direction<Dim>::upper_zeta()}}};
  }

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});

  const Element<Dim> element{self_id, neighbors};
  MockRuntimeSystem runner = [&dg_formulation, &element,
                              &grid_to_inertial_map]() noexcept {
    std::vector<DirectionMap<
        Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        boundary_conditions{2};
    std::vector<Block<Dim>> blocks{2};
    DirectionMap<Dim, BlockNeighbor<Dim>> neighbors_block0{};
    DirectionMap<Dim, BlockNeighbor<Dim>> neighbors_block1{};
    if constexpr (Dim > 1) {
      neighbors_block0[Direction<Dim>::lower_eta()] = BlockNeighbor<Dim>{
          1, element.neighbors().at(Direction<Dim>::lower_eta()).orientation()};
      neighbors_block1[element.neighbors()
                           .at(Direction<Dim>::lower_eta())
                           .orientation()(Direction<Dim>::lower_eta())] =
          BlockNeighbor<Dim>{0, element.neighbors()
                                    .at(Direction<Dim>::lower_eta())
                                    .orientation()
                                    .inverse_map()};
      for (const auto& direction : Direction<Dim>::all_directions()) {
        if (direction != Direction<Dim>::lower_eta()) {
          boundary_conditions[0][direction] = std::make_unique<Outflow<Dim>>();
        }
        if (direction != element.neighbors()
                             .at(Direction<Dim>::lower_eta())
                             .orientation()(Direction<Dim>::lower_eta())) {
          boundary_conditions[1][direction] = std::make_unique<Outflow<Dim>>();
        }
      }
    } else {
      (void)element;
      for (const auto& direction : Direction<Dim>::all_directions()) {
        boundary_conditions[0][direction] = std::make_unique<Outflow<Dim>>();
        boundary_conditions[1][direction] = std::make_unique<Outflow<Dim>>();
      }
    }
    blocks[0] = Block<Dim>{
        domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{}),
        0, neighbors_block0, std::move(boundary_conditions[0])};
    blocks[1] = Block<Dim>{
        domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{}),
        1, neighbors_block1, std::move(boundary_conditions[1])};
    Domain<Dim> domain{std::move(blocks)};
    domain.inject_time_dependent_map_for_block(
        0, grid_to_inertial_map->get_clone());
    domain.inject_time_dependent_map_for_block(
        1, grid_to_inertial_map->get_clone());
    if constexpr (use_boundary_correction == UseBoundaryCorrection::No) {
      if constexpr (metavars::local_time_stepping) {
        std::vector<
            std::unique_ptr<StepChooser<typename metavars::step_choosers>>>
            step_choosers;
        step_choosers.emplace_back(
            std::make_unique<StepChoosers::Constant<>>(0.128));

        return MockRuntimeSystem{
            {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                                  make_array<Dim>(3_st)},
             typename metavars::normal_dot_numerical_flux::type{},
             std::move(domain), dg_formulation, std::move(step_choosers),
             static_cast<std::unique_ptr<StepController>>(
                 std::make_unique<StepControllers::SplitRemaining>()),
             static_cast<std::unique_ptr<LtsTimeStepper>>(
                 std::make_unique<TimeSteppers::AdamsBashforthN>(5))}};
      } else {
        return MockRuntimeSystem{
            {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                                  make_array<Dim>(3_st)},
             typename metavars::normal_dot_numerical_flux::type{},
             std::move(domain), dg_formulation}};
      }
    } else {
      if constexpr (metavars::local_time_stepping) {
        std::vector<
            std::unique_ptr<StepChooser<typename metavars::step_choosers>>>
            step_choosers;
        step_choosers.emplace_back(
            std::make_unique<StepChoosers::Constant<>>(0.128));

        return MockRuntimeSystem{
            {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                                  make_array<Dim>(3_st)},
             typename metavars::normal_dot_numerical_flux::type{},
             std::move(domain), dg_formulation,
             std::make_unique<BoundaryTerms<Dim, HasPrims>>(),
             std::move(step_choosers),
             static_cast<std::unique_ptr<StepController>>(
                 std::make_unique<StepControllers::SplitRemaining>()),
             static_cast<std::unique_ptr<LtsTimeStepper>>(
                 std::make_unique<TimeSteppers::AdamsBashforthN>(5))}};
      } else {
        return MockRuntimeSystem{
            {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                                  make_array<Dim>(3_st)},
             typename metavars::normal_dot_numerical_flux::type{},
             std::move(domain), dg_formulation,
             std::make_unique<BoundaryTerms<Dim, HasPrims>>()}};
      }
    }
  }();
  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<component<metavars>, tag>(runner,
                                                                    self_id);
  };

  const Mesh<Dim> mesh{2, Spectral::Basis::Legendre, quadrature};

  // Set the Jacobian to not be the identity because otherwise bugs creep in
  // easily.
  ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 2.0;
  }
  const auto det_inv_jacobian = determinant(inv_jac);
  const auto jacobian = determinant_and_inverse(inv_jac).second;

  // We don't need the Jacobian and map to be consistent since we are just
  // checking that given a Jacobian, coordinates, etc., the correct terms are
  // added to the evolution equations.
  const auto logical_coords = logical_coordinates(mesh);
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{};
  for (size_t i = 0; i < logical_coords.size(); ++i) {
    inertial_coords[i] = logical_coords[i];
  }

  std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity{};
  std::optional<Scalar<DataVector>> div_mesh_velocity{};
  if (UseMovingMesh) {
    const std::array<double, 3> velocities = {{1.2, -1.4, 0.3}};
    mesh_velocity =
        tnsr::I<DataVector, Dim, Frame::Inertial>{mesh.number_of_grid_points()};
    for (size_t i = 0; i < Dim; ++i) {
      mesh_velocity->get(i) = gsl::at(velocities, i);
    }
    div_mesh_velocity = Scalar<DataVector>{mesh.number_of_grid_points(), 1.5};
  }

  Variables<tmpl::list<Var1, Var2<Dim>>> evolved_vars{
      mesh.number_of_grid_points()};
  Scalar<DataVector> var3{mesh.number_of_grid_points()};
  // Set the variables so they are constant in y & z
  for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
    get(get<Var1>(evolved_vars))[i] = 3.0;
    get(get<Var1>(evolved_vars))[i + 1] = 2.0;
    for (size_t j = 0; j < Dim; ++j) {
      get<Var2<Dim>>(evolved_vars).get(j)[i] = j + 1.0;
      get<Var2<Dim>>(evolved_vars).get(j)[i + 1] = j + 3.0;
    }
    get(var3)[i] = 5.0;
    get(var3)[i + 1] = 6.0;
  }
  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>
      dt_evolved_vars{mesh.number_of_grid_points()};
  const ::TimeSteppers::History<
      Variables<tmpl::list<Var1, Var2<Dim>>>,
      Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>>
      history{1};

  // Compute expected volume fluxes
  [[maybe_unused]] const auto expected_fluxes = [&evolved_vars, &inv_jac, &mesh,
                                                 &mesh_velocity, &var3]() {
    [[maybe_unused]] const auto& var1 = get<Var1>(evolved_vars);
    [[maybe_unused]] const auto& var2 = get<Var2<Dim>>(evolved_vars);
    Variables<db::wrap_tags_in<::Tags::Flux, flux_variables, tmpl::size_t<Dim>,
                               Frame::Inertial>>
        fluxes{mesh.number_of_grid_points()};
    [[maybe_unused]] Scalar<DataVector> dt_var1{mesh.number_of_grid_points()};
    [[maybe_unused]] tnsr::I<DataVector, Dim, Frame::Inertial> dt_var2{
        mesh.number_of_grid_points()};
    Scalar<DataVector> square_var3{mesh.number_of_grid_points()};
    if constexpr (system_type == SystemType::Conservative) {
      (void)inv_jac;
      const auto flux_var1 = make_not_null(
          &get<::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(fluxes));
      const auto flux_var2 = make_not_null(
          &get<::Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
              fluxes));
      if constexpr (HasPrims) {
        typename PrimVarsCompute<Dim>::type prims{mesh.number_of_grid_points()};
        PrimVarsCompute<Dim>::function(make_not_null(&prims), mesh);
        TimeDerivativeTerms<Dim, system_type, HasPrims>::apply(
            make_not_null(&dt_var1), make_not_null(&dt_var2), flux_var1,
            flux_var2, &square_var3, var1, var2, var3, get<PrimVar1>(prims));
      } else {
        TimeDerivativeTerms<Dim, system_type, HasPrims>::apply(
            make_not_null(&dt_var1), make_not_null(&dt_var2), flux_var1,
            flux_var2, &square_var3, var1, var2, var3);
      }
      if (mesh_velocity.has_value()) {
        for (size_t i = 0; i < Dim; ++i) {
          flux_var1->get(i) -= get(var1) * mesh_velocity->get(i);
          for (size_t j = 0; j < Dim; ++j) {
            flux_var2->get(i, j) -= var2.get(j) * mesh_velocity->get(i);
          }
        }
      }
    } else if constexpr (system_type == SystemType::Mixed) {
      const auto flux_var2 = make_not_null(
          &get<::Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
              fluxes));
      const auto partial_derivs =
          partial_derivatives<tmpl::list<Var1>>(evolved_vars, mesh, inv_jac);
      const auto& d_var1 =
          get<::Tags::deriv<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
              partial_derivs);
      if constexpr (HasPrims) {
        typename PrimVarsCompute<Dim>::type prims{mesh.number_of_grid_points()};
        PrimVarsCompute<Dim>::function(make_not_null(&prims), mesh);
        TimeDerivativeTerms<Dim, system_type, HasPrims>::apply(
            make_not_null(&dt_var1), make_not_null(&dt_var2), flux_var2,
            &square_var3, d_var1, var1, var2, var3, get<PrimVar1>(prims));
      } else {
        TimeDerivativeTerms<Dim, system_type, HasPrims>::apply(
            make_not_null(&dt_var1), make_not_null(&dt_var2), flux_var2,
            &square_var3, d_var1, var1, var2, var3);
      }
      if (mesh_velocity.has_value()) {
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            flux_var2->get(i, j) -= var2.get(j) * mesh_velocity->get(i);
          }
        }
      }
    } else {
      (void)inv_jac;
      (void)mesh_velocity;
      (void)var3;
    }
    return fluxes;
  }();

  std::unordered_map<::Direction<Dim>,
                     Variables<tmpl::list<::Tags::NormalDotFlux<Var1>,
                                          ::Tags::NormalDotFlux<Var2<Dim>>>>>
      normal_dot_fluxes_interface{};
  const size_t interface_grid_points =
      mesh.slice_away(0).number_of_grid_points();
  for (const auto& [direction, nhbrs] : element.neighbors()) {
    (void)nhbrs;
    normal_dot_fluxes_interface[direction].initialize(interface_grid_points,
                                                      0.0);
  }

  const Slab time_slab{0.2, 3.4};
  const TimeDelta time_step{time_slab, {4, 100}};
  const TimeStepId time_step_id{true, 3, Time{time_slab, {3, 100}}};
  const TimeStepId next_time_step_id{time_step_id.time_runs_forward(),
                                     time_step_id.slab_number(),
                                     time_step_id.step_time() + time_step};
  // Our moving mesh map doesn't actually move (we set a mesh velocity, etc.
  // separately), but we need the FunctionsOfTime tag for boundary conditions.
  // When checking boundary conditions we just test that the boundary condition
  // function is called, not that the resulting code is correct, and so we just
  // use an Outflow condition, which doesn't actually need the coordinate maps.
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  if constexpr (not std::is_same_v<tmpl::list<>, flux_variables> and
                use_boundary_correction == UseBoundaryCorrection::No) {
    ActionTesting::emplace_component_and_initialize<component<metavars>>(
        &runner, self_id,
        {time_step_id,
         next_time_step_id,
         time_step,
         time_step,
         time_step_id.step_time().value(),
         quadrature,
         evolved_vars,
         dt_evolved_vars,
         history,
         var3,
         mesh,
         clone_unique_ptrs(functions_of_time),
         grid_to_inertial_map->get_clone(),
         normal_dot_fluxes_interface,
         element,
         inertial_coords,
         inv_jac,
         mesh_velocity,
         div_mesh_velocity,
         ElementMap<Dim, Frame::Grid>{
             self_id,
             domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                 domain::CoordinateMaps::Identity<Dim>{})},
         Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                                    tmpl::size_t<Dim>, Frame::Inertial>>{
             mesh.number_of_grid_points(), -100.}});
    for (const auto& [direction, neighbor_ids] : neighbors) {
      (void)direction;
      for (const auto& neighbor_id : neighbor_ids) {
        ActionTesting::emplace_component_and_initialize<component<metavars>>(
            &runner, neighbor_id,
            {time_step_id,
             next_time_step_id,
             time_step,
             time_step,
             time_step_id.step_time().value(),
             quadrature,
             evolved_vars,
             dt_evolved_vars,
             history,
             var3,
             mesh,
             clone_unique_ptrs(functions_of_time),
             grid_to_inertial_map->get_clone(),
             normal_dot_fluxes_interface,
             element,
             inertial_coords,
             inv_jac,
             mesh_velocity,
             div_mesh_velocity,
             ElementMap<Dim, Frame::Grid>{
                 neighbor_id,
                 domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                     domain::CoordinateMaps::Identity<Dim>{})},
             Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                                        tmpl::size_t<Dim>, Frame::Inertial>>{
                 mesh.number_of_grid_points(), -100.}});
      }
    }
  } else {
    ActionTesting::emplace_component_and_initialize<component<metavars>>(
        &runner, self_id,
        {time_step_id,
         next_time_step_id,
         time_step,
         time_step,
         time_step_id.step_time().value(),
         quadrature,
         evolved_vars,
         dt_evolved_vars,
         history,
         var3,
         mesh,
         clone_unique_ptrs(functions_of_time),
         grid_to_inertial_map->get_clone(),
         normal_dot_fluxes_interface,
         element,
         inertial_coords,
         inv_jac,
         mesh_velocity,
         div_mesh_velocity,
         ElementMap<Dim, Frame::Grid>{
             self_id,
             domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                 domain::CoordinateMaps::Identity<Dim>{})}});
    for (const auto& [direction, neighbor_ids] : neighbors) {
      (void)direction;
      for (const auto& neighbor_id : neighbor_ids) {
        ActionTesting::emplace_component_and_initialize<component<metavars>>(
            &runner, neighbor_id,
            {time_step_id,
             next_time_step_id,
             time_step,
             time_step,
             time_step_id.step_time().value(),
             quadrature,
             evolved_vars,
             dt_evolved_vars,
             history,
             var3,
             mesh,
             clone_unique_ptrs(functions_of_time),
             grid_to_inertial_map->get_clone(),
             normal_dot_fluxes_interface,
             element,
             inertial_coords,
             inv_jac,
             mesh_velocity,
             div_mesh_velocity,
             ElementMap<Dim, Frame::Grid>{
                 neighbor_id,
                 domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                     domain::CoordinateMaps::Identity<Dim>{})}});
      }
    }
  }
  // Setup the DataBox
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  // Initialize both the "old" and "new" mortars
  if (use_boundary_correction == UseBoundaryCorrection::No) {
    ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                    self_id);
  }
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);
  const auto variables_before_compute_time_derivatives =
      get_tag(variables_tag{});
  // Start testing the actual dg::ComputeTimeDerivative action
  Outflow<Dim>::number_of_times_called = 0;
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);
  if constexpr (use_boundary_correction == UseBoundaryCorrection::Yes) {
    CHECK(Outflow<Dim>::number_of_times_called ==
          element.external_boundaries().size());
  }

  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>
      expected_dt_evolved_vars{mesh.number_of_grid_points()};

  // Set dt<Var1>
  if constexpr (system_type == SystemType::Nonconservative or
                system_type == SystemType::Mixed) {
    for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
      if (quadrature == Spectral::Quadrature::GaussLobatto) {
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i] = 26.0;
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i + 1] = 39.0;
      } else if (quadrature == Spectral::Quadrature::Gauss) {
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i] =
            26.7320508075688785;
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i + 1] =
            41.196152422706632;
      } else {
        ERROR("Only support Gauss and Gauss-Lobatto quadrature in test, not "
              << quadrature);
      }
    }
    if (UseMovingMesh) {
      const tnsr::i<DataVector, Dim> d_var1 =
          get<::Tags::deriv<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
              partial_derivatives<tmpl::list<Var1>>(
                  variables_before_compute_time_derivatives, mesh, inv_jac));
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) +=
          get<0>(d_var1) * get<0>(*mesh_velocity);
    }
  } else {
    // Deal with source terms:
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) = square(get(var3));
    if constexpr (UseMovingMesh) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -=
          1.5 * get(get<Var1>(evolved_vars));
    }
    // Deal with volume flux divergence
    if (quadrature == Spectral::Quadrature::GaussLobatto and
        dg_formulation == ::dg::Formulation::StrongInertial) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -= 3.0;
      if constexpr (UseMovingMesh) {
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) +=
            mesh_velocity->get(0) * -1.0;
      }
    } else {
      if (dg_formulation == ::dg::Formulation::StrongInertial) {
        const auto div = divergence(expected_fluxes, mesh, inv_jac);
        const Scalar<DataVector>& div_var1_flux = get<::Tags::div<
            ::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>>(div);
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -=
            get(div_var1_flux);
      } else {
        Variables<db::wrap_tags_in<Tags::div, typename fluxes_tag::tags_list>>
            weak_div_fluxes{mesh.number_of_grid_points()};

        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
            det_jac_times_inverse_jacobian{};
        ::dg::metric_identity_det_jac_times_inv_jac(
            make_not_null(&det_jac_times_inverse_jacobian), mesh,
            inertial_coords, jacobian);

        weak_divergence(make_not_null(&weak_div_fluxes), expected_fluxes, mesh,
                        det_jac_times_inverse_jacobian);
        weak_div_fluxes *= get(det_inv_jacobian);

        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) +=
            get(get<Tags::div<
                    Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>>(
                weak_div_fluxes));
      }
    }
  }
  // Set dt<Var2<Dim>>
  if constexpr (system_type == SystemType::Conservative or
                system_type == SystemType::Mixed) {
    // Deal with source terms:
    for (size_t j = 0; j < Dim; ++j) {
      get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) =
          j * get(var3);
    }
    // Deal with volume flux divergence
    if (quadrature == Spectral::Quadrature::GaussLobatto and
        dg_formulation == ::dg::Formulation::StrongInertial) {
      get<0>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) += 4.0;
      if constexpr (Dim > 1) {
        get<1>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -= 18.0;
      }
      if constexpr (Dim > 2) {
        get<2>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -= 21.0;
      }
      if constexpr (UseMovingMesh) {
        for (size_t j = 0; j < Dim; ++j) {
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
              mesh_velocity->get(0) * 2.0;
        }
      }
    } else {
      if (dg_formulation == ::dg::Formulation::StrongInertial) {
        const auto div = divergence(expected_fluxes, mesh, inv_jac);
        const tnsr::I<DataVector, Dim>& div_var2_flux = get<::Tags::div<
            ::Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>(div);
        for (size_t i = 0; i < Dim; ++i) {
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(i) -=
              div_var2_flux.get(i);
        }
      } else {
        Variables<db::wrap_tags_in<Tags::div, typename fluxes_tag::tags_list>>
            weak_div_fluxes{mesh.number_of_grid_points()};

        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
            det_jac_times_inverse_jacobian{};
        ::dg::metric_identity_det_jac_times_inv_jac(
            make_not_null(&det_jac_times_inverse_jacobian), mesh,
            inertial_coords, determinant_and_inverse(inv_jac).second);

        weak_divergence(make_not_null(&weak_div_fluxes), expected_fluxes, mesh,
                        det_jac_times_inverse_jacobian);

        weak_div_fluxes *= get(det_inv_jacobian);

        for (size_t i = 0; i < Dim; ++i) {
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(i) +=
              get<Tags::div<
                  Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>(
                  weak_div_fluxes)
                  .get(i);
        }
      }
    }
    if constexpr (UseMovingMesh) {
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) -=
            1.5 * get<Var2<Dim>>(evolved_vars).get(j);
      }
    }
  } else {
    const tnsr::iJ<DataVector, Dim> d_var2 =
        quadrature == Spectral::Quadrature::GaussLobatto
            ? tnsr::iJ<DataVector, Dim>{}
            : get<::Tags::deriv<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
                  partial_derivatives<tmpl::list<Var1, Var2<Dim>>>(
                      variables_before_compute_time_derivatives, mesh,
                      inv_jac));
    if (quadrature == Spectral::Quadrature::GaussLobatto) {
      for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
        for (size_t j = 0; j < Dim; ++j) {
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j)[i] = -6.;
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j)[i + 1] =
              -12.;
        }
      }
    } else {
      for (size_t d = 0; d < Dim; ++d) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(d) =
            -get(get<Var1>(evolved_vars)) *
            get<0>(get<Var2<Dim>>(evolved_vars)) * d_var2.get(0, d);
        for (size_t j = 1; j < Dim; ++j) {
          get<0>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -=
              -get(get<Var1>(evolved_vars)) *
              get<Var2<Dim>>(evolved_vars).get(j) * d_var2.get(j, d);
        }
      }
    }

    // source term
    for (size_t j = 0; j < Dim; ++j) {
      get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
          j * get(var3);
    }
    if (UseMovingMesh) {
      for (size_t j = 0; j < Dim; ++j) {
        if (quadrature == Spectral::Quadrature::GaussLobatto) {
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
              2.0 * get<0>(*mesh_velocity);
        } else {
          get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
              d_var2.get(0, j) * get<0>(*mesh_velocity);
        }
      }
    }
  }
  if constexpr (HasPrims) {
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) += 5.0;
  }

  CHECK_VARIABLES_APPROX(
      SINGLE_ARG(ActionTesting::get_databox_tag<
                 component<metavars>,
                 db::add_tag_prefix<::Tags::dt,
                                    typename metavars::system::variables_tag>>(
          runner, self_id)),
      expected_dt_evolved_vars);

  const auto mortar_id_east =
      std::make_pair(Direction<Dim>::upper_xi(), east_id);

  if constexpr (use_boundary_correction == UseBoundaryCorrection::No) {
    const auto check_mortar =
        [&get_tag, &time_step_id](
            const std::pair<Direction<Dim>, ElementId<Dim>>& mortar_id,
            const size_t num_points) noexcept {
          CAPTURE(mortar_id);
          CAPTURE(Dim);
          const auto& all_mortar_data = get_tag(
              ::Tags::Mortars<
                  typename metavars::boundary_scheme::mortar_data_tag, Dim>{});
          const ::dg::SimpleBoundaryData<
              typename BoundaryTerms<Dim, HasPrims>::package_field_tags>&
              boundary_data =
                  all_mortar_data.at(mortar_id).local_data(time_step_id);
          CHECK(boundary_data.field_data.number_of_grid_points() == num_points);
          // Actually checking all the fields is a total nightmare because the
          // whole boundary scheme code is a complete disaster. The checks will
          // be added once the boundary scheme code is cleaned up.
          if constexpr (system_type == SystemType::Conservative) {
            const Scalar<DataVector> expected_var1_normal_dot_flux(num_points,
                                                                   0.);
            CHECK(get<::Tags::NormalDotFlux<Var1>>(boundary_data.field_data) ==
                  expected_var1_normal_dot_flux);
          }
        };
    check_mortar(mortar_id_east, Dim == 1 ? 1 : Dim == 2 ? 2 : 4);
    if constexpr (Dim > 1) {
      const auto mortar_id_south =
          std::make_pair(Direction<Dim>::lower_eta(), south_id);
      // The number of points on the mortar should be 3 in 2d and 9 and 3d
      // because we do projection to the southern neighbor, which has 3 points
      // per dimension.
      check_mortar(mortar_id_south, Dim == 2 ? 3 : 9);
    }
  } else {
    // At this point we know the volume terms have been computed correctly and
    // we want to verify that the functions we expect to be called on the
    // interfaces are called. Working out all the numbers explicitly would be
    // tedious.
    using variables_tags = typename variables_tag::tags_list;
    using temporary_tags_for_face = tmpl::list<Var3Squared>;
    using primitive_tags_for_face = tmpl::conditional_t<
        system::has_primitive_and_conservative_vars,
        typename BoundaryTerms<Dim, HasPrims>::dg_package_data_primitive_tags,
        tmpl::list<>>;
    using volume_tags = tmpl::conditional_t<
        system::has_primitive_and_conservative_vars,
        typename BoundaryTerms<Dim, HasPrims>::dg_package_data_volume_tags,
        tmpl::list<>>;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, flux_variables,
                                         tmpl::size_t<Dim>, Frame::Inertial>;
    using mortar_tags_list =
        typename BoundaryTerms<Dim, HasPrims>::dg_package_field_tags;
    std::unordered_map<Direction<Dim>,
                       tnsr::i<DataVector, Dim, Frame::Inertial>>
        face_normals{};
    for (const auto& direction : element.internal_boundaries()) {
      tnsr::i<DataVector, Dim, Frame::Inertial> volume_inv_jac_in_direction{
          inv_jac[0].size()};
      for (size_t inertial_index = 0; inertial_index < Dim; ++inertial_index) {
        volume_inv_jac_in_direction.get(inertial_index) =
            inv_jac.get(direction.dimension(), inertial_index);
      }

      const Mesh<Dim - 1> face_mesh = mesh.slice_away(direction.dimension());
      face_normals[direction] = tnsr::i<DataVector, Dim, Frame::Inertial>{
          face_mesh.number_of_grid_points()};
      ::evolution::dg::project_tensor_to_boundary(
          make_not_null(&face_normals[direction]), volume_inv_jac_in_direction,
          mesh, direction);
      for (auto& component : face_normals[direction]) {
        component *= direction.sign();
      }

      // GCC-7 gives unused direction warnings if we use structured
      // bindings.
      const auto normal_magnitude = magnitude(face_normals.at(direction));
      for (size_t i = 0; i < Dim; ++i) {
        face_normals[direction].get(i) /= get(normal_magnitude);
      }
    }

    const auto& mortar_meshes =
        get_tag(::evolution::dg::Tags::MortarMesh<Dim>{});
    const auto& mortar_sizes =
        get_tag(::evolution::dg::Tags::MortarSize<Dim>{});

    Variables<tmpl::list<Var3Squared>> volume_temporaries{
        mesh.number_of_grid_points()};
    get(get<Var3Squared>(volume_temporaries)) = square(get(var3));
    const auto compute_expected_mortar_data =
        [&element, &expected_fluxes, &face_normals, &get_tag, &mesh,
         &mesh_velocity, &mortar_meshes, &mortar_sizes, &volume_temporaries,
         &variables_before_compute_time_derivatives](
            const Direction<Dim>& local_direction,
            const ElementId<Dim>& local_neighbor_id,
            const bool local_data) noexcept {
          const auto& face_mesh = mesh.slice_away(local_direction.dimension());
          // First project data to the face in the direction of the mortar
          Variables<
              tmpl::append<variables_tags, fluxes_tags, temporary_tags_for_face,
                           primitive_tags_for_face>>
              fields_on_face{face_mesh.number_of_grid_points()};
          ::evolution::dg::project_contiguous_data_to_boundary(
              make_not_null(&fields_on_face),
              variables_before_compute_time_derivatives, mesh, local_direction);
          if constexpr (tmpl::size<fluxes_tags>::value != 0) {
            ::evolution::dg::project_contiguous_data_to_boundary(
                make_not_null(&fields_on_face), expected_fluxes, mesh,
                local_direction);
          } else {
            (void)expected_fluxes;
          }
          ::evolution::dg::project_tensors_to_boundary<temporary_tags_for_face>(
              make_not_null(&fields_on_face), volume_temporaries, mesh,
              local_direction);
          if constexpr (system::has_primitive_and_conservative_vars) {
            ::evolution::dg::project_tensors_to_boundary<
                primitive_tags_for_face>(
                make_not_null(&fields_on_face),
                get_tag(typename system::primitive_variables_tag{}), mesh,
                local_direction);
          }
          std::optional<tnsr::I<DataVector, Dim>> face_mesh_velocity{};
          if (UseMovingMesh) {
            face_mesh_velocity =
                tnsr::I<DataVector, Dim>{face_mesh.number_of_grid_points()};
            ::evolution::dg::project_tensor_to_boundary(
                make_not_null(&*face_mesh_velocity), *mesh_velocity, mesh,
                local_direction);
          }

          // Compute the normal dot mesh velocity and then the packaged data
          Variables<mortar_tags_list> packaged_data{
              face_mesh.number_of_grid_points()};
          std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};

          if (face_mesh_velocity.has_value()) {
            normal_dot_mesh_velocity = dot_product(
                *face_mesh_velocity, face_normals.at(local_direction));
          }
          const double max_char_speed_on_face = dg_package_data(
              make_not_null(&packaged_data), BoundaryTerms<Dim, HasPrims>{},
              fields_on_face, face_normals.at(local_direction),
              face_mesh_velocity, normal_dot_mesh_velocity, get_tag,
              volume_tags{});

          CHECK(max_char_speed_on_face ==
                max(get(
                    get<typename BoundaryTerms<Dim, HasPrims>::MaxAbsCharSpeed>(
                        packaged_data))));
          // Project the face data (stored in packaged_data) to the mortar
          const auto mortar_id =
              std::make_pair(local_direction, local_neighbor_id);
          const auto& mortar_mesh = mortar_meshes.at(mortar_id);
          const auto& mortar_size = mortar_sizes.at(mortar_id);
          auto boundary_data_on_mortar =
              ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
                  ? ::dg::project_to_mortar(packaged_data, face_mesh,
                                            mortar_mesh, mortar_size)
                  : std::move(packaged_data);

          std::vector<double> expected_data{
              boundary_data_on_mortar.data(),
              boundary_data_on_mortar.data() + boundary_data_on_mortar.size()};
          const auto& orientation =
              element.neighbors().at(local_direction).orientation();
          if (local_data or orientation.is_aligned()) {
            return expected_data;
          } else {
            return orient_variables_on_slice(
                expected_data, mortar_mesh.extents(),
                local_direction.dimension(), orientation);
          }
        };

    const auto check_mortar_data = [&compute_expected_mortar_data,
                                    &det_inv_jacobian, &get_tag, &mesh,
                                    &quadrature](const auto& mortar_data,
                                                 const auto& mortar_id) {
      CHECK_ITERABLE_APPROX(mortar_data.local_mortar_data()->second,
                            compute_expected_mortar_data(
                                mortar_id.first, mortar_id.second, true));

      // Check face normal and/or Jacobians
      const bool using_gauss_points = quadrature == Spectral::Quadrature::Gauss;

      Scalar<DataVector> local_face_normal_magnitude{};
      mortar_data.get_local_face_normal_magnitude(
          make_not_null(&local_face_normal_magnitude));
      CHECK(
          local_face_normal_magnitude ==
          get<::evolution::dg::Tags::MagnitudeOfNormal>(
              *get_tag(::evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>{})
                   .at(mortar_id.first)));

      if (using_gauss_points) {
        Scalar<DataVector> local_volume_det_inv_jacobian{};
        mortar_data.get_local_volume_det_inv_jacobian(
            make_not_null(&local_volume_det_inv_jacobian));
        CHECK(local_volume_det_inv_jacobian == det_inv_jacobian);

        // We use IrregularGridInterpolant to avoid reusing/copying the
        // apply_matrices-based interpolation in the source tree.
        const Mesh<Dim - 1> face_mesh =
            mesh.slice_away(mortar_id.first.dimension());
        const intrp::Irregular<Dim> interpolator{
            mesh, interface_logical_coordinates(face_mesh, mortar_id.first)};
        Variables<tmpl::list<::Tags::TempScalar<0>>> volume_det_jacobian{
            mesh.number_of_grid_points()};
        get(get<::Tags::TempScalar<0>>(volume_det_jacobian)) =
            1.0 / get(det_inv_jacobian);
        const Scalar<DataVector> expected_local_face_det_jacobian =
            get<::Tags::TempScalar<0>>(
                interpolator.interpolate(volume_det_jacobian));

        Scalar<DataVector> local_face_det_jacobian{};
        mortar_data.get_local_face_det_jacobian(
            make_not_null(&local_face_det_jacobian));
        CHECK_ITERABLE_APPROX(local_face_det_jacobian,
                              expected_local_face_det_jacobian);
      }
    };
    if (LocalTimeStepping) {
      const auto& east_mortar_data =
          get_tag(::evolution::dg::Tags::MortarDataHistory<
                      Dim, typename dt_variables_tag::type>{})
              .at(mortar_id_east)
              .local_data(time_step_id);
      check_mortar_data(east_mortar_data, mortar_id_east);
    } else {
      CHECK_ITERABLE_APPROX(
          get_tag(::evolution::dg::Tags::MortarData<Dim>{})
              .at(mortar_id_east)
              .local_mortar_data()
              ->second,
          compute_expected_mortar_data(mortar_id_east.first,
                                       mortar_id_east.second, true));
    }
    CHECK_ITERABLE_APPROX(
        *std::get<2>(
            ActionTesting::get_inbox_tag<
                component<metavars>,
                ::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                    Dim>>(runner, mortar_id_east.second)
                .at(time_step_id)
                .at(std::pair{
                    element.neighbors()
                        .at(mortar_id_east.first)
                        .orientation()(mortar_id_east.first.opposite()),
                    element.id()})),
        compute_expected_mortar_data(mortar_id_east.first,
                                     mortar_id_east.second, false));

    CHECK(std::get<3>(
              ActionTesting::get_inbox_tag<
                  component<metavars>,
                  ::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                      Dim>>(runner, mortar_id_east.second)
                  .at(time_step_id)
                  .at(std::pair{
                      element.neighbors()
                          .at(mortar_id_east.first)
                          .orientation()(mortar_id_east.first.opposite()),
                      element.id()})) ==
          (LocalTimeStepping ? next_time_step_id : time_step_id));

    if constexpr (Dim > 1) {
      const auto mortar_id_south =
          std::make_pair(Direction<Dim>::lower_eta(), south_id);
      CHECK(std::get<3>(
                ActionTesting::get_inbox_tag<
                    component<metavars>,
                    ::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                        Dim>>(runner, mortar_id_south.second)
                    .at(time_step_id)
                    .at(std::pair{
                        element.neighbors()
                            .at(mortar_id_south.first)
                            .orientation()(mortar_id_south.first.opposite()),
                        element.id()})) ==
            (LocalTimeStepping ? next_time_step_id : time_step_id));

      if (LocalTimeStepping) {
        const auto& south_mortar_data =
            get_tag(::evolution::dg::Tags::MortarDataHistory<
                        Dim, typename dt_variables_tag::type>{})
                .at(mortar_id_south)
                .local_data(time_step_id);
        check_mortar_data(south_mortar_data, mortar_id_south);
      } else {
        CHECK_ITERABLE_APPROX(get_tag(::evolution::dg::Tags::MortarData<Dim>{})
                                  .at(mortar_id_south)
                                  .local_mortar_data()
                                  ->second,
                              compute_expected_mortar_data(
                                  Direction<Dim>::lower_eta(), south_id, true));
      }
      CHECK_ITERABLE_APPROX(
          *std::get<2>(
              ActionTesting::get_inbox_tag<
                  component<metavars>,
                  ::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                      Dim>>(runner, mortar_id_south.second)
                  .at(time_step_id)
                  .at(std::pair{
                      element.neighbors()
                          .at(mortar_id_south.first)
                          .orientation()(mortar_id_south.first.opposite()),
                      element.id()})),
          compute_expected_mortar_data(mortar_id_south.first,
                                       mortar_id_south.second, false));
    }
  }
}

template <SystemType system_type, UseBoundaryCorrection use_boundary_correction,
          size_t Dim>
void test() noexcept {
  // The test impl is structured in the following way:
  // - the static mesh volume contributions are computed "by-hand" and used more
  //   or less as a regression test. This is relatively easy for Gauss-Lobatto
  //   points, a bit more tedious for Gauss points. We also assume the solution
  //   is constant in the y & z direction to make the math easier. The math
  //   implemented and checked against are the static mesh contributions coded
  //   up in TimeDerivativeTerms, with the addition of the strong or weak flux
  //   divergence that are computed by generic code outside the
  //   TimeDerivativeTerms struct.
  //
  //   The conservative equations are:
  //     dt var1 = -d_i (var1**2 + var2^i) + var3**2
  //     dt var2^j = -d_i (var1 * var2^i * var2^j + delta^i_j var1**3) +var3 * j
  //   if there are primitive variables then:
  //     dt var1 += prim_var1
  //
  //   The non-conservative equations are:
  //     dt var1 = -var2^i d_i var1 + var3**2
  //     dt var2^j = -var1 * var2^i d_i var2^j + var3 * j
  //
  //   The mixed conservative non-conservative equations are:
  //     dt var1 = -var2^i d_i var1 + var3**2
  //     dt var2^j = -d_i (var1 * var2^i * var2^j + delta^i_j var1**3) +var3 * j
  //   if there are primitive variables then:
  //     dt var1 += prim_var1
  //
  //   The mesh only has 2 points per dimension to make the math not too
  //   terrible. The differentiation matrices & weights for Gauss points are:
  //    D^{strong}: {{-0.866025403784438, 0.866025403784438},
  //                 {-0.866025403784438, 0.866025403784438}}
  //    w: {1.0, 1.0}
  //    D^{weak}: {{-0.866025403784438, -0.866025403784438},
  //               {0.866025403784438, 0.866025403784438}}
  //
  //   Gauss-Lobatto:
  //    D^{strong}: {{-1/2, 1/2}, {-1/2, 1/2}}
  //    w: {1.0, 1.0}
  //    D^{weak}: {{-1/2, -1/2}, {1/2, 1/2}}
  //
  //   The weak divergence matrix is given by:
  //     D_{i,l}^{weak} = (w_l/w_i) D_{l,i}^{strong}
  //
  // - the additional parts, such as moving mesh and boundary corrections become
  //   increasingly tedious (and unmaintainable) to work out separately
  //   (implementing the entire moving-mesh DG algorithm in Python is _a lot_ of
  //   work and extra code), so instead we verify that the expected mathematical
  //   manipulations occur. Specifically, we check adding the mesh velocity to
  //   the fluxes, and lifting the boundary contributions to the volume, even
  //   though the mesh velocity and boundary contributions are not the correct
  //   DG values

  Parallel::register_derived_classes_with_charm<
      BoundaryCorrection<Dim, true>>();
  Parallel::register_derived_classes_with_charm<
      BoundaryCorrection<Dim, false>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<Dim>>();

  const auto invoke_tests_with_quadrature_and_formulation =
      [](const Spectral::Quadrature quadrature,
         const ::dg::Formulation local_dg_formulation) noexcept {
        const auto moving_mesh_helper = [&local_dg_formulation,
                                         &quadrature](auto moving_mesh) {
          const auto prim_helper = [&local_dg_formulation, &moving_mesh,
                                    &quadrature](auto use_prims) {
            // Clang doesn't want moving mesh to be captured, but GCC requires
            // it. Silence the Clang warning by "using" it.
            (void)moving_mesh;
            if constexpr (not(decltype(use_prims)::value and system_type ==
                              SystemType::Nonconservative)) {
              test_impl<false, std::decay_t<decltype(moving_mesh)>::value, Dim,
                        system_type, use_boundary_correction,
                        std::decay_t<decltype(use_prims)>::value>(
                  quadrature, local_dg_formulation);
              test_impl<true, std::decay_t<decltype(moving_mesh)>::value, Dim,
                        system_type, use_boundary_correction,
                        std::decay_t<decltype(use_prims)>::value>(
                  quadrature, local_dg_formulation);
            }
          };
          prim_helper(std::integral_constant<bool, false>{});
          prim_helper(std::integral_constant<bool, true>{});
        };
        moving_mesh_helper(std::integral_constant<bool, false>{});
        moving_mesh_helper(std::integral_constant<bool, true>{});
      };
  for (const auto dg_formulation :
       {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
    invoke_tests_with_quadrature_and_formulation(
        Spectral::Quadrature::GaussLobatto, dg_formulation);
    if constexpr (use_boundary_correction == UseBoundaryCorrection::Yes) {
      invoke_tests_with_quadrature_and_formulation(Spectral::Quadrature::Gauss,
                                                   dg_formulation);
    }
  }
}
}  // namespace TestHelpers::evolution::dg::Actions
