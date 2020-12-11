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
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeImpl.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "Time/Tags.hpp"
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
struct SimpleNormalizedFaceNormal
    : db::ComputeTag,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>> {
  using base = ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>;
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
      ::Tags::TimeStepId, ::evolution::dg::Tags::Quadrature,
      typename Metavariables::system::variables_tag,
      db::add_tag_prefix<::Tags::dt,
                         typename Metavariables::system::variables_tag>,
      Var3, domain::Tags::Mesh<Metavariables::volume_dim>,
      domain::Tags::Interface<
          internal_directions,
          db::add_tag_prefix<
              ::Tags::NormalDotFlux,
              typename metavariables::boundary_scheme::variables_tag>>,
      domain::Tags::Element<Metavariables::volume_dim>,
      domain::Tags::InverseJacobian<Metavariables::volume_dim, Frame::Logical,
                                    Frame::Inertial>,
      domain::Tags::MeshVelocity<Metavariables::volume_dim>,
      domain::Tags::DivMeshVelocity>;
  using simple_tags = tmpl::conditional_t<
      Metavariables::system_type == SystemType::Conservative,
      tmpl::push_back<
          common_simple_tags,
          db::add_tag_prefix<
              ::Tags::Flux, typename Metavariables::system::variables_tag,
              tmpl::size_t<Metavariables::volume_dim>, Frame::Inertial>>,
      tmpl::conditional_t<
          Metavariables::system_type == SystemType::Nonconservative,
          common_simple_tags,
          tmpl::push_back<
              common_simple_tags,
              db::add_tag_prefix<
                  ::Tags::Flux,
                  Tags::Variables<tmpl::list<Var2<Metavariables::volume_dim>>>,
                  tmpl::size_t<Metavariables::volume_dim>, Frame::Inertial>>>>;
  using common_compute_tags = tmpl::list<
      domain::Tags::InternalDirectionsCompute<Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          SimpleNormalizedFaceNormal<Metavariables::volume_dim>>,
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
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              ::Actions::SetupDataBox,
              ::dg::Actions::InitializeMortars<
                  typename Metavariables::boundary_scheme>,
              ::evolution::dg::Initialization::Mortars<
                  Metavariables::volume_dim>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              ::evolution::dg::Actions::ComputeTimeDerivative<Metavariables>>>>;
};

template <size_t Dim, SystemType SystemTypeIn,
          UseBoundaryCorrection UseBoundaryCorrectionIn, bool UseMovingMesh,
          bool HasPrimitiveVariables>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr SystemType system_type = SystemTypeIn;
  static constexpr UseBoundaryCorrection use_boundary_correction =
      UseBoundaryCorrectionIn;
  static constexpr bool use_moving_mesh = UseMovingMesh;
  static constexpr bool local_time_stepping = false;
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
      tmpl::list<domain::Tags::InitialExtents<Dim>, normal_dot_numerical_flux>;

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

template <bool UseMovingMesh, size_t Dim, SystemType system_type,
          UseBoundaryCorrection use_boundary_correction, bool HasPrims>
void test_impl(const Spectral::Quadrature quadrature,
               const ::dg::Formulation dg_formulation) noexcept {
  CAPTURE(UseMovingMesh);
  CAPTURE(Dim);
  CAPTURE(system_type);
  CAPTURE(use_boundary_correction);
  CAPTURE(quadrature);
  using metavars = Metavariables<Dim, system_type, use_boundary_correction,
                                 UseMovingMesh, HasPrims>;
  using system = typename metavars::system;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
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
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  }
  const Element<Dim> element{self_id, neighbors};
  MockRuntimeSystem runner = [&dg_formulation]() noexcept {
    if constexpr (use_boundary_correction == UseBoundaryCorrection::No) {
      return MockRuntimeSystem{
          {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                                make_array<Dim>(3_st)},
           typename metavars::normal_dot_numerical_flux::type{},
           dg_formulation}};
    } else {
      return MockRuntimeSystem{
          {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                                make_array<Dim>(3_st)},
           typename metavars::normal_dot_numerical_flux::type{}, dg_formulation,
           std::make_unique<BoundaryTerms<Dim, HasPrims>>()}};
    }
  }();

  const Mesh<Dim> mesh{2, Spectral::Basis::Legendre, quadrature};

  ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 1.0;
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

  using flux_tags = typename system::flux_variables;

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

  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  if constexpr (not std::is_same_v<tmpl::list<>, flux_tags>) {
    ActionTesting::emplace_component_and_initialize<component<metavars>>(
        &runner, self_id,
        {time_step_id, quadrature, evolved_vars, dt_evolved_vars, var3, mesh,
         normal_dot_fluxes_interface, element, inv_jac, mesh_velocity,
         div_mesh_velocity,
         Variables<db::wrap_tags_in<::Tags::Flux, flux_tags, tmpl::size_t<Dim>,
                                    Frame::Inertial>>{2, -100.}});
    for (const auto& [direction, neighbor_ids] : neighbors) {
      (void)direction;
      for (const auto& neighbor_id : neighbor_ids) {
        ActionTesting::emplace_component_and_initialize<component<metavars>>(
            &runner, neighbor_id,
            {time_step_id, quadrature, evolved_vars, dt_evolved_vars, var3,
             mesh, normal_dot_fluxes_interface, element, inv_jac, mesh_velocity,
             div_mesh_velocity,
             Variables<db::wrap_tags_in<::Tags::Flux, flux_tags,
                                        tmpl::size_t<Dim>, Frame::Inertial>>{
                 2, -100.}});
      }
    }
  } else {
    ActionTesting::emplace_component_and_initialize<component<metavars>>(
        &runner, self_id,
        {time_step_id, quadrature, evolved_vars, dt_evolved_vars, var3, mesh,
         normal_dot_fluxes_interface, element, inv_jac, mesh_velocity,
         div_mesh_velocity});
    for (const auto& [direction, neighbor_ids] : neighbors) {
      (void)direction;
      for (const auto& neighbor_id : neighbor_ids) {
        ActionTesting::emplace_component_and_initialize<component<metavars>>(
            &runner, neighbor_id,
            {time_step_id, quadrature, evolved_vars, dt_evolved_vars, var3,
             mesh, normal_dot_fluxes_interface, element, inv_jac, mesh_velocity,
             div_mesh_velocity});
      }
    }
  }
  // Setup the DataBox
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  // Initialize both the "old" and "new" mortars
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);
  // Start testing the actual dg::ComputeTimeDerivative action
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>
      expected_dt_evolved_vars{mesh.number_of_grid_points()};

  // Set dt<Var1>
  if constexpr (system_type == SystemType::Nonconservative or
                system_type == SystemType::Mixed) {
    for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
      if (quadrature == Spectral::Quadrature::GaussLobatto) {
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i] = 25.5;
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i + 1] = 37.5;
      } else if (quadrature == Spectral::Quadrature::Gauss) {
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i] = 25.5;
        get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i + 1] = 25.5;
      } else {
        ERROR("Only support Gauss and Gauss-Lobatto quadrature in test, not "
              << quadrature);
      }
    }
    if (UseMovingMesh) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) +=
          -0.5 * get<0>(*mesh_velocity);
    }
  } else {
    // Deal with source terms:
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) = square(get(var3));
    // Deal with volume flux divergence
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -= 1.5;
    if constexpr (UseMovingMesh) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -=
          1.5 * get(get<Var1>(evolved_vars)) - mesh_velocity->get(0) * -0.5;
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
    get<0>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) += 2.0;
    if constexpr (Dim > 1) {
      get<1>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -= 9.0;
    }
    if constexpr (Dim > 2) {
      get<2>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -= 10.5;
    }
    if constexpr (UseMovingMesh) {
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) -=
            1.5 * get<Var2<Dim>>(evolved_vars).get(j) -
            mesh_velocity->get(0) * 1.0;
      }
    }
  } else {
    for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j)[i] = -3.;
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j)[i + 1] =
            -6.;
      }
    }
    for (size_t j = 0; j < Dim; ++j) {
      get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
          j * get(var3);
    }
    if (UseMovingMesh) {
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
            get<0>(*mesh_velocity);
      }
    }
  }
  if constexpr (HasPrims) {
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) += 5.0;
  }

  CHECK(ActionTesting::get_databox_tag<
            component<metavars>,
            db::add_tag_prefix<::Tags::dt,
                               typename metavars::system::variables_tag>>(
            runner, self_id) == expected_dt_evolved_vars);

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<component<metavars>, tag>(runner,
                                                                    self_id);
  };

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
    using variables_tag = typename system::variables_tag;
    using variables_tags = typename variables_tag::tags_list;
    using flux_variables = typename system::flux_variables;
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
    using flux_variables_tag = ::Tags::Variables<flux_variables>;
    using fluxes_tag = db::add_tag_prefix<::Tags::Flux, flux_variables_tag,
                                          tmpl::size_t<Dim>, Frame::Inertial>;
    using mortar_tags_list =
        typename BoundaryTerms<Dim, HasPrims>::dg_package_field_tags;
    const auto& face_meshes =
        get_tag(domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                        domain::Tags::Mesh<Dim - 1>>{});
    const auto& face_normals = get_tag(
        domain::Tags::Interface<
            domain::Tags::InternalDirections<Dim>,
            ::Tags::Normalized<
                domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>{});
    const auto& face_mesh_velocities =
        get_tag(domain::Tags::Interface<
                domain::Tags::InternalDirections<Dim>,
                domain::Tags::MeshVelocity<Dim, Frame::Inertial>>{});
    const auto& mortar_meshes =
        get_tag(::evolution::dg::Tags::MortarMesh<Dim>{});
    const auto& mortar_sizes =
        get_tag(::evolution::dg::Tags::MortarSize<Dim>{});

    Variables<tmpl::list<Var3Squared>> volume_temporaries{
        mesh.number_of_grid_points()};
    get(get<Var3Squared>(volume_temporaries)) = square(get(var3));
    const auto compute_expected_mortar_data =
        [&face_mesh_velocities, &face_meshes, &face_normals, &get_tag, &mesh,
         &mortar_meshes, &mortar_sizes, &volume_temporaries](
            const Direction<Dim>& local_direction,
            const ElementId<Dim>& local_neighbor_id) noexcept {
          const auto& face_mesh = face_meshes.at(local_direction);
          // First project data to the face in the direction of the mortar
          Variables<
              tmpl::append<variables_tags, fluxes_tags, temporary_tags_for_face,
                           primitive_tags_for_face>>
              fields_on_face{face_mesh.number_of_grid_points()};
          ::evolution::dg::project_contiguous_data_to_boundary(
              make_not_null(&fields_on_face), get_tag(variables_tag{}), mesh,
              local_direction);
          if constexpr (tmpl::size<fluxes_tags>::value != 0) {
            ::evolution::dg::project_contiguous_data_to_boundary(
                make_not_null(&fields_on_face), get_tag(fluxes_tag{}), mesh,
                local_direction);
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

          // Compute the normal dot mesh velocity and then the packaged data
          Variables<mortar_tags_list> packaged_data{
              face_mesh.number_of_grid_points()};
          std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};

          if (face_mesh_velocities.at(local_direction).has_value()) {
            normal_dot_mesh_velocity =
                dot_product(*face_mesh_velocities.at(local_direction),
                            face_normals.at(local_direction));
          }
          const double max_char_speed_on_face = dg_package_data(
              make_not_null(&packaged_data), BoundaryTerms<Dim, HasPrims>{},
              fields_on_face, face_normals.at(local_direction),
              face_mesh_velocities.at(local_direction),
              normal_dot_mesh_velocity, get_tag, volume_tags{});

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

          return std::vector<double>{
              boundary_data_on_mortar.data(),
              boundary_data_on_mortar.data() + boundary_data_on_mortar.size()};
        };

    CHECK_ITERABLE_APPROX(get_tag(::evolution::dg::Tags::MortarData<Dim>{})
                              .at(mortar_id_east)
                              .local_mortar_data()
                              ->second,
                          compute_expected_mortar_data(mortar_id_east.first,
                                                       mortar_id_east.second));
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
                                     mortar_id_east.second));

    if constexpr (Dim > 1) {
      const auto mortar_id_south =
          std::make_pair(Direction<Dim>::lower_eta(), south_id);
      CHECK_ITERABLE_APPROX(
          get_tag(::evolution::dg::Tags::MortarData<Dim>{})
              .at(mortar_id_south)
              .local_mortar_data()
              ->second,
          compute_expected_mortar_data(Direction<Dim>::lower_eta(), south_id));
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
                                       mortar_id_south.second));
    }
  }
}

template <SystemType system_type, UseBoundaryCorrection use_boundary_correction,
          size_t Dim>
void test() noexcept {
  Parallel::register_derived_classes_with_charm<
      BoundaryCorrection<Dim, use_boundary_correction>>();

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
              test_impl<std::decay_t<decltype(moving_mesh)>::value, Dim,
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
  for (const auto dg_formulation : {::dg::Formulation::StrongInertial}) {
    invoke_tests_with_quadrature_and_formulation(
        Spectral::Quadrature::GaussLobatto, dg_formulation);
  }
}
}  // namespace TestHelpers::evolution::dg::Actions
