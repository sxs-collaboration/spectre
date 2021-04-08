// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/BoundaryConditionsImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// We use offsets and then fill the Variables with offset+DataVector index
// This is a way to generate unique but known numbers
constexpr double offset_dt_evolved_vars = 1.0;
constexpr double offset_evolved_vars = 20.0;
constexpr double offset_temporaries = 50.0;
constexpr double offset_volume_fluxes = 200.0;
constexpr double offset_partial_derivs = 3000.0;
constexpr double offset_primitive_vars = 7000.0;
constexpr double offset_boundary_condition = 10000.0;
constexpr double offset_boundary_correction = 20000.0;
const std::array expected_velocities{1.2, -1.4, 0.3};

namespace Tags {
struct BoundaryCorrectionVolumeTag : db::SimpleTag {
  using type = double;
};

struct BoundaryConditionVolumeTag : db::SimpleTag {
  using type = double;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

struct Var3Squared : db::SimpleTag {
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
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim, Frame::Inertial>;
};
}  // namespace Tags

using SystemType = TestHelpers::evolution::dg::Actions::SystemType;

template <size_t Dim, bool HasPrims, SystemType SysType,
          bool HasInverseSpatialMetric>
struct BoundaryTerms;

template <size_t Dim, bool HasPrims, SystemType SysType,
          bool HasInverseSpatialMetric>
class BoundaryCorrection : public PUP::able {
 public:
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;

  ~BoundaryCorrection() override = default;

  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT

  using creatable_classes = tmpl::list<
      BoundaryTerms<Dim, HasPrims, SysType, HasInverseSpatialMetric>>;
};

template <size_t Dim, bool HasPrims, SystemType SysType,
          bool HasInverseSpatialMetric>
struct BoundaryTerms final
    : public BoundaryCorrection<Dim, HasPrims, SysType,
                                HasInverseSpatialMetric> {
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  explicit BoundaryTerms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoundaryTerms);  // NOLINT
  BoundaryTerms(const bool mesh_is_moving, const double sign_of_normal)
      : mesh_is_moving_(mesh_is_moving), sign_of_normal_(sign_of_normal) {}
  BoundaryTerms() = default;
  BoundaryTerms(const BoundaryTerms&) = default;
  BoundaryTerms& operator=(const BoundaryTerms&) = default;
  BoundaryTerms(BoundaryTerms&&) = default;
  BoundaryTerms& operator=(BoundaryTerms&&) = default;
  ~BoundaryTerms() override = default;

  using variables_tags = tmpl::list<Tags::Var1, Tags::Var2<Dim>>;
  using variables_tag = ::Tags::Variables<variables_tags>;

  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection<Dim, HasPrims, SysType, HasInverseSpatialMetric>::pup(p);
    p | mesh_is_moving_;
    p | sign_of_normal_;
  }

  using dg_package_field_tags = tmpl::push_back<
      tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                   variables_tags>,
      MaxAbsCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_package_data_primitive_tags =
      tmpl::conditional_t<HasPrims, tmpl::list<Tags::PrimVar1>, tmpl::list<>>;
  using dg_package_data_volume_tags = tmpl::conditional_t<
      HasPrims, tmpl::list<Tags::BoundaryCorrectionVolumeTag>, tmpl::list<>>;

  // Conservative system, flat background
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
    if (mesh_velocity.has_value()) {
      REQUIRE(normal_dot_mesh_velocity.has_value());
      CHECK_ITERABLE_APPROX(*normal_dot_mesh_velocity,
                            dot_product(normal_covector, *mesh_velocity));
    }

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

  // Conservative system, curved background
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, var1, var2,
                           flux_var1, flux_var2, var3_squared, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity);
  }

  // Conservative system with prim vars, flat background
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

      const double volume_number) const noexcept {
    dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                    out_var1, out_var2, max_abs_char_speed, var1, var2,
                    flux_var1, flux_var2, var3_squared, normal_covector,
                    mesh_velocity, normal_dot_mesh_velocity);
    get(*out_var1) += get(prim_var1) + volume_number;
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          (get(prim_var1) + volume_number) *
          get(dot_product(*mesh_velocity, normal_covector));
    }
    return max(get(*max_abs_char_speed));
  }

  // Conservative system with prim vars, curved background
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const noexcept {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, var1, var2,
                           flux_var1, flux_var2, var3_squared, prim_var1,
                           normal_covector, mesh_velocity,
                           normal_dot_mesh_velocity, volume_number);
  }

  // Nonconservative system, flat background
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
    if (mesh_velocity.has_value()) {
      REQUIRE(normal_dot_mesh_velocity.has_value());
      CHECK_ITERABLE_APPROX(*normal_dot_mesh_velocity,
                            dot_product(normal_covector, *mesh_velocity));
    }

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

  // Nonconservative system, curved background
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, var1, var2,
                           var3_squared, normal_covector, mesh_velocity,
                           normal_dot_mesh_velocity);
  }

  // Mixed system, no prims, flat background
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
    if (mesh_velocity.has_value()) {
      REQUIRE(normal_dot_mesh_velocity.has_value());
      CHECK_ITERABLE_APPROX(*normal_dot_mesh_velocity,
                            dot_product(normal_covector, *mesh_velocity));
    }

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

  // Mixed system, no prims, curved background
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, var1, var2,
                           flux_var2, var3_squared, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity);
  }

  // Mixed system with prims, flat background
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

      const double volume_number) const noexcept {
    dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                    out_var1, out_var2, max_abs_char_speed, var1, var2,
                    flux_var2, var3_squared, normal_covector, mesh_velocity,
                    normal_dot_mesh_velocity);
    get(*out_var1) += get(prim_var1) + volume_number;
    return max(get(*max_abs_char_speed));
  }

  // Mixed system with prims, curved background
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const noexcept {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, var1, var2,
                           flux_var2, var3_squared, prim_var1, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity,
                           volume_number);
  }

  void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> boundary_correction_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var2,
      const Scalar<DataVector>& int_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_normal_dot_flux_var2,
      const Scalar<DataVector>& int_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_var2,
      const Scalar<DataVector>& int_max_abs_char_speed,
      const Scalar<DataVector>& ext_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_normal_dot_flux_var2,
      const Scalar<DataVector>& ext_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_var2,
      const Scalar<DataVector>& ext_max_abs_char_speed,
      const dg::Formulation formulation) const noexcept {
    static_assert(Dim == 1,
                  "Flux dot normal assumes 1d, mostly because normal vector is "
                  "assumed to be 1d.");

    get(*boundary_correction_var1) =
        offset_boundary_correction *
        (formulation == dg::Formulation::WeakInertial ? 2.0 : 1.0);
    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_var2->get(i) = offset_boundary_correction + 1.0 + i;
    }
    const size_t num_pts = get(int_var1).size();

    const double mesh_velocity = mesh_is_moving_ ? 1.2 : 0.0;
    const double normalization_factor =
        HasInverseSpatialMetric ? sqrt(offset_temporaries + 1.0) : 1.0;
    if (SysType == SystemType::Conservative) {
      CHECK_ITERABLE_APPROX(
          get(int_normal_dot_flux_var1),
          DataVector(sign_of_normal_ / normalization_factor *
                     (offset_volume_fluxes - mesh_velocity * get(int_var1))));
    } else {
      CHECK_ITERABLE_APPROX(
          get(int_normal_dot_flux_var1),
          DataVector(offset_evolved_vars +
                     sign_of_normal_ / normalization_factor * get<0>(int_var2) -
                     sign_of_normal_ / normalization_factor * mesh_velocity));
    }

    if (SysType == SystemType::Conservative) {
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            int_normal_dot_flux_var2.get(i),
            DataVector(sign_of_normal_ / normalization_factor *
                       (offset_volume_fluxes + 1.0 + i -
                        mesh_velocity * int_var2.get(i))));
      }
    } else if (SysType == SystemType::Mixed) {
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            int_normal_dot_flux_var2.get(i),
            DataVector(
                sign_of_normal_ / normalization_factor *
                (offset_volume_fluxes + i - mesh_velocity * int_var2.get(i))));
      }
    } else {
      for (size_t i = 0; i < Dim; ++i) {
        DataVector expected{num_pts, 0.0};
        for (size_t j = 0; j < Dim; ++j) {
          expected += int_var2.get(j) /
                          square(normalization_factor)  // n_i n_j var2^j, bot
                                                        // n_i = (\pm 1) in 1d
                      - sign_of_normal_ / normalization_factor *
                            int_var2.get(i) * mesh_velocity;  // var2^i v^j n_j
        }
        CHECK_ITERABLE_APPROX(int_normal_dot_flux_var2.get(i), expected);
      }
    }
    CHECK_ITERABLE_APPROX(
        get(int_var1),
        DataVector(num_pts,
                   offset_evolved_vars +
                       (HasPrims ? offset_primitive_vars + 3.5 : 0.0)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(int_var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1.0 + i));
    }
    CHECK_ITERABLE_APPROX(
        get(int_max_abs_char_speed),
        DataVector(num_pts,
                   2.0 * offset_temporaries +
                       sign_of_normal_ / normalization_factor * mesh_velocity));

    if (SysType == SystemType::Conservative) {
      // The two comes from the dg_package_data also subtracting off the mesh
      // velocity.
      CHECK_ITERABLE_APPROX(
          get(ext_normal_dot_flux_var1),
          DataVector{-sign_of_normal_ / normalization_factor *
                     (offset_boundary_condition + 1.0 + Dim -
                      2 * mesh_velocity * offset_boundary_condition -
                      (HasPrims ? mesh_velocity * (offset_boundary_condition +
                                                   3.0 + 2 * Dim + 3.5)
                                : 0.0))});
    } else {
      CHECK_ITERABLE_APPROX(
          get(ext_normal_dot_flux_var1),
          DataVector(
              (offset_boundary_condition -
               sign_of_normal_ / normalization_factor * get<0>(ext_var2) +
               sign_of_normal_ / normalization_factor * mesh_velocity)));
    }
    if (SysType == SystemType::Conservative or SysType == SystemType::Mixed) {
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            ext_normal_dot_flux_var2.get(i),
            DataVector(-sign_of_normal_ / normalization_factor *
                       (offset_boundary_condition + 2.0 + Dim + i -
                        2.0 * mesh_velocity * ext_var2.get(i))));
      }
    } else {
      static_assert(Dim == 1);
      CHECK_ITERABLE_APPROX(
          get<0>(ext_normal_dot_flux_var2),
          DataVector((get<0>(ext_var2) / square(normalization_factor) +
                      sign_of_normal_ / normalization_factor *
                          get<0>(ext_var2) * mesh_velocity)));
    }
    CHECK_ITERABLE_APPROX(
        get(ext_var1),
        DataVector(num_pts, offset_boundary_condition +
                                (HasPrims ? offset_boundary_condition +
                                                2.0 * Dim + 3.0 + 3.5
                                          : 0.0)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          ext_var2.get(i),
          DataVector(num_pts, offset_boundary_condition + 1.0 + i));
    }
    CHECK_ITERABLE_APPROX(
        get(ext_max_abs_char_speed),
        DataVector(num_pts,
                   2.0 * (offset_boundary_condition + 2.0 + 2 * Dim) -
                       sign_of_normal_ / normalization_factor * mesh_velocity));
  }

 private:
  bool mesh_is_moving_{false};
  double sign_of_normal_{0.0};
};

template <size_t Dim, bool HasPrims, SystemType SysType,
          bool HasInverseSpatialMetric>
PUP::able::PUP_ID
    // NOLINTNEXTLINE
    BoundaryTerms<Dim, HasPrims, SysType, HasInverseSpatialMetric>::my_PUP_ID =
        0;

// Forward declare different boundary conditions.
//
// We template them on the system so we can test that we have access to all the
// different tags that we should have access to.
template <typename System>
class Outflow;
template <typename System>
class TimeDerivative;
template <typename System>
class Ghost;
template <typename System>
class GhostAndTimeDerivative;

template <typename System>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  // Note: Outflow is intentionally first so it gets applied first. This makes
  // the test easier because the other BCs modify the time derivatives.
  using creatable_classes =
      tmpl::list<Outflow<System>, Ghost<System>, TimeDerivative<System>,
                 GhostAndTimeDerivative<System>,
                 domain::BoundaryConditions::Periodic<BoundaryCondition>,
                 domain::BoundaryConditions::None<BoundaryCondition>>;

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

template <typename System>
class Outflow : public BoundaryCondition<System> {
 public:
  Outflow() = default;
  explicit Outflow(const bool mesh_is_moving)
      : mesh_is_moving_(mesh_is_moving) {}
  Outflow(Outflow&&) noexcept = default;
  Outflow& operator=(Outflow&&) noexcept = default;
  Outflow(const Outflow&) = default;
  Outflow& operator=(const Outflow&) = default;
  ~Outflow() override = default;

  explicit Outflow(CkMigrateMessage* msg) noexcept
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Outflow);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<Outflow<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::Outflow;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | mesh_is_moving_;
  }

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Var1, Tags::Var2<System::volume_dim>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::PrimVar1, Tags::PrimVar2<System::volume_dim>>;
  using dg_interior_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Var1>>;
  using dg_gridless_tags = tmpl::list<Tags::BoundaryConditionVolumeTag>;

  std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    CHECK(volume_number == 2.5);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(var3_squared),
                          DataVector(num_pts, offset_temporaries));
    CHECK_ITERABLE_APPROX(get(var1), DataVector(num_pts, offset_evolved_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1 + i));
      for (size_t j = 0; j < num_pts; ++j) {
        // Catch doesn't allow `CHECK(a or b) so we do `CHECK((a or b))` instead
        if constexpr (System::volume_dim == 1) {
          const double normalization_factor =
              System::has_inverse_spatial_metric
                  ? sqrt(offset_temporaries + 1.0)
                  : 1.0;
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) ==
                     1.0 / normalization_factor or
                 approx(outward_directed_normal_covector.get(i)[j]) ==
                     -1.0 / normalization_factor));
        } else {
          static_assert(not System::has_inverse_spatial_metric);
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) == 1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == -1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == 0.0));
        }
      }
    }
    CHECK_ITERABLE_APPROX(get(dt_var1),
                          DataVector(num_pts, offset_dt_evolved_vars));
    REQUIRE(face_mesh_velocity.has_value() == mesh_is_moving_);
    if (mesh_is_moving_) {
      for (size_t i = 0; i < System::volume_dim; ++i) {
        CHECK_ITERABLE_APPROX(
            face_mesh_velocity->get(i),
            DataVector(num_pts, gsl::at(expected_velocities, i)));
      }
    }
    return std::nullopt;
  }

  std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_outflow(face_mesh_velocity, outward_directed_normal_covector, var1, var2,
               var3_squared, dt_var1, volume_number);
    CHECK_ITERABLE_APPROX(get(dot_product(outward_directed_normal_covector,
                                          outward_directed_normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return std::nullopt;
  }

  std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_outflow(face_mesh_velocity, outward_directed_normal_covector, var1, var2,
               var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_outflow(face_mesh_velocity, outward_directed_normal_covector, var1, var2,
               prim_var1, prim_var2, var3_squared, dt_var1, volume_number);
    CHECK_ITERABLE_APPROX(get(dot_product(outward_directed_normal_covector,
                                          outward_directed_normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return std::nullopt;
  }

 private:
  bool mesh_is_moving_{false};
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow<System>::my_PUP_ID = 0;

template <typename System>
class Ghost : public BoundaryCondition<System> {
 public:
  Ghost() = default;
  explicit Ghost(const bool mesh_is_moving) : mesh_is_moving_(mesh_is_moving) {}
  Ghost(Ghost&&) noexcept = default;
  Ghost& operator=(Ghost&&) noexcept = default;
  Ghost(const Ghost&) = default;
  Ghost& operator=(const Ghost&) = default;
  ~Ghost() override = default;

  explicit Ghost(CkMigrateMessage* msg) noexcept
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Ghost);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<Ghost<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::Ghost;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | mesh_is_moving_;
  }

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Var1, Tags::Var2<System::volume_dim>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::PrimVar1, Tags::PrimVar2<System::volume_dim>>;
  using dg_interior_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Var1>>;
  using dg_gridless_tags = tmpl::list<Tags::BoundaryConditionVolumeTag>;

  // Nonconservative system, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    get(*out_var1) = offset_boundary_condition;
    for (size_t i = 0; i < System::volume_dim; ++i) {
      out_var2->get(i) =
          offset_boundary_condition + 1.0 + static_cast<double>(i);
    }
    get(*out_var3_squared) = offset_boundary_condition + 1.0 +
                             (2 + System::volume_dim) * System::volume_dim;

    CHECK(volume_number == 2.5);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(var3_squared),
                          DataVector(num_pts, offset_temporaries));
    CHECK_ITERABLE_APPROX(get(var1), DataVector(num_pts, offset_evolved_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1 + i));
      for (size_t j = 0; j < num_pts; ++j) {
        // Catch doesn't allow `CHECK(a or b) so we do `CHECK((a or b))` instead
        if constexpr (System::volume_dim == 1) {
          const double normalization_factor =
              System::has_inverse_spatial_metric
                  ? sqrt(offset_temporaries + 1.0)
                  : 1.0;
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) ==
                     1.0 / normalization_factor or
                 approx(outward_directed_normal_covector.get(i)[j]) ==
                     -1.0 / normalization_factor));
        } else {
          static_assert(not System::has_inverse_spatial_metric);
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) == 1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == -1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == 0.0));
        }
      }
    }
    CHECK_ITERABLE_APPROX(get(dt_var1),
                          DataVector(num_pts, offset_dt_evolved_vars));
    REQUIRE(face_mesh_velocity.has_value() == mesh_is_moving_);
    if (mesh_is_moving_) {
      for (size_t i = 0; i < System::volume_dim; ++i) {
        CHECK_ITERABLE_APPROX(
            face_mesh_velocity->get(i),
            DataVector(num_pts, gsl::at(expected_velocities, i)));
      }
    }
    return std::nullopt;
  }

  // Nonconservative system, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3_squared, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, var3_squared,
                    dt_var1, volume_number);
  }

  // Mixed conservative non-conservative system, no prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_ghost(out_var1, out_var2, out_var3_squared, face_mesh_velocity,
             outward_directed_normal_covector, var1, var2, var3_squared,
             dt_var1, volume_number);
    for (size_t i = 0; i < System::volume_dim; ++i) {
      for (size_t j = 0; j < System::volume_dim; ++j) {
        flux_var2->get(i, j) = offset_boundary_condition + 1.0 +
                               static_cast<double>(i + 2 * System::volume_dim);
      }
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
  }

  // Mixed conservative non-conservative system, with prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    get(*out_prim_var1) = get(*out_var3_squared) + 1.0;
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, with prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
  }

  // Conservative system, no prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    for (size_t i = 0; i < System::volume_dim; ++i) {
      flux_var1->get(i) = offset_boundary_condition + 1.0 +
                          static_cast<double>(i + System::volume_dim);
    }
    return std::nullopt;
  }

  // Conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, with prims
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    get(*out_prim_var1) = get(*out_var3_squared) + 1.0;
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  // Conservative system, with prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
  }

  // public so that GhostAndTimeDerivative can call into this
  void check_normal_vector_set_inverse_spatial_metric(
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector) const noexcept {
    CHECK_ITERABLE_APPROX(
        get(dot_product(outward_directed_normal_covector,
                        outward_directed_normal_vector)),
        DataVector(get<0>(outward_directed_normal_vector).size(), 1.0));
    for (size_t i = 0; i < inv_spatial_metric->size(); ++i) {
      (*inv_spatial_metric)[i] =
          DataVector{get<0>(outward_directed_normal_vector).size(),
                     (offset_temporaries + 1.0 + i)};
    }
  }

 private:
  bool mesh_is_moving_{false};
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID Ghost<System>::my_PUP_ID = 0;

template <typename System>
class TimeDerivative : public BoundaryCondition<System> {
 public:
  TimeDerivative() = default;
  TimeDerivative(const bool mesh_is_moving, const double expected_dt_var1)
      : mesh_is_moving_(mesh_is_moving), expected_dt_var1_(expected_dt_var1) {}
  TimeDerivative(TimeDerivative&&) noexcept = default;
  TimeDerivative& operator=(TimeDerivative&&) noexcept = default;
  TimeDerivative(const TimeDerivative&) = default;
  TimeDerivative& operator=(const TimeDerivative&) = default;
  ~TimeDerivative() override = default;

  explicit TimeDerivative(CkMigrateMessage* msg) noexcept
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, TimeDerivative);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<TimeDerivative<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::TimeDerivative;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | mesh_is_moving_;
    p | expected_dt_var1_;
  }

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Var1, Tags::Var2<System::volume_dim>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::PrimVar1, Tags::PrimVar2<System::volume_dim>>;
  using dg_interior_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Var1>>;
  using dg_interior_deriv_vars_tags = tmpl::conditional_t<
      System::system_type == SystemType::Conservative, tmpl::list<>,
      tmpl::list<::Tags::deriv<Tags::Var1, tmpl::size_t<System::volume_dim>,
                               Frame::Inertial>>>;
  using dg_gridless_tags = tmpl::list<Tags::BoundaryConditionVolumeTag>;

  // Conservative, no prims, flat background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    CHECK(volume_number == 2.5);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(var3_squared),
                          DataVector(num_pts, offset_temporaries));
    CHECK_ITERABLE_APPROX(get(var1), DataVector(num_pts, offset_evolved_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1 + i));
      for (size_t j = 0; j < num_pts; ++j) {
        // Catch doesn't allow `CHECK(a or b) so we do `CHECK((a or b))` instead
        if constexpr (System::volume_dim == 1) {
          const double normalization_factor =
              System::has_inverse_spatial_metric
                  ? sqrt(offset_temporaries + 1.0)
                  : 1.0;
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) ==
                     1.0 / normalization_factor or
                 approx(outward_directed_normal_covector.get(i)[j]) ==
                     -1.0 / normalization_factor));
        } else {
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) == 1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == -1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == 0.0));
        }
      }
    }
    CHECK_ITERABLE_APPROX(get(dt_var1), DataVector(num_pts, expected_dt_var1_));

    REQUIRE(face_mesh_velocity.has_value() == mesh_is_moving_);
    if (mesh_is_moving_) {
      for (size_t i = 0; i < System::volume_dim; ++i) {
        CHECK_ITERABLE_APPROX(
            face_mesh_velocity->get(i),
            DataVector(num_pts, gsl::at(expected_velocities, i)));
      }
    }
    get(*dt_correction_var1) = offset_boundary_condition;
    for (size_t i = 0; i < System::volume_dim; ++i) {
      dt_correction_var2->get(i) =
          offset_boundary_condition + 1.0 + static_cast<double>(i);
    }
    return std::nullopt;
  }

  // Conservative, no prims, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    return dg_time_derivative(dt_correction_var1, dt_correction_var2,
                              face_mesh_velocity,
                              outward_directed_normal_covector, var1, var2,
                              var3_squared, dt_var1, volume_number);
  }

  // Mixed and non-conservative system, flat background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Mixed and non-conservative system, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    return dg_time_derivative(dt_correction_var1, dt_correction_var2,
                              face_mesh_velocity,
                              outward_directed_normal_covector, var1, var2,
                              var3_squared, dt_var1, d_var1, volume_number);
  }

  // Mixed system with primitive vars, flat background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, prim_var1, prim_var2, var3_squared, dt_var1,
                       volume_number);
    // Sets the dt_correction again, but that's fine, values stay the same.
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, var3_squared, dt_var1, d_var1,
                       volume_number);
    return std::nullopt;
  }

  // Mixed system with primitive vars, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    return dg_time_derivative(
        dt_correction_var1, dt_correction_var2, face_mesh_velocity,
        outward_directed_normal_covector, var1, var2, prim_var1, prim_var2,
        var3_squared, dt_var1, d_var1, volume_number);
  }

  // Conservative system with primitive vars
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  // Conservative system with primitive vars, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, prim_var1, prim_var2, var3_squared, dt_var1,
                       volume_number);
    return std::nullopt;
  }

 private:
  void check_normal_vector(
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector) const noexcept {
    CHECK_ITERABLE_APPROX(
        get(dot_product(outward_directed_normal_covector,
                        outward_directed_normal_vector)),
        DataVector(get<0>(outward_directed_normal_vector).size(), 1.0));
  }

  bool mesh_is_moving_{false};
  double expected_dt_var1_{std::numeric_limits<double>::signaling_NaN()};
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID TimeDerivative<System>::my_PUP_ID = 0;

template <typename System>
class GhostAndTimeDerivative : public BoundaryCondition<System> {
 public:
  GhostAndTimeDerivative() = default;
  explicit GhostAndTimeDerivative(const bool mesh_is_moving)
      : ghost_{mesh_is_moving},
        time_derivative_{mesh_is_moving, offset_dt_evolved_vars} {}
  GhostAndTimeDerivative(GhostAndTimeDerivative&&) noexcept = default;
  GhostAndTimeDerivative& operator=(GhostAndTimeDerivative&&) noexcept =
      default;
  GhostAndTimeDerivative(const GhostAndTimeDerivative&) = default;
  GhostAndTimeDerivative& operator=(const GhostAndTimeDerivative&) = default;
  ~GhostAndTimeDerivative() override = default;

  explicit GhostAndTimeDerivative(CkMigrateMessage* msg) noexcept
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, GhostAndTimeDerivative);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<GhostAndTimeDerivative<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | ghost_;
    p | time_derivative_;
  }

  using dg_interior_evolved_variables_tags =
      typename Ghost<System>::dg_interior_evolved_variables_tags;
  using dg_interior_primitive_variables_tags =
      typename Ghost<System>::dg_interior_primitive_variables_tags;
  using dg_interior_temporary_tags =
      typename Ghost<System>::dg_interior_temporary_tags;
  using dg_interior_dt_vars_tags =
      typename Ghost<System>::dg_interior_dt_vars_tags;
  using dg_interior_deriv_vars_tags =
      typename TimeDerivative<System>::dg_interior_deriv_vars_tags;
  using dg_gridless_tags = typename Ghost<System>::dg_gridless_tags;

  // Nonconservative, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    ghost_.dg_ghost(out_var1, out_var2, out_var3_squared, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, var3_squared,
                    dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Nonconservative, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3_squared, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, var3_squared,
                    dt_var1, d_var1, volume_number);
  }

  // Mixed conservative non-conservative system, no prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    ghost_.dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, d_var1, volume_number);
  }

  // Mixed conservative non-conservative system, prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    ghost_.dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const noexcept {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, d_var1, volume_number);
  }

  // Conservative system, no prims
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    ghost_.dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, with prims
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    ghost_.dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, with prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const noexcept {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    ghost_.dg_ghost(out_var1, out_var2, flux_var1, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  template <typename... Args>
  std::optional<std::string> dg_time_derivative(Args&&... args) const noexcept {
    return time_derivative_.dg_time_derivative(std::forward<Args>(args)...);
  }

 private:
  Ghost<System> ghost_;
  TimeDerivative<System> time_derivative_;
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID GhostAndTimeDerivative<System>::my_PUP_ID = 0;

template <bool AddTypeAlias, size_t Dim>
struct InverseSpatialMetricTagImpl {
  using inverse_spatial_metric_tag = Tags::InverseSpatialMetric<Dim>;
};

template <size_t Dim>
struct InverseSpatialMetricTagImpl<false, Dim> {};

template <size_t Dim, SystemType SysType, bool HasPrimitiveVariables,
          bool HasInverseSpatialMetric>
struct System
    : public InverseSpatialMetricTagImpl<HasInverseSpatialMetric, Dim> {
  static constexpr SystemType system_type = SysType;
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveVariables;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool has_inverse_spatial_metric = HasInverseSpatialMetric;

  using boundary_correction_base =
      BoundaryCorrection<Dim, HasPrimitiveVariables, SysType,
                         HasInverseSpatialMetric>;
  using boundary_conditions_base = BoundaryCondition<System>;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Var1, Tags::Var2<Dim>>>;
  using flux_variables = tmpl::conditional_t<
      system_type == SystemType::Conservative,
      tmpl::list<Tags::Var1, Tags::Var2<Dim>>,
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<>, tmpl::list<Tags::Var2<Dim>>>>;
  using gradient_variables = tmpl::conditional_t<
      system_type == SystemType::Conservative, tmpl::list<>,
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<Tags::Var1, Tags::Var2<Dim>>,
                          tmpl::list<Tags::Var1>>>;
  using primitive_variables_tag =
      ::Tags::Variables<tmpl::list<Tags::PrimVar1, Tags::PrimVar2<Dim>>>;

  struct compute_volume_time_derivative_terms {
    using temporary_tags = tmpl::append<
        tmpl::list<Tags::Var3Squared>,
        tmpl::conditional_t<HasInverseSpatialMetric,
                            tmpl::list<Tags::InverseSpatialMetric<Dim>>,
                            tmpl::list<>>>;
  };
};

template <typename TagsList>
void fill_variables(const gsl::not_null<Variables<TagsList>*> variables,
                    const double offset) {
  double count = offset;
  tmpl::for_each<TagsList>([&count, &variables](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    auto& tensor = get<tag>(*variables);
    for (auto& tensor_component : tensor) {
      tensor_component = count;
      count += 1.0;
    }
  });
}

// Note: clang-8 wants us to capture Dim in the closures if we do `constexpr
// size_t Dim =...`, but then GCC-7 fails to build. Assigning `Dim` as a
// template parameter gets around that.
template <typename System, size_t Dim = System::volume_dim>
void test_1d(const bool moving_mesh, const dg::Formulation formulation,
             const Spectral::Quadrature quadrature) {
  CAPTURE(moving_mesh);
  CAPTURE(formulation);
  CAPTURE(quadrature);
  CAPTURE(System::has_primitive_and_conservative_vars);
  CAPTURE(System::system_type);
  // gcc-8 complains that there are no definitions for the destructors of None
  // and Periodic. We can generate them via explicit instantiations or by just
  // creating an empty dummy object.
  [[maybe_unused]] const domain::BoundaryConditions::None<
      BoundaryCondition<System>>
      instantiate_none_for_gcc_8{};
  [[maybe_unused]] const domain::BoundaryConditions::Periodic<
      BoundaryCondition<System>>
      instantiate_periodic_for_gcc_8{};
  static_assert(System::volume_dim == 1);

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;
  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre, quadrature};
  const ElementId<Dim> self_id{0, {{{1, 0}}}};
  const Element<Dim> element{self_id, {}};
  ElementMap<Dim, Frame::Grid> element_map{
      self_id, domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                   domain::CoordinateMaps::Identity<Dim>{})};
  auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});
  const double time{1.2};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity{};
  if (moving_mesh) {
    const std::array<double, 3> velocities = {{1.2, -1.4, 0.3}};
    mesh_velocity =
        tnsr::I<DataVector, Dim, Frame::Inertial>{mesh.number_of_grid_points()};
    for (size_t i = 0; i < Dim; ++i) {
      mesh_velocity->get(i) = gsl::at(velocities, i);
    }
  }
  // Set the Jacobian to not be the identity because otherwise bugs creep in
  // easily.
  ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
      inv_jacobian{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jacobian.get(i, i) = 2.0;
  }
  const auto det_inv_jacobian = determinant(inv_jacobian);
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_and_magnitude{};
  normal_covector_and_magnitude[Direction<Dim>::lower_xi()] = std::nullopt;
  normal_covector_and_magnitude[Direction<Dim>::upper_xi()] = std::nullopt;
  const double boundary_condition_volume_tag_number{2.5};
  const double boundary_correction_volume_tag_number{3.5};

  Variables<
      db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>
      dt_evolved_vars{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&dt_evolved_vars), offset_dt_evolved_vars);
  Variables<typename System::variables_tag::tags_list> evolved_vars{
      mesh.number_of_grid_points()};
  fill_variables(make_not_null(&evolved_vars), offset_evolved_vars);
  Variables<
      typename System::compute_volume_time_derivative_terms::temporary_tags>
      temporaries{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&temporaries), offset_temporaries);
  Variables<db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                             tmpl::size_t<Dim>, Frame::Inertial>>
      volume_fluxes{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&volume_fluxes), offset_volume_fluxes);
  Variables<db::wrap_tags_in<::Tags::deriv, typename System::gradient_variables,
                             tmpl::size_t<Dim>, Frame::Inertial>>
      partial_derivs{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&partial_derivs), offset_partial_derivs);
  Variables<tmpl::conditional_t<
      System::has_primitive_and_conservative_vars,
      typename System::primitive_variables_tag::tags_list, tmpl::list<>>>
      primitive_vars{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&primitive_vars), offset_primitive_vars);
  const Variables<tmpl::conditional_t<
      System::has_primitive_and_conservative_vars,
      typename System::primitive_variables_tag::tags_list, tmpl::list<>>>*
      primitive_vars_ptr =
          System::has_primitive_and_conservative_vars ? &primitive_vars
                                                      : nullptr;
  constexpr bool has_prims = System::has_primitive_and_conservative_vars;
  using BndryTerms = BoundaryTerms<Dim, has_prims, System::system_type,
                                   System::has_inverse_spatial_metric>;

  Domain<Dim> domain{};
  {
    // For the initial tests, set the boundary conditions to:
    // lower_xi: Outflow
    // upper_xi: Outflow
    DirectionMap<Dim,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        boundary_conditions{};
    boundary_conditions[Direction<Dim>::lower_xi()] =
        std::make_unique<Outflow<System>>(moving_mesh);
    boundary_conditions[Direction<Dim>::upper_xi()] =
        std::make_unique<Outflow<System>>(moving_mesh);
    domain = Domain<Dim>{make_vector(Block<Dim>{
        domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{}),
        0,
        {},
        std::move(boundary_conditions)})};
    domain.inject_time_dependent_map_for_block(
        0, grid_to_inertial_map->get_clone());
  }

  using simple_tags = tmpl::list<
      domain::Tags::Domain<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      ::Tags::Time, domain::Tags::FunctionsOfTime,
      domain::Tags::MeshVelocity<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
      typename System::variables_tag, dt_variables_tag,
      Tags::BoundaryConditionVolumeTag, Tags::BoundaryCorrectionVolumeTag,
      ::dg::Tags::Formulation>;
  using compute_tags = tmpl::list<>;

  auto box = db::create<simple_tags, compute_tags>(
      std::move(domain), mesh, element, std::move(element_map),
      grid_to_inertial_map->get_clone(), time,
      clone_unique_ptrs(functions_of_time), mesh_velocity, inv_jacobian,
      det_inv_jacobian, normal_covector_and_magnitude, evolved_vars,
      dt_evolved_vars, boundary_condition_volume_tag_number,
      boundary_correction_volume_tag_number, formulation);

  {
    INFO("Outflow only");
    // Outflow both sides, dt in volume shouldn't change.
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim>(
            make_not_null(&box), BndryTerms{moving_mesh, 0.0}, temporaries,
            volume_fluxes, partial_derivs, primitive_vars_ptr);
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        DataVector(mesh.number_of_grid_points(), offset_dt_evolved_vars));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
                            DataVector(mesh.number_of_grid_points(),
                                       offset_dt_evolved_vars + 1 + i));
    }
  }

  const auto expected_ghost_dt_correction = [&box, &formulation, &mesh,
                                             &quadrature](
                                                const auto& ghost_direction) {
    Variables<tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
        expected_on_boundary{mesh.slice_away(ghost_direction.dimension())
                                 .number_of_grid_points()};
    get(get<::Tags::dt<Tags::Var1>>(expected_on_boundary)) =
        offset_boundary_correction *
        (formulation == dg::Formulation::WeakInertial ? 2.0 : 1.0);
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::dt<Tags::Var2<Dim>>>(expected_on_boundary).get(i) =
          offset_boundary_correction + 1.0 + i;
    }
    // lift into volume and add to volume time derivative
    Variables<tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
        expected_dt_volume_correction{mesh.number_of_grid_points(), 0.0};
    const Scalar<DataVector>& volume_det_inv_jacobian =
        db::get<domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(
            box);
    const Scalar<DataVector>& magnitude_of_interior_face_normal =
        get<evolution::dg::Tags::MagnitudeOfNormal>(
            *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box)
                 .at(ghost_direction));
    if (quadrature == Spectral::Quadrature::Gauss) {
      Scalar<DataVector> face_det_inv_jacobian{
          mesh.slice_away(ghost_direction.dimension()).number_of_grid_points()};
      const Matrix identity{};
      auto interpolation_matrices = make_array<Dim>(std::cref(identity));
      const std::pair<Matrix, Matrix>& matrices =
          Spectral::boundary_interpolation_matrices(
              mesh.slice_through(ghost_direction.dimension()));
      gsl::at(interpolation_matrices, ghost_direction.dimension()) =
          ghost_direction.side() == Side::Upper ? matrices.second
                                                : matrices.first;
      apply_matrices(make_not_null(&get(face_det_inv_jacobian)),
                     interpolation_matrices, get(volume_det_inv_jacobian),
                     mesh.extents());
      Scalar<DataVector> face_det_jacobian{1.0 / get(face_det_inv_jacobian)};

      evolution::dg::lift_boundary_terms_gauss_points(
          make_not_null(&expected_dt_volume_correction),
          volume_det_inv_jacobian, mesh, ghost_direction, expected_on_boundary,
          magnitude_of_interior_face_normal, face_det_jacobian);
    } else {
      ::dg::lift_flux(make_not_null(&expected_on_boundary),
                      mesh.extents(ghost_direction.dimension()),
                      magnitude_of_interior_face_normal);
      add_slice_to_data(make_not_null(&expected_dt_volume_correction),
                        expected_on_boundary, mesh.extents(),
                        ghost_direction.dimension(),
                        index_to_slice_at(mesh.extents(), ghost_direction));
    }
    return expected_dt_volume_correction;
  };

  const auto check_outflow_and_ghost = [&box, &dt_evolved_vars,
                                        &expected_ghost_dt_correction,
                                        &moving_mesh, &partial_derivs,
                                        &primitive_vars_ptr, &temporaries,
                                        &volume_fluxes](const Direction<Dim>&
                                                            outflow_direction) {
    INFO("Ghost");
    CAPTURE(outflow_direction);
    db::mutate<domain::Tags::Domain<Dim>, dt_variables_tag>(
        make_not_null(&box),
        [&moving_mesh, &outflow_direction](const auto domain_ptr,
                                           const auto dt_vars_ptr) {
          DirectionMap<Dim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
              boundary_conditions{};
          boundary_conditions[outflow_direction.opposite()] =
              std::make_unique<Ghost<System>>(moving_mesh);
          boundary_conditions[outflow_direction] =
              std::make_unique<Outflow<System>>(moving_mesh);
          *domain_ptr = Domain<Dim>{make_vector(Block<Dim>{
              domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                  domain::CoordinateMaps::Identity<Dim>{}),
              0,
              {},
              std::move(boundary_conditions)})};

          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        });
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim>(
            make_not_null(&box),
            BndryTerms{moving_mesh, outflow_direction.opposite().sign()},
            temporaries, volume_fluxes, partial_derivs, primitive_vars_ptr);

    auto expected_dt_evolved_vars = dt_evolved_vars;
    expected_dt_evolved_vars +=
        expected_ghost_dt_correction(outflow_direction.opposite());
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
          get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars).get(i));
    }
  };
  // Outflow +xi, Ghost -xi
  check_outflow_and_ghost(Direction<Dim>::upper_xi());
  // Ghost +xi, Outflow -xi
  check_outflow_and_ghost(Direction<Dim>::lower_xi());

  const auto expected_time_derivative_dt_correction = [&mesh, &quadrature](
                                                          const auto&
                                                              dt_direction) {
    Variables<tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
        expected_dt_volume_correction{mesh.number_of_grid_points(), 0.0};
    const Mesh<Dim> mesh_gl{5, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    Variables<
        db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>
        dt_correction_gl{mesh_gl.number_of_grid_points(), 0.0};
    const size_t boundary_index =
        dt_direction.side() == Side::Lower
            ? 0
            : mesh_gl.extents(dt_direction.dimension()) - 1;
    get(get<::Tags::dt<Tags::Var1>>(dt_correction_gl))[boundary_index] +=
        offset_boundary_condition;
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::dt<Tags::Var2<Dim>>>(dt_correction_gl)
          .get(i)[boundary_index] += offset_boundary_condition + 1.0 + i;
    }
    if (quadrature == Spectral::Quadrature::GaussLobatto) {
      expected_dt_volume_correction += dt_correction_gl;
    } else {
      // Interpolate to Gauss mesh
      expected_dt_volume_correction +=
          intrp::RegularGrid<Dim>{mesh_gl, mesh}.interpolate(dt_correction_gl);
    }
    return expected_dt_volume_correction;
  };

  const auto check_outflow_and_dt = [&box, &dt_evolved_vars,
                                     &expected_time_derivative_dt_correction,
                                     &moving_mesh, &partial_derivs,
                                     &primitive_vars_ptr, &temporaries,
                                     &volume_fluxes](const Direction<Dim>&
                                                         outflow_direction) {
    INFO("TimeDerivative");
    CAPTURE(outflow_direction);
    db::mutate<domain::Tags::Domain<Dim>, dt_variables_tag>(
        make_not_null(&box),
        [&moving_mesh, &outflow_direction](const auto domain_ptr,
                                           const auto dt_vars_ptr) {
          DirectionMap<Dim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
              boundary_conditions{};
          boundary_conditions[outflow_direction.opposite()] =
              std::make_unique<TimeDerivative<System>>(moving_mesh,
                                                       offset_dt_evolved_vars);
          boundary_conditions[outflow_direction] =
              std::make_unique<Outflow<System>>(moving_mesh);
          *domain_ptr = Domain<Dim>{make_vector(Block<Dim>{
              domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                  domain::CoordinateMaps::Identity<Dim>{}),
              0,
              {},
              std::move(boundary_conditions)})};

          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        });
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim>(
            make_not_null(&box),
            BndryTerms{moving_mesh, outflow_direction.opposite().sign()},
            temporaries, volume_fluxes, partial_derivs, primitive_vars_ptr);

    auto expected_dt_evolved_vars = dt_evolved_vars;
    expected_dt_evolved_vars +=
        expected_time_derivative_dt_correction(outflow_direction.opposite());
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
          get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars).get(i));
    }
  };
  // Outflow +xi, TimeDerivative -xi
  check_outflow_and_dt(Direction<Dim>::upper_xi());
  // Outflow -xi, TimeDerivative +xi
  check_outflow_and_dt(Direction<Dim>::lower_xi());

  const auto check_ghost_and_dt_opposite =
      [&box, &dt_evolved_vars, &expected_ghost_dt_correction,
       &expected_time_derivative_dt_correction, &mesh, &moving_mesh,
       &partial_derivs, &primitive_vars_ptr, &temporaries,
       &volume_fluxes](const Direction<Dim>& ghost_direction) {
        INFO("Ghost and TimeDerivative on opposite sides");
        CAPTURE(ghost_direction);
        auto expected_dt_evolved_vars = dt_evolved_vars;
        expected_dt_evolved_vars +=
            expected_ghost_dt_correction(ghost_direction);

        // Project to the boundary to figure out what will be the projected
        // dt_var1 passed into the time derivative boundary condition. This is
        // necessary because we apply _and lift_ the Ghost boundary correction
        // before the TimeDerivative correction. This order is determined in the
        // BoundaryCondition base class's `creatable_classes` typelist.
        Variables<
            tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
            expected_dt_on_boundary{mesh.slice_away(ghost_direction.dimension())
                                        .number_of_grid_points()};
        evolution::dg::project_contiguous_data_to_boundary(
            make_not_null(&expected_dt_on_boundary), expected_dt_evolved_vars,
            mesh, ghost_direction.opposite());

        db::mutate<domain::Tags::Domain<Dim>, dt_variables_tag>(
            make_not_null(&box),
            [&expected_dt_var1 =
                 get<::Tags::dt<Tags::Var1>>(expected_dt_on_boundary),
             &moving_mesh,
             &ghost_direction](const auto domain_ptr, const auto dt_vars_ptr) {
              DirectionMap<Dim,
                           std::unique_ptr<
                               domain::BoundaryConditions::BoundaryCondition>>
                  boundary_conditions{};
              boundary_conditions[ghost_direction.opposite()] =
                  std::make_unique<TimeDerivative<System>>(
                      moving_mesh, get(expected_dt_var1)[0]);
              boundary_conditions[ghost_direction] =
                  std::make_unique<Ghost<System>>(moving_mesh);
              *domain_ptr = Domain<Dim>{make_vector(
                  Block<Dim>{domain::make_coordinate_map_base<Frame::Logical,
                                                              Frame::Inertial>(
                                 domain::CoordinateMaps::Identity<Dim>{}),
                             0,
                             {},
                             std::move(boundary_conditions)})};

              fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
            });
        evolution::dg::Actions::detail::
            apply_boundary_conditions_on_all_external_faces<System, Dim>(
                make_not_null(&box),
                BndryTerms{moving_mesh, ghost_direction.sign()}, temporaries,
                volume_fluxes, partial_derivs, primitive_vars_ptr);

        expected_dt_evolved_vars +=
            expected_time_derivative_dt_correction(ghost_direction.opposite());

        CHECK_ITERABLE_APPROX(
            get(get<::Tags::dt<Tags::Var1>>(box)),
            get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
        for (size_t i = 0; i < Dim; ++i) {
          CHECK_ITERABLE_APPROX(
              get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
              get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars)
                  .get(i));
        }
      };
  // Ghost +xi, TimeDerivative -xi
  check_ghost_and_dt_opposite(Direction<Dim>::upper_xi());
  // Ghost -xi, TimeDerivative +xi
  check_ghost_and_dt_opposite(Direction<Dim>::lower_xi());

  const auto check_ghost_and_dt_combined_bc =
      [&box, &dt_evolved_vars, &expected_ghost_dt_correction,
       &expected_time_derivative_dt_correction, &moving_mesh, &partial_derivs,
       &primitive_vars_ptr, &temporaries,
       &volume_fluxes](const Direction<Dim>& outflow_direction) {
        INFO("GhostAndTimeDerivative combined on one side");
        CAPTURE(outflow_direction);
        // Since the Ghost and TimeDerivative are applied in the same direction
        // they both receive the dt_vars _without_ either boundary condition
        // applied, which is different from way Ghost and TimeDerivative are
        // applied in different directions.
        db::mutate<domain::Tags::Domain<Dim>, dt_variables_tag>(
            make_not_null(&box),
            [&moving_mesh, &outflow_direction](const auto domain_ptr,
                                               const auto dt_vars_ptr) {
              DirectionMap<Dim,
                           std::unique_ptr<
                               domain::BoundaryConditions::BoundaryCondition>>
                  boundary_conditions{};
              boundary_conditions[outflow_direction.opposite()] =
                  std::make_unique<GhostAndTimeDerivative<System>>(moving_mesh);
              boundary_conditions[outflow_direction] =
                  std::make_unique<Outflow<System>>(moving_mesh);
              *domain_ptr = Domain<Dim>{make_vector(
                  Block<Dim>{domain::make_coordinate_map_base<Frame::Logical,
                                                              Frame::Inertial>(
                                 domain::CoordinateMaps::Identity<Dim>{}),
                             0,
                             {},
                             std::move(boundary_conditions)})};

              fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
            });
        evolution::dg::Actions::detail::
            apply_boundary_conditions_on_all_external_faces<System, Dim>(
                make_not_null(&box),
                BndryTerms{moving_mesh, outflow_direction.opposite().sign()},
                temporaries, volume_fluxes, partial_derivs, primitive_vars_ptr);

        auto expected_dt_evolved_vars = dt_evolved_vars;
        expected_dt_evolved_vars += expected_time_derivative_dt_correction(
            outflow_direction.opposite());
        expected_dt_evolved_vars +=
            expected_ghost_dt_correction(outflow_direction.opposite());

        CHECK_ITERABLE_APPROX(
            get(get<::Tags::dt<Tags::Var1>>(box)),
            get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
        for (size_t i = 0; i < Dim; ++i) {
          CHECK_ITERABLE_APPROX(
              get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
              get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars)
                  .get(i));
        }
      };
  // Outflow +xi, GhostAndTimeDerivative -xi
  check_ghost_and_dt_combined_bc(Direction<Dim>::upper_xi());
  // Outflow -xi, GhostAndTimeDerivative +xi
  check_ghost_and_dt_combined_bc(Direction<Dim>::lower_xi());
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ComputeTimeDerivative.BoundaryConditions",
                  "[Unit][Evolution][Actions]") {
  // The test proceeds as follows:
  //
  // 1. prepare all the data in the DataBox
  // 2. call the boundary condition function, which should apply the boundary
  //    condition. We do so switching around the direction the boundary
  //    condition is applied in, checking different ones on each side.
  // 3. inside the boundary conditions we check we have received the expected
  //    values of the different tags
  // 4. we return pre-determined numbers so that we can check the time
  //    derivatives changed in the expected way given the numbers.
  //
  // Notes:
  // - the test is currently only in 1d, but most (if not all) places that need
  //   generalization have a `static_assert(Dim == 1)`. Going to more dimensions
  //   is straightforward but _extremely_ tedious.
  for (const bool moving_mesh : {true, false}) {
    for (const dg::Formulation formulation :
         {dg::Formulation::WeakInertial, dg::Formulation::StrongInertial}) {
      for (const Spectral::Quadrature quadrature :
           {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
        // Second last template parameter on System:
        // - true: has primitive variables
        // - false: no primitive variables

        // last template parameter on System being `false` means flat background
        test_1d<System<1, SystemType::Conservative, false, false>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Conservative, true, false>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Nonconservative, false, false>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Mixed, false, false>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Mixed, true, false>>(
            moving_mesh, formulation, quadrature);

        // last template parameter on System being `true` means curved
        // background
        test_1d<System<1, SystemType::Conservative, false, true>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Conservative, true, true>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Nonconservative, false, true>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Mixed, false, true>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Mixed, true, true>>(
            moving_mesh, formulation, quadrature);
      }
    }
  }
}
}  // namespace
