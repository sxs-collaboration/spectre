// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct VolumeDouble {
  double value;
};

struct VolumeDoubleConversion {
  // convert to a std::array to test non-trivial type conversion
  using unpacked_container = std::array<double, 1>;
  using packed_container = VolumeDouble;
  using packed_type = double;

  static inline unpacked_container unpack(const packed_container packed,
                                          const size_t /*grid_point_index*/) {
    return {{packed.value}};
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container unpacked,
                          const size_t /*grid_point_index*/) {
    packed->value = unpacked[0];
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

namespace Tags {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <typename Type>
struct VolumeDouble : db::SimpleTag {
  using type = Type;
};

template <size_t Dim>
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct AuxVar : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <typename Type>
struct AuxBoundaryDouble : db::SimpleTag {
  using type = Type;
};
}  // namespace Tags

template <size_t Dim, bool IncludeTypeAlias>
struct InverseSpatialMetric {};

template <size_t Dim>
struct InverseSpatialMetric<Dim, true> {
  using inverse_spatial_metric_tag = Tags::InverseSpatialMetric<Dim>;
};

template <size_t Dim, bool CurvedBackground>
struct System : public InverseSpatialMetric<Dim, CurvedBackground> {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Var1, Tags::Var2<Dim>>>;
  using flux_variables = tmpl::list<Tags::Var1, Tags::Var2<Dim>>;
  using gradient_variables = tmpl::list<>;
  using sourced_variables = tmpl::list<>;

  struct TimeDerivativeTerms {
    using temporary_tags = tmpl::list<>;
  };

  using compute_volume_time_derivative_terms = TimeDerivativeTerms;
};

template <size_t Dim>
struct AuxSystem {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Var1, Tags::Var2<Dim>>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  using sourced_variables = tmpl::list<>;
  using auxiliary_variables = tmpl::list<Tags::AuxVar<Dim>>;

  struct TimeDerivativeTerms {
    using temporary_tags = tmpl::list<>;
  };

  using compute_volume_time_derivative_terms = TimeDerivativeTerms;
};

struct CorrectionBase : public PUP::able {
  CorrectionBase() = default;
  CorrectionBase(const CorrectionBase&) = default;
  CorrectionBase& operator=(const CorrectionBase&) = default;
  CorrectionBase(CorrectionBase&&) = default;
  CorrectionBase& operator=(CorrectionBase&&) = default;
  ~CorrectionBase() override = default;

  explicit CorrectionBase(CkMigrateMessage* msg) : PUP::able(msg) {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_abstract(CorrectionBase);  // NOLINT
#pragma GCC diagnostic pop

  virtual std::unique_ptr<CorrectionBase> get_clone() const = 0;
};

template <size_t Dim, typename VolumeDoubleType>
struct Correction final : public CorrectionBase {
 private:
  struct AbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using dg_package_field_tags =
      tmpl::list<Tags::Var1, ::Tags::NormalDotFlux<Tags::Var1>, Tags::Var2<Dim>,
                 ::Tags::NormalDotFlux<Tags::Var2<Dim>>, AbsCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags =
      tmpl::list<Tags::VolumeDouble<VolumeDoubleType>>;
  using dg_boundary_terms_volume_tags =
      tmpl::list<Tags::VolumeDouble<VolumeDoubleType>>;

  Correction() = default;
  Correction(const Correction&) = default;
  Correction& operator=(const Correction&) = default;
  Correction(Correction&&) = default;
  Correction& operator=(Correction&&) = default;
  ~Correction() override = default;

  std::unique_ptr<CorrectionBase> get_clone() const override {
    return std::make_unique<Correction>(*this);
  }

  explicit Correction(CkMigrateMessage* msg) : CorrectionBase(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Correction);  // NOLINT
  void pup(PUP::er& p) override { CorrectionBase::pup(p); }

  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_var1,
      const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_var1,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_var2,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,
      const Scalar<DataVector>& var1,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::Ij<DataVector, Dim, Frame::Inertial>& flux_var2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const VolumeDoubleType volume_double_in) const {
    double volume_double = 0.0;
    if constexpr (std::is_same_v<double, VolumeDoubleType>) {
      volume_double = volume_double_in;
    } else {
      volume_double = volume_double_in.value;
    }
    *packaged_var1 = var1;
    *packaged_var2 = var2;
    dot_product(packaged_normal_dot_flux_var1, flux_var1, normal_covector);
    for (size_t i = 0; i < Dim; ++i) {
      packaged_normal_dot_flux_var2->get(i) =
          flux_var2.get(i, 0) * get<0>(normal_covector);
      for (size_t j = 1; j < Dim; ++j) {
        packaged_normal_dot_flux_var2->get(i) +=
            flux_var2.get(i, j) * normal_covector.get(j);
      }
    }

    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_abs_char_speed) =
          abs(volume_double * get(var1) - get(*normal_dot_mesh_velocity));
    } else {
      get(*packaged_abs_char_speed) = abs(volume_double * get(var1));
    }
    return max(get(*packaged_abs_char_speed));
  }

  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_var1,
      const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_var1,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_var2,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,
      const Scalar<DataVector>& var1,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::Ij<DataVector, Dim, Frame::Inertial>& flux_var2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const VolumeDoubleType volume_double_in) const {
    const double max_speed = dg_package_data(
        packaged_var1, packaged_normal_dot_flux_var1, packaged_var2,
        packaged_normal_dot_flux_var2, packaged_abs_char_speed, var1, var2,
        flux_var1, flux_var2, normal_covector, mesh_velocity,
        normal_dot_mesh_velocity, volume_double_in);

    // We add the normal vector to the flux just to verify that it is being
    // used. This is total nonsense in terms of physics.
    for (size_t i = 0; i < Dim; ++i) {
      packaged_normal_dot_flux_var2->get(i) += normal_vector.get(i);
    }
    return max_speed;
  }

  void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> boundary_correction_var1,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var2,
      const Scalar<DataVector>& var1_int,
      const Scalar<DataVector>& normal_dot_flux_var1_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var2_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_var2_int,
      const Scalar<DataVector>& abs_char_speed_int,
      const Scalar<DataVector>& var1_ext,
      const Scalar<DataVector>& normal_dot_flux_var1_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var2_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_var2_ext,
      const Scalar<DataVector>& abs_char_speed_ext,
      const dg::Formulation dg_formulation,
      const VolumeDoubleType volume_double_in) const {
    // The below code is a Rusanov solver.
    if (dg_formulation == dg::Formulation::WeakInertial) {
      get(*boundary_correction_var1) =
          0.5 *
              (get(normal_dot_flux_var1_int) - get(normal_dot_flux_var1_ext)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (get(var1_ext) - get(var1_int));
      for (size_t i = 0; i < Dim; ++i) {
        boundary_correction_var2->get(i) =
            0.5 * (normal_dot_flux_var2_int.get(i) -
                   normal_dot_flux_var2_ext.get(i)) -
            0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                (var2_ext.get(i) - var2_int.get(i));
      }
    } else {
      get(*boundary_correction_var1) =
          -0.5 *
              (get(normal_dot_flux_var1_int) + get(normal_dot_flux_var1_ext)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (get(var1_ext) - get(var1_int));
      for (size_t i = 0; i < Dim; ++i) {
        boundary_correction_var2->get(i) =
            -0.5 * (normal_dot_flux_var2_int.get(i) +
                    normal_dot_flux_var2_ext.get(i)) -
            0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                (var2_ext.get(i) - var2_int.get(i));
      }
    }
    if constexpr (std::is_same_v<double, VolumeDoubleType>) {
      CHECK(volume_double_in == 2.3);
    } else {
      CHECK(volume_double_in.value == 2.3);
    }
  }
};

template <size_t Dim, typename VolumeDoubleType>
PUP::able::PUP_ID Correction<Dim, VolumeDoubleType>::my_PUP_ID = 0;

template <size_t Dim, typename VolumeDoubleType>
struct AuxCorrection final : public CorrectionBase {
 private:
  struct NormalDotAux : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct Var1TimesNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };

  static double as_double(const VolumeDoubleType value) {
    if constexpr (std::is_same_v<double, VolumeDoubleType>) {
      return value;
    } else {
      return value.value;
    }
  }

 public:
  // Physical pass interface.
  using dg_package_field_tags =
      tmpl::list<Tags::Var1, NormalDotAux, Var1TimesNormal>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags =
      tmpl::list<Tags::VolumeDouble<VolumeDoubleType>>;
  using dg_boundary_terms_volume_tags =
      tmpl::list<Tags::AuxBoundaryDouble<VolumeDoubleType>>;

  // Auxiliary pass interface.
  using dg_auxiliary_package_field_tags =
      tmpl::list<Tags::Var1, Var1TimesNormal>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags =
      tmpl::list<Tags::VolumeDouble<VolumeDoubleType>>;
  using dg_auxiliary_boundary_terms_volume_tags =
      tmpl::list<Tags::AuxBoundaryDouble<VolumeDoubleType>>;

  AuxCorrection() = default;
  AuxCorrection(const AuxCorrection&) = default;
  AuxCorrection& operator=(const AuxCorrection&) = default;
  AuxCorrection(AuxCorrection&&) = default;
  AuxCorrection& operator=(AuxCorrection&&) = default;
  ~AuxCorrection() override = default;

  std::unique_ptr<CorrectionBase> get_clone() const override {
    return std::make_unique<AuxCorrection>(*this);
  }

  explicit AuxCorrection(CkMigrateMessage* msg) : CorrectionBase(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AuxCorrection);  // NOLINT
  void pup(PUP::er& p) override { CorrectionBase::pup(p); }

  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_var1,
      const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_aux,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          var1_times_normal,
      const Scalar<DataVector>& var1,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*var2*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& aux_var,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const VolumeDoubleType volume_double) const {
    get(*packaged_var1) = get(var1);
    get(*packaged_normal_dot_aux) = get<0>(aux_var) * get<0>(normal_covector);
    for (size_t i = 1; i < Dim; ++i) {
      get(*packaged_normal_dot_aux) += aux_var.get(i) * normal_covector.get(i);
    }
    for (size_t i = 0; i < Dim; ++i) {
      var1_times_normal->get(i) = get(var1) * normal_covector.get(i);
    }
    CHECK(as_double(volume_double) == 2.3);
    return 1.0;
  }

  void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> boundary_correction_var1,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var2,
      const Scalar<DataVector>& /*packaged_var1_int*/,
      const Scalar<DataVector>& normal_dot_aux_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var1_times_normal_int,
      const Scalar<DataVector>& /*packaged_var1_ext*/,
      const Scalar<DataVector>& normal_dot_aux_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var1_times_normal_ext,
      const dg::Formulation /*dg_formulation*/,
      const VolumeDoubleType aux_boundary_double) const {
    get(*boundary_correction_var1) =
        -0.5 * (get(normal_dot_aux_int) + get(normal_dot_aux_ext));
    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_var2->get(i) =
          -0.5 * (var1_times_normal_int.get(i) + var1_times_normal_ext.get(i));
    }
    CHECK(as_double(aux_boundary_double) == 3.7);
  }

  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_var1,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          var1_times_normal,
      const Scalar<DataVector>& var1,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*var2*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const VolumeDoubleType volume_double) const {
    get(*packaged_var1) = get(var1);
    for (size_t i = 0; i < Dim; ++i) {
      var1_times_normal->get(i) = get(var1) * normal_covector.get(i);
    }
    CHECK(as_double(volume_double) == 2.3);
    return 0.0;
  }

  void dg_auxiliary_boundary_terms(
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          aux_var_correction,
      const Scalar<DataVector>& /*packaged_var1_int*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var1_times_normal_int,
      const Scalar<DataVector>& /*packaged_var1_ext*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& var1_times_normal_ext,
      const dg::Formulation /*dg_formulation*/,
      const VolumeDoubleType aux_boundary_double) const {
    for (size_t i = 0; i < Dim; ++i) {
      aux_var_correction->get(i) =
          0.5 * (var1_times_normal_int.get(i) + var1_times_normal_ext.get(i));
    }
    CHECK(as_double(aux_boundary_double) == 3.7);
  }
};

template <size_t Dim, typename VolumeDoubleType>
// NOLINTNEXTLINE
PUP::able::PUP_ID AuxCorrection<Dim, VolumeDoubleType>::my_PUP_ID = 0;

template <size_t Dim, bool CurvedBackground, typename VolumeDoubleType>
void test_impl(const gsl::not_null<std::mt19937*> gen) {
  PUPable_reg(SINGLE_ARG(Correction<Dim, VolumeDoubleType>));
  const Correction<Dim, VolumeDoubleType> correction{};
  const Mesh<Dim - 1> face_mesh{Dim * Dim, Spectral::Basis::Legendre,
                                Spectral::Quadrature::Gauss};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      System<Dim, CurvedBackground>>(
      gen, correction, face_mesh,
      tuples::TaggedTuple<Tags::VolumeDouble<VolumeDoubleType>>{
          VolumeDoubleType{2.3}},
      tuples::TaggedTuple<>{});

  const std::string curved_suffix =
      CurvedBackground ? std::string{"_curved"} : std::string{""};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      System<Dim, CurvedBackground>, tmpl::list<VolumeDoubleConversion>>(
      gen, "BoundaryCorrectionsHelper", "dg_package_data" + curved_suffix,
      "dg_boundary_terms", correction, face_mesh,
      tuples::TaggedTuple<Tags::VolumeDouble<VolumeDoubleType>>{
          VolumeDoubleType{2.3}},
      tuples::TaggedTuple<>{});
}

template <size_t Dim, typename VolumeDoubleType>
void test_aux_impl(const gsl::not_null<std::mt19937*> gen) {
  PUPable_reg(SINGLE_ARG(AuxCorrection<Dim, VolumeDoubleType>));
  const AuxCorrection<Dim, VolumeDoubleType> correction{};
  const Mesh<Dim - 1> face_mesh{Dim * Dim, Spectral::Basis::Legendre,
                                Spectral::Quadrature::Gauss};

  const tuples::TaggedTuple<Tags::VolumeDouble<VolumeDoubleType>,
                            Tags::AuxBoundaryDouble<VolumeDoubleType>>
      volume_data{VolumeDoubleType{2.3}, VolumeDoubleType{3.7}};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      AuxSystem<Dim>>(gen, correction, face_mesh, volume_data,
                      tuples::TaggedTuple<>{});

  TestHelpers::evolution::dg::test_auxiliary_boundary_correction_conservation<
      AuxSystem<Dim>>(gen, correction, face_mesh, volume_data,
                      tuples::TaggedTuple<>{});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      AuxSystem<Dim>, tmpl::list<VolumeDoubleConversion>>(
      gen, "BoundaryCorrectionsHelper", "dg_package_data_aux_system",
      "dg_boundary_terms_aux_system", correction, face_mesh, volume_data,
      tuples::TaggedTuple<>{}, 1.0e-12, std::make_tuple(1.25));

  TestHelpers::evolution::dg::test_auxiliary_boundary_correction_with_python<
      AuxSystem<Dim>, tmpl::list<VolumeDoubleConversion>>(
      gen, "BoundaryCorrectionsHelper", "dg_auxiliary_package_data_aux_system",
      "dg_auxiliary_boundary_terms_aux_system", correction, face_mesh,
      volume_data, tuples::TaggedTuple<>{}, 1.0e-12, std::make_tuple(1.25));
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen) {
  test_impl<Dim, false, double>(gen);
  test_impl<Dim, false, VolumeDouble>(gen);

  test_impl<Dim, true, double>(gen);
  test_impl<Dim, true, VolumeDouble>(gen);

  test_aux_impl<Dim, double>(gen);
  test_aux_impl<Dim, VolumeDouble>(gen);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryCorrectionsHelper",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/DiscontinuousGalerkin/"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen));
  test<2>(make_not_null(&gen));
  test<3>(make_not_null(&gen));
}
