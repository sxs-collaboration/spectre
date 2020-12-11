// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

struct VolumeDouble : db::SimpleTag {
  using type = double;
};
}  // namespace Tags

template <size_t Dim>
struct System {
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
struct Correction {
 private:
  struct AbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using dg_package_field_tags =
      tmpl::list<Tags::Var1, ::Tags::NormalDotFlux<Tags::Var1>, Tags::Var2<Dim>,
                 ::Tags::NormalDotFlux<Tags::Var2<Dim>>, AbsCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<Tags::VolumeDouble>;

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
      const boost::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const boost::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const double volume_double) const noexcept {
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

    if (static_cast<bool>(normal_dot_mesh_velocity)) {
      get(*packaged_abs_char_speed) =
          abs(volume_double * get(var1) - get(*normal_dot_mesh_velocity));
    } else {
      get(*packaged_abs_char_speed) = abs(volume_double * get(var1));
    }
    return max(get(*packaged_abs_char_speed));
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
      const dg::Formulation dg_formulation) const noexcept {
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
  }
};

template <size_t Dim>
void test() {
  const Correction<Dim> correction{};
  const Mesh<Dim - 1> face_mesh{Dim * Dim, Spectral::Basis::Legendre,
                                Spectral::Quadrature::Gauss};
  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      System<Dim>>(correction, face_mesh,
                   tuples::TaggedTuple<Tags::VolumeDouble>{2.3});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryCorrectionsHelper",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
