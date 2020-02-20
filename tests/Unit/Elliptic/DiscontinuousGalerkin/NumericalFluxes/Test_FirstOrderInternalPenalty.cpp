// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Utilities/MakeString.hpp"
#include "tests/Unit/Elliptic/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AuxiliaryFieldTag : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

struct AnArgument : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct Fluxes {
  using argument_tags = tmpl::list<AnArgument>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const double an_argument,
      const tnsr::i<DataVector, Dim>& auxiliary_field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_field->get(d) = auxiliary_field.get(d) * an_argument;
    }
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_aux_field,
      const double an_argument, const Scalar<DataVector>& field) {
    std::fill(flux_for_aux_field->begin(), flux_for_aux_field->end(), 0.);
    for (size_t d = 0; d < Dim; d++) {
      flux_for_aux_field->get(d, d) = get(field) * an_argument;
    }
  }
};

template <size_t Dim>
struct FluxesComputerTag : db::SimpleTag {
  using type = Fluxes<Dim>;
};

template <size_t Dim>
void apply_ip_flux(
    const gsl::not_null<Scalar<DataVector>*> n_dot_num_f_field,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> n_dot_num_f_aux,
    const tnsr::i<DataVector, Dim>& n_dot_aux_flux_int,
    const tnsr::i<DataVector, Dim>& n_dot_aux_flux_ext,
    const tnsr::i<DataVector, Dim>& div_aux_flux_int,
    const tnsr::i<DataVector, Dim>& div_aux_flux_ext,
    const double fluxes_argument, const double penalty_parameter,
    const tnsr::i<DataVector, Dim>& face_normal_int) noexcept {
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, FluxesComputerTag<Dim>, tmpl::list<ScalarFieldTag>,
          tmpl::list<AuxiliaryFieldTag<Dim>>>;

  NumericalFlux numerical_flux{penalty_parameter};
  Fluxes<Dim> fluxes_computer{};
  using PackagedData = Variables<typename NumericalFlux::package_tags>;

  auto packaged_data_interior = make_with_value<PackagedData>(
      face_normal_int, std::numeric_limits<double>::signaling_NaN());
  numerical_flux.package_data(
      make_not_null(&packaged_data_interior), n_dot_aux_flux_int,
      div_aux_flux_int, fluxes_computer, fluxes_argument, face_normal_int);
  auto packaged_data_exterior = make_with_value<PackagedData>(
      face_normal_int, std::numeric_limits<double>::signaling_NaN());
  tnsr::i<DataVector, Dim> face_normal_ext{face_normal_int};
  for (size_t d = 0; d < Dim; d++) {
    face_normal_ext.get(d) *= -1.;
  }
  numerical_flux.package_data(
      make_not_null(&packaged_data_exterior), n_dot_aux_flux_ext,
      div_aux_flux_ext, fluxes_computer, fluxes_argument, face_normal_ext);

  EllipticNumericalFluxesTestHelpers::apply_numerical_flux(
      numerical_flux, packaged_data_interior, packaged_data_exterior,
      n_dot_num_f_field, n_dot_num_f_aux);
}

template <size_t Dim>
void apply_ip_dirichlet_flux(
    const gsl::not_null<Scalar<DataVector>*> n_dot_num_f_field,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> n_dot_num_f_aux,
    const Scalar<DataVector>& dirichlet_field, const double fluxes_argument,
    const double penalty_parameter,
    const tnsr::i<DataVector, Dim>& face_normal) noexcept {
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, FluxesComputerTag<Dim>, tmpl::list<ScalarFieldTag>,
          tmpl::list<AuxiliaryFieldTag<Dim>>>;

  NumericalFlux numerical_flux{penalty_parameter};
  Fluxes<Dim> fluxes_computer{};

  numerical_flux.compute_dirichlet_boundary(n_dot_num_f_field, n_dot_num_f_aux,
                                            dirichlet_field, fluxes_computer,
                                            fluxes_argument, face_normal);
}

template <size_t Dim>
void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &apply_ip_flux<Dim>, "FirstOrderInternalPenalty",
      {"normal_dot_numerical_flux_for_field",
       "normal_dot_numerical_flux_for_auxiliary_field"},
      {{{-1.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &apply_ip_dirichlet_flux<Dim>, "FirstOrderInternalPenalty",
      {MakeString{} << "normal_dot_dirichlet_flux_for_field_" << Dim << "d",
       MakeString{} << "normal_dot_dirichlet_flux_for_auxiliary_field_" << Dim
                    << "d"},
      {{{-1.0, 1.0}}}, used_for_size);
}

template <size_t Dim>
void test_conservation(const DataVector& used_for_size) {
  elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
      Dim, FluxesComputerTag<Dim>, tmpl::list<ScalarFieldTag>,
      tmpl::list<AuxiliaryFieldTag<Dim>>>
      numerical_flux{1.5};
  EllipticNumericalFluxesTestHelpers::test_conservation<
      Dim, tmpl::list<ScalarFieldTag, AuxiliaryFieldTag<Dim>>>(numerical_flux,
                                                               used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.NumericalFluxes.FirstOrderInternalPenalty",
                  "[Unit][Elliptic][Fluxes]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/DiscontinuousGalerkin/NumericalFluxes"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_equations, (1, 2, 3))
  CHECK_FOR_DATAVECTORS(test_conservation, (1, 2, 3))
}
