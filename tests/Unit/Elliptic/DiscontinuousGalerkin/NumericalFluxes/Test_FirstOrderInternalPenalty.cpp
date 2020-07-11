// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"

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
    const Scalar<DataVector>& face_normal_magnitude_int,
    const Scalar<DataVector>& face_normal_magnitude_ext,
    const tnsr::i<DataVector, Dim>& face_normal_int) noexcept {
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, FluxesComputerTag<Dim>, tmpl::list<ScalarFieldTag>,
          tmpl::list<AuxiliaryFieldTag<Dim>>>;
  const DataVector& used_for_size = *(face_normal_int.begin());

  NumericalFlux numerical_flux{penalty_parameter};
  Fluxes<Dim> fluxes_computer{};

  const Mesh<Dim> volume_mesh_int{3, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> volume_mesh_ext{4, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
  const auto direction = Direction<Dim>::lower_xi();

  auto packaged_data_interior = TestHelpers::NumericalFluxes::get_packaged_data(
      numerical_flux, used_for_size, volume_mesh_int, direction,
      face_normal_magnitude_int, n_dot_aux_flux_int, div_aux_flux_int,
      face_normal_int, fluxes_computer, fluxes_argument);
  tnsr::i<DataVector, Dim> face_normal_ext{face_normal_int};
  for (size_t d = 0; d < Dim; d++) {
    face_normal_ext.get(d) *= -1.;
  }
  auto packaged_data_exterior = TestHelpers::NumericalFluxes::get_packaged_data(
      numerical_flux, used_for_size, volume_mesh_ext, direction.opposite(),
      face_normal_magnitude_ext, n_dot_aux_flux_ext, div_aux_flux_ext,
      face_normal_ext, fluxes_computer, fluxes_argument);

  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      numerical_flux, packaged_data_interior, packaged_data_exterior,
      n_dot_num_f_field, n_dot_num_f_aux);
}

template <size_t Dim>
void apply_ip_dirichlet_flux(
    const gsl::not_null<Scalar<DataVector>*> n_dot_num_f_field,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> n_dot_num_f_aux,
    const Scalar<DataVector>& dirichlet_field, const double fluxes_argument,
    const double penalty_parameter,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::i<DataVector, Dim>& face_normal) noexcept {
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, FluxesComputerTag<Dim>, tmpl::list<ScalarFieldTag>,
          tmpl::list<AuxiliaryFieldTag<Dim>>>;

  NumericalFlux numerical_flux{penalty_parameter};
  Fluxes<Dim> fluxes_computer{};

  const Mesh<Dim> volume_mesh{3, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const auto direction = Direction<Dim>::lower_xi();

  numerical_flux.compute_dirichlet_boundary(
      n_dot_num_f_field, n_dot_num_f_aux, dirichlet_field, volume_mesh,
      direction, face_normal, face_normal_magnitude, fluxes_computer,
      fluxes_argument);
}

template <size_t Dim>
void test_equations(const DataVector& used_for_size) {
  static_assert(tt::assert_conforms_to<
                elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
                    Dim, FluxesComputerTag<Dim>, tmpl::list<ScalarFieldTag>,
                    tmpl::list<AuxiliaryFieldTag<Dim>>>,
                ::dg::protocols::NumericalFlux>);

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

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::uniform_int_distribution<size_t> dist_poly_degree(1, 6);
  const auto gen_nn = make_not_null(&gen);
  const auto dist_nn = make_not_null(&dist);
  const auto dist_poly_degree_nn = make_not_null(&dist_poly_degree);

  EllipticNumericalFluxesTestHelpers::test_conservation<Dim>(
      numerical_flux,
      [&gen_nn, &dist_nn, &dist_poly_degree_nn, &used_for_size]() {
        return std::make_tuple(
            Mesh<Dim>{(*dist_poly_degree_nn)(*gen_nn) + 1,
                      Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto},
            Direction<Dim>::upper_xi(),
            make_with_random_values<Scalar<DataVector>>(gen_nn, dist_nn,
                                                        used_for_size),
            make_with_random_values<tnsr::i<DataVector, Dim>>(gen_nn, dist_nn,
                                                              used_for_size),
            make_with_random_values<tnsr::i<DataVector, Dim>>(gen_nn, dist_nn,
                                                              used_for_size),
            make_with_random_values<tnsr::i<DataVector, Dim>>(gen_nn, dist_nn,
                                                              used_for_size),
            Fluxes<Dim>{}, (*dist_nn)(*gen_nn));
      },
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
