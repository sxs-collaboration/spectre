// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DiscontinuousGalerkin/InterpolateFromBoundary.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
void test(const double eps) {
  using dt_variables_tags = tmpl::list<Tags::dt<Var1>, Tags::dt<Var2<Dim>>>;
  Mesh<Dim> volume_mesh{8, Spectral::Basis::Legendre,
                        Spectral::Quadrature::Gauss};
  Mesh<Dim> volume_mesh_gl{8, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  CAPTURE(Dim);
  CAPTURE(volume_mesh);
  const double correction_value = 2.0e-3;

  for (const auto& direction : Direction<Dim>::all_directions()) {
    CAPTURE(direction);
    Variables<dt_variables_tags> volume_dt{volume_mesh.number_of_grid_points(),
                                           3.0};

    Variables<dt_variables_tags> volume_dt_correction_gl{
        volume_mesh_gl.number_of_grid_points(), 0.0};
    for (SliceIterator si(volume_mesh.extents(), direction.dimension(),
                          direction.side() == Side::Upper
                              ? volume_mesh.extents(direction.dimension()) - 1
                              : 0);
         si; ++si) {
      get(get<Tags::dt<Var1>>(volume_dt_correction_gl))[si.volume_offset()] =
          correction_value;
      for (size_t i = 0; i < Dim; ++i) {
        get<Tags::dt<Var2<Dim>>>(volume_dt_correction_gl)
            .get(i)[si.volume_offset()] = correction_value * (i + 2.0);
      }
    }
    const Variables<dt_variables_tags> volume_dt_correction_gauss =
        intrp::RegularGrid<Dim>{volume_mesh_gl, volume_mesh}.interpolate(
            volume_dt_correction_gl);

    const Variables<dt_variables_tags> expected_volume_dt_gauss =
        volume_dt + volume_dt_correction_gauss;

    Variables<dt_variables_tags> dt_correction_on_boundary{
        volume_mesh.slice_away(direction.dimension()).number_of_grid_points(),
        correction_value};
    for (size_t i = 0; i < Dim; ++i) {
      get<Tags::dt<Var2<Dim>>>(dt_correction_on_boundary).get(i) *= (2.0 + i);
    }
    evolution::dg::interpolate_dt_terms_gauss_points(make_not_null(&volume_dt),
                                                     volume_mesh, direction,
                                                     dt_correction_on_boundary);

    Approx local_approx = Approx::custom().epsilon(eps).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::dt<Var1>>(volume_dt),
                                 get<Tags::dt<Var1>>(expected_volume_dt_gauss),
                                 local_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<Tags::dt<Var2<Dim>>>(volume_dt),
        get<Tags::dt<Var2<Dim>>>(expected_volume_dt_gauss), local_approx);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.InterpolateDtCorrection",
                  "[Unit][Evolution]") {
  test<1>(1.0e-14);
  test<2>(1.0e-14);
  test<3>(1.0e-14);
}
}  // namespace
