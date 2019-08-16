// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarFieldTag"; }
};

template <size_t Dim, typename... CoordMaps>
void test_mass(const Mesh<Dim>& mesh,
               const domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                           CoordMaps...>& coordinate_map,
               const DataVector& scalar_field,
               const DataVector& expected_massive_scalar_field) noexcept {
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto jacobian = coordinate_map.jacobian(logical_coords);

  Variables<tmpl::list<ScalarFieldTag>> vars{num_grid_points};
  get<ScalarFieldTag>(vars) = Scalar<DataVector>(scalar_field);
  const auto massive_vars = mass(vars, mesh, jacobian);
  CHECK_ITERABLE_APPROX(get(get<::Tags::Mass<ScalarFieldTag>>(massive_vars)),
                        expected_massive_scalar_field);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Mass",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  {
    INFO("1D");
    const Mesh<1> mesh{
        {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            Affine{-1.0, 1.0, -0.3, 0.7});
    test_mass(mesh, coord_map, {1., 2., 3., 4.},
              {0.0924277621726216, 0.8129975722321499, 1.2703357611011825,
               0.3242389044940451});
  }
  {
    INFO("2D");
    const Mesh<2> mesh{{{4, 2}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine2D{
            Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
    test_mass(mesh, coord_map, {1., 2., 3., 4., 5., 6., 7., 8.},
              {0.0254423591604666, 0.1710691409734632, 0.2282364145820922,
               0.0544187519506445, 0.0393312480493555, 0.2405135854179076,
               0.2976808590265367, 0.0683076408395334});
  }
  {
    INFO("3D");
    const Mesh<3> mesh{{{4, 2, 3}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                     Affine{-1.0, 1.0, 2.3, 2.8}});
    test_mass(
        mesh, coord_map,
        {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
         13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.},
        {0.0021201965967055456, 0.014255761747788576,  0.019019701215174337,
         0.004534895995887041,  0.0032776040041129547, 0.020042798784825604,
         0.024806738252211368,  0.005692303403294447,  0.03625856416459998,
         0.19591193588004313,   0.2149676937495862,    0.04591736176132595,
         0.040888193794229616,  0.21906008402819127,   0.23811584189773435,
         0.05054699139095557,   0.016009085485594442,  0.083700206192233,
         0.08846414565961877,   0.018423784884775935,  0.01716649289300185,
         0.08948724322927003,   0.0942511826966558,    0.01958119229218334});
  }
}

namespace {

template <size_t VolumeDim, typename... CoordMaps>
void test_mass_on_face(
    const Mesh<VolumeDim>& mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, CoordMaps...>&
        coordinate_map,
    const Direction<VolumeDim>& face_direction,
    const DataVector& scalar_field_on_face,
    const DataVector& expected_massive_scalar_field_on_face) noexcept {
  const size_t face_dimension = face_direction.dimension();
  const auto face_mesh = mesh.slice_away(face_dimension);
  const auto face_logical_coords =
      interface_logical_coordinates(face_mesh, face_direction);
  const auto jacobian_on_face = coordinate_map.jacobian(face_logical_coords);

  Variables<tmpl::list<ScalarFieldTag>> vars{face_mesh.number_of_grid_points()};
  get<ScalarFieldTag>(vars) = Scalar<DataVector>(scalar_field_on_face);
  const auto massive_vars =
      mass_on_face(vars, face_mesh, face_dimension, jacobian_on_face);
  CHECK_ITERABLE_APPROX(get(get<::Tags::Mass<ScalarFieldTag>>(massive_vars)),
                        expected_massive_scalar_field_on_face);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MassMatrixOnFace",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  {
    INFO("1D");
    const Mesh<1> mesh{
        {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            Affine{-1.0, 1.0, -0.3, 0.7});
    test_mass_on_face(mesh, coord_map, Direction<1>::lower_xi(), {1.}, {1.});
  }
  {
    INFO("2D");
    const Mesh<2> mesh{{{4, 2}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine2D{
            Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
    // Computed with LGL quadrature, check these values
    test_mass_on_face(mesh, coord_map, Direction<2>::lower_xi(), {1., 5.},
                      {0.125, 0.625});
    test_mass_on_face(mesh, coord_map, Direction<2>::upper_xi(), {4., 8.},
                      {0.5, 1.});
    test_mass_on_face(
        mesh, coord_map, Direction<2>::lower_eta(), {1., 2., 3., 4.},
        {0.0833333333333333, 0.8333333333333334, 1.25, 0.3333333333333333});
    test_mass_on_face(
        mesh, coord_map, Direction<2>::upper_eta(), {5., 6., 7., 8.},
        {0.4166666666666666, 2.5, 2.916666666666667, 0.6666666666666666});
  }
  {
    INFO("3D");
    const Mesh<3> mesh{{{4, 10, 10}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                     Affine{-1.0, 1.0, 2.3, 2.8}});
    // Computed with LGL quadrature, check these values
    test_mass_on_face(
        mesh, coord_map, Direction<3>::lower_xi(),
        {1.0, 5.0, 9.0, 13.0, 17.0, 21.0},
        {0.010416666666666671, 0.05208333333333335, 0.3750000000000001,
         0.5416666666666669, 0.1770833333333334, 0.2187500000000001});

    // test_mass_on_face(
    //     mesh, coord_map, Direction<3>::lower_xi(),
    //     {1.0,   5.0,   9.0,   13.0,  17.0,  21.0,  25.0,  29.0,  33.0,  37.0,
    //      41.0,  45.0,  49.0,  53.0,  57.0,  61.0,  65.0,  69.0,  73.0,  77.0,
    //      81.0,  85.0,  89.0,  93.0,  97.0,  101.0, 105.0, 109.0, 113.0,
    //      117.0, 121.0, 125.0, 129.0, 133.0, 137.0, 141.0, 145.0, 149.0,
    //      153.0, 157.0, 161.0, 165.0, 169.0, 173.0, 177.0, 181.0, 185.0,
    //      189.0, 193.0, 197.0, 201.0, 205.0, 209.0, 213.0, 217.0, 221.0,
    //      225.0, 229.0, 233.0, 237.0, 241.0, 245.0, 249.0, 253.0, 257.0,
    //      261.0, 265.0, 269.0, 273.0, 277.0, 281.0, 285.0, 289.0, 293.0,
    //      297.0, 301.0, 305.0, 309.0, 313.0, 317.0, 321.0, 325.0, 329.0,
    //      333.0, 337.0, 341.0, 345.0, 349.0, 353.0, 357.0, 361.0, 365.0,
    //      369.0, 373.0, 377.0, 381.0, 385.0, 389.0, 393.0, 397.0},
    //     {1.5432098765432106e-05, 0.00046286802378843424,
    //     0.001405558387894538,
    //      0.002636496449885992,   0.0038667888473099083, 0.004776621517265181,
    //      0.005070185480550003,   0.004529021472104623, 0.003054928957003712,
    //      0.0005709876543209879,  0.0037955177950651613,  0.02498974762047933,
    //      0.04590549163134158,    0.06447953387163818,    0.07777474085129053,
    //      0.08323261740225829,    0.07907867361616032,    0.06464242699107284,
    //      0.04053892391766708,    0.007128167566341887, 0.012650025491050843,
    //      0.07963197527885786,    0.14066231998278528,    0.19087461530617994,
    //      0.2232824854699858,     0.23249001064400582,    0.2155035979263329,
    //      0.17227183009127633,    0.10586368478248322, 0.018272259042628997,
    //      0.02453969772586193,    0.15207437233876933,    0.2647615631666367,
    //      0.3544821115337584,     0.4095254984033971,     0.42148244726189044,
    //      0.3864654599428206,     0.30580986753355704,    0.18613903174265642,
    //      0.0318407648178539,     0.03662076496569973,    0.22513740772742,
    //      0.3890179386023464,     0.5171380381298373,     0.593405195083865,
    //      0.6068154819784157,     0.5530088847053192,     0.43505556447244653,
    //      0.26334254358419823,    0.044809258995297176,   0.04571909166525245,
    //      0.2797161732370975,     0.4810931903425467,     0.6367075267147707,
    //      0.7275080640293712,     0.7409183509239219,     0.672578373290253,
    //      0.5271308162126469,     0.3179213090938766,     0.0539075856948499,
    //      0.04887658803250203,    0.2980657697839889,     0.5110513893681609,
    //      0.6743155956243698,     0.7682339641582,        0.7801909130166934,
    //      0.7062989440334332,     0.5520996937350814,     0.3321304291878783,
    //      0.056177655124494026,   0.04388465633315169,    0.26700132887617045,
    //      0.45675742106769596,    0.6013576589753841,     0.6836587441709874,
    //      0.6928662693450074,     0.6259866415955384,     0.488366931176187,
    //      0.2932330383797986,     0.049506889884729836,   0.02971612712721793,
    //      0.18048151059235346,    0.3082225866675839,     0.4051261279104876,
    //      0.4598260994190404,     0.46528397597000826,    0.41972526765501106,
    //      0.32695952202731543,    0.19603068688954373, 0.033048776898494704,
    //      0.00557098765432099,    0.0337893657365557, 0.057627893903676064,
    //      0.07564716736980578,    0.08575172914328444,    0.08666156181323971,
    //      0.07808085640047004,    0.06075135698788615,    0.03638142666977148,
    //      0.006126543209876546});
  }
}
