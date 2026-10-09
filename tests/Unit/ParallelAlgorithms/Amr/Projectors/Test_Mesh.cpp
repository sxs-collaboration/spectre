// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "Utilities/Literals.hpp"

namespace {
void test_mesh_1d() {
  const auto topologies = domain::topologies::hypercube<1>;
  const auto legendre = Spectral::Basis::Legendre;
  const auto gauss_lobatto = Spectral::Quadrature::GaussLobatto;
  const auto refine = std::array{amr::Flag::IncreaseResolution};
  const auto coarsen = std::array{amr::Flag::DecreaseResolution};
  const auto split = std::array{amr::Flag::Split};
  const auto join = std::array{amr::Flag::Join};
  const auto stay = std::array{amr::Flag::DoNothing};
  Mesh<1> mesh_3{std::array{3_st}, legendre, gauss_lobatto};
  Mesh<1> mesh_4{std::array{4_st}, legendre, gauss_lobatto};
  CHECK(amr::projectors::p_refined_mesh(mesh_3, refine, topologies) == mesh_4);
  CHECK(amr::projectors::p_refined_mesh(mesh_4, coarsen, topologies) == mesh_3);
  CHECK(amr::projectors::p_refined_mesh(mesh_3, join, topologies) == mesh_3);
  CHECK(amr::projectors::p_refined_mesh(mesh_3, split, topologies) == mesh_3);
  CHECK(amr::projectors::p_refined_mesh(mesh_3, stay, topologies) == mesh_3);

  CHECK(amr::projectors::parent_mesh(std::vector{mesh_3, mesh_4}) == mesh_4);
}

void test_mesh_2d() {
  const auto topologies = domain::topologies::hypercube<2>;
  const auto legendre = Spectral::Basis::Legendre;
  const auto gauss_lobatto = Spectral::Quadrature::GaussLobatto;
  const auto refine_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution};
  const auto refine_stay =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing};
  const auto refine_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution};
  const auto coarsen_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution};
  const auto coarsen_stay =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing};
  const auto coarsen_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution};
  Mesh<2> mesh_3_5{std::array{3_st, 5_st}, legendre, gauss_lobatto};
  Mesh<2> mesh_3_6{std::array{3_st, 6_st}, legendre, gauss_lobatto};
  Mesh<2> mesh_4_5{std::array{4_st, 5_st}, legendre, gauss_lobatto};
  Mesh<2> mesh_4_6{std::array{4_st, 6_st}, legendre, gauss_lobatto};
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5, refine_refine, topologies) ==
        mesh_4_6);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5, refine_stay, topologies) ==
        mesh_4_5);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_6, refine_coarsen, topologies) ==
        mesh_4_5);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_5, coarsen_refine, topologies) ==
        mesh_3_6);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_6, coarsen_stay, topologies) ==
        mesh_3_6);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_6, coarsen_coarsen,
                                        topologies) == mesh_3_5);

  CHECK(amr::projectors::parent_mesh(
            std::vector{mesh_3_5, mesh_4_5, mesh_3_6}) == mesh_4_6);
#ifdef SPECTRE_DEBUG
  const Mesh<2> mesh_mismatch_basis{
      std::array{2_st, 4_st}, std::array{legendre, Spectral::Basis::Chebyshev},
      std::array{gauss_lobatto, gauss_lobatto}};
  CHECK_THROWS_WITH(
      amr::projectors::parent_mesh(std::vector{mesh_3_5, mesh_mismatch_basis}),
      Catch::Matchers::ContainsSubstring(
          "AMR does not currently support joining elements with "
          "different quadratures or bases"));
  const Mesh<2> mesh_mismatch_quadrature{
      std::array{2_st, 4_st}, std::array{legendre, legendre},
      std::array{Spectral::Quadrature::Gauss, gauss_lobatto}};
  CHECK_THROWS_WITH(amr::projectors::parent_mesh(
                        std::vector{mesh_3_5, mesh_mismatch_quadrature}),
                    Catch::Matchers::ContainsSubstring(
                        "AMR does not currently support joining elements with "
                        "different quadratures or bases"));
#endif
}

void test_mesh_3d() {
  const auto topologies = domain::topologies::hypercube<3>;
  const auto legendre = Spectral::Basis::Legendre;
  const auto gauss_lobatto = Spectral::Quadrature::GaussLobatto;
  const auto refine_refine_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto refine_stay_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::IncreaseResolution};
  const auto refine_coarsen_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto coarsen_refine_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto coarsen_stay_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::IncreaseResolution};
  const auto coarsen_coarsen_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto refine_refine_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::DecreaseResolution};
  const auto refine_stay_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::DecreaseResolution};
  const auto refine_coarsen_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::DecreaseResolution};
  const auto coarsen_refine_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::DecreaseResolution};
  const auto coarsen_stay_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::DecreaseResolution};
  const auto coarsen_coarsen_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::DecreaseResolution};
  Mesh<3> mesh_3_5_7{std::array{3_st, 5_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_3_6_7{std::array{3_st, 6_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_5_7{std::array{4_st, 5_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_6_7{std::array{4_st, 6_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_3_5_8{std::array{3_st, 5_st, 8_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_3_6_8{std::array{3_st, 6_st, 8_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_5_8{std::array{4_st, 5_st, 8_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_6_8{std::array{4_st, 6_st, 8_st}, legendre, gauss_lobatto};
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5_7, refine_refine_refine,
                                        topologies) == mesh_4_6_8);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5_7, refine_stay_refine,
                                        topologies) == mesh_4_5_8);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_6_7, refine_coarsen_refine,
                                        topologies) == mesh_4_5_8);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_5_7, coarsen_refine_refine,
                                        topologies) == mesh_3_6_8);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_6_7, coarsen_stay_refine,
                                        topologies) == mesh_3_6_8);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_6_7, coarsen_coarsen_refine,
                                        topologies) == mesh_3_5_8);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5_8, refine_refine_coarsen,
                                        topologies) == mesh_4_6_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5_8, refine_stay_coarsen,
                                        topologies) == mesh_4_5_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_6_8, refine_coarsen_coarsen,
                                        topologies) == mesh_4_5_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_5_8, coarsen_refine_coarsen,
                                        topologies) == mesh_3_6_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_6_8, coarsen_stay_coarsen,
                                        topologies) == mesh_3_6_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_6_8, coarsen_coarsen_coarsen,
                                        topologies) == mesh_3_5_7);

  CHECK(amr::projectors::parent_mesh(
            std::vector{mesh_3_5_7, mesh_4_5_8, mesh_3_6_7}) == mesh_4_6_8);
#ifdef SPECTRE_DEBUG
  const Mesh<3> mesh_mismatch_basis{
      std::array{2_st, 4_st, 3_st},
      std::array{legendre, Spectral::Basis::Chebyshev, legendre},
      std::array{gauss_lobatto, gauss_lobatto, gauss_lobatto}};
  CHECK_THROWS_WITH(amr::projectors::parent_mesh(
                        std::vector{mesh_3_5_7, mesh_mismatch_basis}),
                    Catch::Matchers::ContainsSubstring(
                        "AMR does not currently support joining elements with "
                        "different quadratures or bases"));
  const Mesh<3> mesh_mismatch_quadrature{
      std::array{2_st, 4_st, 3_st}, std::array{legendre, legendre, legendre},
      std::array{Spectral::Quadrature::Gauss, gauss_lobatto, gauss_lobatto}};
  CHECK_THROWS_WITH(amr::projectors::parent_mesh(
                        std::vector{mesh_3_5_7, mesh_mismatch_quadrature}),
                    Catch::Matchers::ContainsSubstring(
                        "AMR does not currently support joining elements with "
                        "different quadratures or bases"));
#endif
}

template <size_t Dim>
void test_equiangular_mesh() {
  const auto topologies = domain::topologies::hypertorus<Dim>;
  const auto refine = make_array<Dim>(amr::Flag::IncreaseResolution);
  const auto coarsen = make_array<Dim>(amr::Flag::DecreaseResolution);
  const auto stay = make_array<Dim>(amr::Flag::DoNothing);
  Mesh<Dim> mesh_3{make_array<Dim>(3_st), Spectral::bases::hypertorus<Dim>,
                   Spectral::quadratures::hypertorus<Dim>};
  Mesh<Dim> mesh_5{make_array<Dim>(5_st), Spectral::bases::hypertorus<Dim>,
                   Spectral::quadratures::hypertorus<Dim>};
  Mesh<Dim> mesh_7{make_array<Dim>(7_st), Spectral::bases::hypertorus<Dim>,
                   Spectral::quadratures::hypertorus<Dim>};
  CHECK(amr::projectors::p_refined_mesh(mesh_5, refine, topologies) == mesh_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_5, coarsen, topologies) == mesh_3);
  CHECK(amr::projectors::p_refined_mesh(mesh_5, stay, topologies) == mesh_5);
}

void test_hypercube() {
  test_mesh_1d();
  test_mesh_2d();
  test_mesh_3d();
}

void test_hypertorus() {
  test_equiangular_mesh<1>();
  test_equiangular_mesh<2>();
  test_equiangular_mesh<3>();
}

void test_annulus() {
  const auto topologies = domain::topologies::annulus;
  const auto bases = Spectral::bases::annulus<>;
  const auto quadratures = Spectral::quadratures::annulus<>;
  const auto refine_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution};
  const auto refine_stay =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing};
  const auto refine_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution};
  const auto coarsen_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution};
  const auto coarsen_stay =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing};
  const auto coarsen_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution};
  Mesh<2> mesh_3_5{std::array{3_st, 5_st}, bases, quadratures};
  Mesh<2> mesh_3_7{std::array{3_st, 7_st}, bases, quadratures};
  Mesh<2> mesh_4_5{std::array{4_st, 5_st}, bases, quadratures};
  Mesh<2> mesh_4_7{std::array{4_st, 7_st}, bases, quadratures};
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5, refine_refine, topologies) ==
        mesh_4_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_5, refine_stay, topologies) ==
        mesh_4_5);
  CHECK(amr::projectors::p_refined_mesh(mesh_3_7, refine_coarsen, topologies) ==
        mesh_4_5);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_5, coarsen_refine, topologies) ==
        mesh_3_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_7, coarsen_stay, topologies) ==
        mesh_3_7);
  CHECK(amr::projectors::p_refined_mesh(mesh_4_7, coarsen_coarsen,
                                        topologies) == mesh_3_5);

  CHECK(amr::projectors::parent_mesh(
            std::vector{mesh_3_5, mesh_4_5, mesh_3_7}) == mesh_4_7);
#ifdef SPECTRE_DEBUG
  const Mesh<2> mesh_mismatch_basis{
      std::array{2_st, 5_st},
      Spectral::bases::annulus<Spectral::Basis::Chebyshev>, quadratures};
  CHECK_THROWS_WITH(
      amr::projectors::parent_mesh(std::vector{mesh_3_5, mesh_mismatch_basis}),
      Catch::Matchers::ContainsSubstring(
          "AMR does not currently support joining elements with "
          "different quadratures or bases"));
  const Mesh<2> mesh_mismatch_quadrature{
      std::array{2_st, 5_st}, bases,
      Spectral::quadratures::annulus<Spectral::Quadrature::Gauss>};
  CHECK_THROWS_WITH(amr::projectors::parent_mesh(
                        std::vector{mesh_3_5, mesh_mismatch_quadrature}),
                    Catch::Matchers::ContainsSubstring(
                        "AMR does not currently support joining elements with "
                        "different quadratures or bases"));
#endif
  const ElementId<2> lower_child{0,
                                 std::array{SegmentId{1, 0}, SegmentId{0, 0}}};
  const ElementId<2> upper_child{0,
                                 std::array{SegmentId{1, 1}, SegmentId{0, 0}}};
  CHECK(amr::projectors::child_mesh(
            mesh_3_5, lower_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing},
            topologies) == mesh_3_5);
  CHECK(amr::projectors::child_mesh(
            mesh_3_5, upper_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing},
            topologies) == mesh_3_5);
}

void test_spherical_surface() {
  const auto topologies = domain::topologies::spherical_surface;
  const Mesh<2> mesh_0{std::array{5_st, 9_st},
                       Spectral::bases::spherical_surface,
                       Spectral::quadratures::spherical_surface};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<2>(amr::Flag::DoNothing), topologies) == mesh_0);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, make_array<2>(amr::Flag::Join),
                                        topologies) == mesh_0);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, make_array<2>(amr::Flag::Split),
                                        topologies) == mesh_0);
  const Mesh<2> mesh_r{std::array{6_st, 11_st},
                       Spectral::bases::spherical_surface,
                       Spectral::quadratures::spherical_surface};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<2>(amr::Flag::IncreaseResolution), topologies) ==
        mesh_r);
  const Mesh<2> mesh_c{std::array{4_st, 7_st},
                       Spectral::bases::spherical_surface,
                       Spectral::quadratures::spherical_surface};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<2>(amr::Flag::DecreaseResolution), topologies) ==
        mesh_c);
}

void test_disk() {
  const auto topologies = domain::topologies::disk;
  const Mesh<2> mesh_0{std::array{3_st, 9_st}, Spectral::bases::disk,
                       Spectral::quadratures::disk};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<2>(amr::Flag::DoNothing), topologies) == mesh_0);
  const Mesh<2> mesh_r{std::array{4_st, 13_st}, Spectral::bases::disk,
                       Spectral::quadratures::disk};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<2>(amr::Flag::IncreaseResolution), topologies) ==
        mesh_r);
  const Mesh<2> mesh_c{std::array{2_st, 5_st}, Spectral::bases::disk,
                       Spectral::quadratures::disk};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<2>(amr::Flag::DecreaseResolution), topologies) ==
        mesh_c);
  const Mesh<2> mesh_a{std::array{4_st, 5_st}, Spectral::bases::annulus<>,
                       Spectral::quadratures::annulus<>};
  CHECK(amr::projectors::parent_mesh(std::vector{mesh_0, mesh_a}) == mesh_r);
  const Mesh<2> mesh_uc{std::array{3_st, 9_st}, Spectral::bases::annulus<>,
                        Spectral::quadratures::annulus<>};
  const ElementId<2> lower_child{0,
                                 std::array{SegmentId{1, 0}, SegmentId{0, 0}}};
  const ElementId<2> upper_child{0,
                                 std::array{SegmentId{1, 1}, SegmentId{0, 0}}};

  CHECK(amr::projectors::child_mesh(
            mesh_0, lower_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing},
            topologies) == mesh_0);
  CHECK(amr::projectors::child_mesh(
            mesh_0, upper_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing},
            topologies) == mesh_uc);
}

void test_full_cylinder() {
  const auto topologies = domain::topologies::full_cylinder;
  const Mesh<3> mesh_0{std::array{3_st, 9_st, 5_st},
                       Spectral::bases::full_cylinder<>,
                       Spectral::quadratures::full_cylinder<>};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<3>(amr::Flag::DoNothing), topologies) == mesh_0);
  const Mesh<3> mesh_r{std::array{4_st, 13_st, 6_st},
                       Spectral::bases::full_cylinder<>,
                       Spectral::quadratures::full_cylinder<>};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<3>(amr::Flag::IncreaseResolution), topologies) ==
        mesh_r);
  const Mesh<3> mesh_c{std::array{2_st, 5_st, 4_st},
                       Spectral::bases::full_cylinder<>,
                       Spectral::quadratures::full_cylinder<>};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<3>(amr::Flag::DecreaseResolution), topologies) ==
        mesh_c);
  const Mesh<3> mesh_cs{std::array{4_st, 5_st, 6_st},
                        Spectral::bases::cylindrical_shell<>,
                        Spectral::quadratures::cylindrical_shell<>};
  CHECK(amr::projectors::parent_mesh(std::vector{mesh_0, mesh_cs}) == mesh_r);
  const Mesh<3> mesh_uc{std::array{3_st, 9_st, 6_st},
                        Spectral::bases::cylindrical_shell<>,
                        Spectral::quadratures::cylindrical_shell<>};
  const ElementId<3> lower_child{
      0, std::array{SegmentId{1, 0}, SegmentId{0, 0}, SegmentId{0, 0}}};
  const ElementId<3> upper_child{
      0, std::array{SegmentId{1, 1}, SegmentId{0, 0}, SegmentId{0, 0}}};

  CHECK(amr::projectors::child_mesh(
            mesh_0, lower_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing,
                       amr::Flag::DoNothing},
            topologies) == mesh_0);
  CHECK(amr::projectors::child_mesh(
            mesh_0, upper_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing,
                       amr::Flag::IncreaseResolution},
            topologies) == mesh_uc);
}

void test_full_sphere() {
  const auto topologies = domain::topologies::full_sphere;
  const Mesh<3> mesh_0{std::array{3_st, 5_st, 9_st},
                       Spectral::bases::full_sphere,
                       Spectral::quadratures::full_sphere};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<3>(amr::Flag::DoNothing), topologies) == mesh_0);
  const Mesh<3> mesh_r{std::array{4_st, 7_st, 13_st},
                       Spectral::bases::full_sphere,
                       Spectral::quadratures::full_sphere};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<3>(amr::Flag::IncreaseResolution), topologies) ==
        mesh_r);
  const Mesh<3> mesh_c{std::array{2_st, 3_st, 5_st},
                       Spectral::bases::full_sphere,
                       Spectral::quadratures::full_sphere};
  CHECK(amr::projectors::p_refined_mesh(
            mesh_0, make_array<3>(amr::Flag::DecreaseResolution), topologies) ==
        mesh_c);
  const Mesh<3> mesh_cs{std::array{3_st, 7_st, 13_st},
                        Spectral::bases::spherical_shell<>,
                        Spectral::quadratures::spherical_shell<>};
  CHECK(amr::projectors::parent_mesh(std::vector{mesh_0, mesh_cs}) == mesh_r);
  const Mesh<3> mesh_uc{std::array{3_st, 5_st, 9_st},
                        Spectral::bases::spherical_shell<>,
                        Spectral::quadratures::spherical_shell<>};
  const ElementId<3> lower_child{
      0, std::array{SegmentId{1, 0}, SegmentId{0, 0}, SegmentId{0, 0}}};
  const ElementId<3> upper_child{
      0, std::array{SegmentId{1, 1}, SegmentId{0, 0}, SegmentId{0, 0}}};

  CHECK(amr::projectors::child_mesh(
            mesh_0, lower_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing,
                       amr::Flag::DoNothing},
            topologies) == mesh_0);
  CHECK(amr::projectors::child_mesh(
            mesh_0, upper_child,
            std::array{amr::Flag::Split, amr::Flag::DoNothing,
                       amr::Flag::DoNothing},
            topologies) == mesh_uc);
}

void test_cartoon_sphere() {
  const auto topologies = domain::topologies::cartoon_sphere;
  const auto refine = std::array{amr::Flag::IncreaseResolution,
                                 amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto coarsen = std::array{amr::Flag::DecreaseResolution,
                                  amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto stay = std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                               amr::Flag::DoNothing};
  const auto split =
      std::array{amr::Flag::Split, amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto join =
      std::array{amr::Flag::Join, amr::Flag::DoNothing, amr::Flag::DoNothing};
  const Mesh<3> mesh_0{std::array{3_st, 1_st, 1_st},
                       Spectral::bases::cartoon_sphere<>,
                       Spectral::quadratures::cartoon_sphere<>};
  const Mesh<3> mesh_c{std::array{2_st, 1_st, 1_st},
                       Spectral::bases::cartoon_sphere<>,
                       Spectral::quadratures::cartoon_sphere<>};
  const Mesh<3> mesh_r{std::array{4_st, 1_st, 1_st},
                       Spectral::bases::cartoon_sphere<>,
                       Spectral::quadratures::cartoon_sphere<>};
  CHECK(amr::projectors::p_refined_mesh(mesh_0, refine, topologies) == mesh_r);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, coarsen, topologies) == mesh_c);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, stay, topologies) == mesh_0);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, split, topologies) == mesh_0);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, join, topologies) == mesh_0);
  CHECK(amr::projectors::parent_mesh(std::vector{mesh_0, mesh_r}) == mesh_r);
  const ElementId<3> lower_child{
      0, std::array{SegmentId{1, 0}, SegmentId{0, 0}, SegmentId{0, 0}}};
  const ElementId<3> upper_child{
      0, std::array{SegmentId{1, 1}, SegmentId{0, 0}, SegmentId{0, 0}}};
  CHECK(amr::projectors::child_mesh(mesh_0, lower_child, split, topologies) ==
        mesh_0);
  CHECK(amr::projectors::child_mesh(mesh_0, upper_child, split, topologies) ==
        mesh_0);
}

void test_cartoon_sphere_inner() {
  const auto topologies = domain::topologies::cartoon_sphere_inner;
  const auto refine = std::array{amr::Flag::IncreaseResolution,
                                 amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto coarsen = std::array{amr::Flag::DecreaseResolution,
                                  amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto stay = std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                               amr::Flag::DoNothing};
  const auto split =
      std::array{amr::Flag::Split, amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto join =
      std::array{amr::Flag::Join, amr::Flag::DoNothing, amr::Flag::DoNothing};
  const Mesh<3> mesh_0{std::array{2_st, 1_st, 1_st},
                       Spectral::bases::cartoon_sphere_inner,
                       Spectral::quadratures::cartoon_sphere_inner};
  const Mesh<3> mesh_c{std::array{1_st, 1_st, 1_st},
                       Spectral::bases::cartoon_sphere_inner,
                       Spectral::quadratures::cartoon_sphere_inner};
  const Mesh<3> mesh_r{std::array{3_st, 1_st, 1_st},
                       Spectral::bases::cartoon_sphere_inner,
                       Spectral::quadratures::cartoon_sphere_inner};
  CHECK(amr::projectors::p_refined_mesh(mesh_0, refine, topologies) == mesh_r);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, coarsen, topologies) == mesh_c);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, stay, topologies) == mesh_0);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, split, topologies) == mesh_0);
  CHECK(amr::projectors::p_refined_mesh(mesh_0, join, topologies) == mesh_0);
  CHECK(amr::projectors::parent_mesh(std::vector{mesh_0, mesh_r}) == mesh_r);
  const ElementId<3> lower_child{
      0, std::array{SegmentId{1, 0}, SegmentId{0, 0}, SegmentId{0, 0}}};
  const ElementId<3> upper_child{
      0, std::array{SegmentId{1, 1}, SegmentId{0, 0}, SegmentId{0, 0}}};
  CHECK(amr::projectors::child_mesh(mesh_0, lower_child, split, topologies) ==
        mesh_0);
  const Mesh<3> mesh_uc{std::array{2_st, 1_st, 1_st},
                        Spectral::bases::cartoon_sphere<>,
                        Spectral::quadratures::cartoon_sphere<>};
  CHECK(amr::projectors::child_mesh(mesh_0, upper_child, split, topologies) ==
        mesh_uc);
  CHECK(amr::projectors::child_mesh(mesh_c, upper_child, split, topologies) ==
        mesh_uc);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.Mesh", "[ParallelAlgorithms][Unit]") {
  test_hypercube();
  test_hypertorus();
  test_annulus();
  test_spherical_surface();
  test_disk();
  test_full_cylinder();
  test_full_sphere();
  test_cartoon_sphere();
  test_cartoon_sphere_inner();
}
