// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz {

template <size_t Dim>
void test_data_on_overlap_consistency(const Index<Dim>& volume_extents,
                                      const size_t overlap_extent,
                                      const Direction<Dim>& direction) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size{volume_extents.product()};
  using tags_list = tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, Dim>,
                               ::Tags::Tempijj<2, Dim>>;
  const auto vars = make_with_random_values<Variables<tags_list>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  CAPTURE(vars);

  const Variables<tags_list> vars_on_overlap =
      data_on_overlap(vars, volume_extents, overlap_extent, direction);
  CAPTURE(vars_on_overlap);

  {
    INFO("Test Variables and Tensor data on overlap is consistent");
    tmpl::for_each<tags_list>([&](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CAPTURE(db::tag_name<tag>());
      const auto tensor_on_overlap = data_on_overlap(
          get<tag>(vars), volume_extents, overlap_extent, direction);
      for (size_t i = 0; i < tensor_on_overlap.size(); i++) {
        CAPTURE(i);
        CHECK_ITERABLE_APPROX(tensor_on_overlap[i],
                              get<tag>(vars_on_overlap)[i]);
      }
    });
  }
  {
    INFO("Test extended_overlap_data and add_overlap_data are consistent");
    Variables<tags_list> extended_vars{used_for_size.size(), 0.};
    add_overlap_data(make_not_null(&extended_vars), vars_on_overlap,
                     volume_extents, overlap_extent, direction);
    CHECK_VARIABLES_APPROX(
        extended_vars, extended_overlap_data(vars_on_overlap, volume_extents,
                                             overlap_extent, direction));
  }
}

template <size_t Dim>
void test_data_on_overlap(const DataVector& scalar_volume_data,
                          const Index<Dim>& volume_extents,
                          const size_t overlap_extent,
                          const Direction<Dim>& direction,
                          const DataVector& expected_overlap_data,
                          const DataVector& expected_extended_data) {
  CAPTURE(volume_extents);
  CAPTURE(overlap_extent);
  CAPTURE(direction);
  {
    INFO("Tensor data on overlap");
    const Scalar<DataVector> scalar{scalar_volume_data};
    CHECK_ITERABLE_APPROX(
        get(data_on_overlap(scalar, volume_extents, overlap_extent, direction)),
        expected_overlap_data);
  }
  Variables<tmpl::list<::Tags::TempScalar<0>>> vars{scalar_volume_data.size()};
  get(get<::Tags::TempScalar<0>>(vars)) = scalar_volume_data;
  const auto vars_on_overlap =
      data_on_overlap(vars, volume_extents, overlap_extent, direction);
  {
    INFO("Variables data on overlap");
    CHECK_ITERABLE_APPROX(get(get<::Tags::TempScalar<0>>(vars_on_overlap)),
                          expected_overlap_data);
  }
  Variables<tmpl::list<::Tags::TempScalar<0>>> extended_vars{
      scalar_volume_data.size(), 0.};
  add_overlap_data(make_not_null(&extended_vars), vars_on_overlap,
                   volume_extents, overlap_extent, direction);
  {
    INFO("Add overlap data");
    CHECK_ITERABLE_APPROX(get(get<::Tags::TempScalar<0>>(extended_vars)),
                          expected_extended_data);
  }
  {
    INFO("Extended overlap data");
    CHECK_VARIABLES_APPROX(
        extended_overlap_data(vars_on_overlap, volume_extents, overlap_extent,
                              direction),
        extended_vars);
  }
  {
    INFO("Test consistency with non-scalars");
    test_data_on_overlap_consistency(volume_extents, overlap_extent, direction);
  }
}

template <size_t Dim>
void test_overlap_iterator(const Index<Dim>& volume_extents,
                           const size_t overlap_extent,
                           const Direction<Dim>& direction,
                           const std::vector<size_t>& expected_volume_offsets) {
  CAPTURE(Dim);
  CAPTURE(volume_extents);
  CAPTURE(overlap_extent);
  CAPTURE(direction);
  ASSERT(std::is_sorted(expected_volume_offsets.begin(),
                        expected_volume_offsets.end()),
         "Volume indices must be sorted to optimize array access performance.");

  OverlapIterator overlap_iterator{volume_extents, overlap_extent, direction};
  // Reset the iterator once to test resetting works
  ++overlap_iterator;
  overlap_iterator.reset();
  for (size_t i = 0; i < expected_volume_offsets.size(); ++i) {
    CHECK(overlap_iterator);
    CHECK(overlap_iterator.volume_offset() == expected_volume_offsets[i]);
    CHECK(overlap_iterator.overlap_offset() == i);
    ++overlap_iterator;
  }
  CHECK_FALSE(overlap_iterator);
}

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.OverlapHelpers",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  {
    INFO("Overlap extents");
    CHECK(overlap_extent(3, 0) == 0);
    CHECK(overlap_extent(3, 1) == 1);
    CHECK(overlap_extent(3, 2) == 2);
    CHECK(overlap_extent(3, 3) == 2);
    CHECK(overlap_extent(3, 4) == 2);
    CHECK(overlap_extent(0, 0) == 0);
  }
  {
    INFO("Overlap num_points");
    CHECK(overlap_num_points(Index<1>{{{3}}}, 0, 0) == 0);
    CHECK(overlap_num_points(Index<1>{{{3}}}, 1, 0) == 1);
    CHECK(overlap_num_points(Index<1>{{{3}}}, 2, 0) == 2);
    CHECK(overlap_num_points(Index<1>{{{3}}}, 3, 0) == 3);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 0, 0) == 0);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 1, 0) == 3);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 2, 0) == 6);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 0, 1) == 0);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 1, 1) == 2);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 2, 1) == 4);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 3, 1) == 6);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 0, 0) == 0);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 1, 0) == 12);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 2, 0) == 24);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 0, 1) == 0);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 1, 1) == 8);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 2, 1) == 16);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 3, 1) == 24);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 0, 2) == 0);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 1, 2) == 6);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 2, 2) == 12);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 3, 2) == 18);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 4, 2) == 24);
  }
  {
    INFO("Overlap width");
    DataVector coords{-1., -0.8, 0., 0.8, 1.};
    CHECK(overlap_width(0, coords) == approx(0.));
    CHECK(overlap_width(1, coords) == approx(0.2));
    CHECK(overlap_width(2, coords) == approx(1.));
    CHECK(overlap_width(3, coords) == approx(1.8));
    CHECK(overlap_width(4, coords) == approx(2.));
  }
  {
    INFO("Overlap iterator");
    {
      // Give an example to include in the docs
      // [overlap_iterator]
      // Overlap region:
      // + - - - + -xi->
      // | X X O |
      // | X X O |
      // + - - - +
      // v eta
      const Index<2> volume_extents{{{3, 2}}};
      const size_t overlap_extent = 2;
      const auto direction = Direction<2>::lower_xi();
      const std::array<size_t, 4> expected_volume_offsets{{0, 1, 3, 4}};
      size_t i = 0;
      for (OverlapIterator overlap_iterator{volume_extents, overlap_extent,
                                            direction};
           overlap_iterator; ++overlap_iterator) {
        CHECK(overlap_iterator.volume_offset() ==
              gsl::at(expected_volume_offsets, i));
        CHECK(overlap_iterator.overlap_offset() == i);
        ++i;
      }
      CHECK(i == expected_volume_offsets.size());
      // [overlap_iterator]
    }
    // Overlap region: [X X O] -xi->
    test_overlap_iterator(Index<1>{3}, 2, Direction<1>::lower_xi(), {{0, 1}});
    // Overlap region: [O X X] -xi->
    test_overlap_iterator(Index<1>{3}, 2, Direction<1>::upper_xi(), {{1, 2}});
    // Overlap region:
    // + - - - + -xi->
    // | X X O |
    // | X X O |
    // + - - - +
    // v eta
    test_overlap_iterator(Index<2>{{{3, 2}}}, 2, Direction<2>::lower_xi(),
                          {{0, 1, 3, 4}});
    // Overlap region:
    // + - - - + -xi->
    // | O X X |
    // | O X X |
    // + - - - +
    // v eta
    test_overlap_iterator(Index<2>{{{3, 2}}}, 2, Direction<2>::upper_xi(),
                          {{1, 2, 4, 5}});
    // Overlap region:
    // + - - - - + -xi->
    // | X X X X |
    // | X X X X |
    // | O O O O |
    // + - - - - +
    // v eta
    test_overlap_iterator(Index<2>{{{4, 3}}}, 2, Direction<2>::lower_eta(),
                          {{0, 1, 2, 3, 4, 5, 6, 7}});
    // Overlap region:
    // + - - + -xi->
    // | O O |
    // | X X |
    // | X X |
    // + - - +
    // v eta
    test_overlap_iterator(Index<2>{{{2, 3}}}, 2, Direction<2>::upper_eta(),
                          {{2, 3, 4, 5}});
    test_overlap_iterator(Index<3>{{{3, 3, 2}}}, 2, Direction<3>::lower_xi(),
                          {{0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16}});
    test_overlap_iterator(Index<3>{{{3, 3, 2}}}, 2, Direction<3>::upper_xi(),
                          {{1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17}});
    test_overlap_iterator(Index<3>{{{3, 3, 2}}}, 2, Direction<3>::lower_eta(),
                          {{0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14}});
    test_overlap_iterator(Index<3>{{{3, 3, 2}}}, 2, Direction<3>::upper_eta(),
                          {{3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17}});
    test_overlap_iterator(Index<3>{{{3, 2, 3}}}, 2, Direction<3>::lower_zeta(),
                          {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}});
    test_overlap_iterator(Index<3>{{{3, 2, 3}}}, 2, Direction<3>::upper_zeta(),
                          {{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}});
  }
  {
    INFO("Data on overlap");
    {
      INFO("1D");
      const Index<1> volume_extents{3};
      const size_t overlap_extent = 2;
      DataVector scalar{1., 2., 3.};
      // Overlap region: [X X O] -xi->
      test_data_on_overlap(scalar, volume_extents, overlap_extent,
                           Direction<1>::lower_xi(), {1., 2.}, {1., 2., 0.});
      // Overlap region: [O X X] -xi->
      test_data_on_overlap(scalar, volume_extents, overlap_extent,
                           Direction<1>::upper_xi(), {2., 3.}, {0., 2., 3.});
    }
    {
      INFO("2D");
      const Index<2> volume_extents{{{3, 2}}};
      const size_t overlap_extent_xi = 2;
      const size_t overlap_extent_eta = 1;
      DataVector scalar{1., 2., 3., 4., 5., 6.};
      // Overlap region:
      // + - - - + -xi->
      // | X X O |
      // | X X O |
      // + - - - +
      // v eta
      test_data_on_overlap(scalar, volume_extents, overlap_extent_xi,
                           Direction<2>::lower_xi(), {1., 2., 4., 5.},
                           {1., 2., 0., 4., 5., 0.});
      // Overlap region:
      // + - - - + -xi->
      // | O X X |
      // | O X X |
      // + - - - +
      // v eta
      test_data_on_overlap(scalar, volume_extents, overlap_extent_xi,
                           Direction<2>::upper_xi(), {2., 3., 5., 6.},
                           {0., 2., 3., 0., 5., 6.});
      // Overlap region:
      // + - - - + -xi->
      // | X X X |
      // | O O O |
      // + - - - +
      // v eta
      test_data_on_overlap(scalar, volume_extents, overlap_extent_eta,
                           Direction<2>::lower_eta(), {1., 2., 3.},
                           {1., 2., 3., 0., 0., 0.});
      // Overlap region:
      // + - - - + -xi->
      // | O O O |
      // | X X X |
      // + - - - +
      // v eta
      test_data_on_overlap(scalar, volume_extents, overlap_extent_eta,
                           Direction<2>::upper_eta(), {4., 5., 6.},
                           {0., 0., 0., 4., 5., 6.});
    }
    {
      INFO("3D");
      const Index<3> volume_extents{{{3, 2, 3}}};
      const size_t overlap_extent_xi = 2;
      const size_t overlap_extent_eta = 1;
      const size_t overlap_extent_zeta = 2;
      DataVector scalar{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
                        10., 11., 12., 13., 14., 15., 16., 17., 18.};
      test_data_on_overlap(
          scalar, volume_extents, overlap_extent_xi, Direction<3>::lower_xi(),
          {1., 2., 4., 5., 7., 8., 10., 11., 13., 14., 16., 17.},
          {1., 2., 0., 4., 5., 0., 7., 8., 0., 10., 11., 0., 13., 14., 0., 16.,
           17., 0.});
      test_data_on_overlap(
          scalar, volume_extents, overlap_extent_xi, Direction<3>::upper_xi(),
          {2., 3., 5., 6., 8., 9., 11., 12., 14., 15., 17., 18.},
          {0., 2., 3., 0., 5., 6., 0., 8., 9., 0., 11., 12., 0., 14., 15., 0.,
           17., 18.});
      test_data_on_overlap(scalar, volume_extents, overlap_extent_eta,
                           Direction<3>::lower_eta(),
                           {1., 2., 3., 7., 8., 9., 13., 14., 15.},
                           {1., 2., 3., 0., 0., 0., 7., 8., 9., 0., 0., 0., 13.,
                            14., 15., 0., 0., 0.});
      test_data_on_overlap(scalar, volume_extents, overlap_extent_eta,
                           Direction<3>::upper_eta(),
                           {4., 5., 6., 10., 11., 12., 16., 17., 18.},
                           {0., 0., 0., 4., 5., 6., 0., 0., 0., 10., 11., 12.,
                            0., 0., 0., 16., 17., 18.});
      test_data_on_overlap(scalar, volume_extents, overlap_extent_zeta,
                           Direction<3>::lower_zeta(),
                           {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.},
                           {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                            0., 0., 0., 0., 0., 0.});
      test_data_on_overlap(
          scalar, volume_extents, overlap_extent_zeta,
          Direction<3>::upper_zeta(),
          {7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.},
          {0., 0., 0., 0., 0., 0., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
           16., 17., 18.});
    }
  }
}

// [[OutputRegex, Overlap extent '4' exceeds volume extents]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.ParallelSchwarz.OverlapHelpers.AssertOverlapNumPoints",
    "[Unit][ParallelAlgorithms][LinearSolver]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  overlap_num_points(Index<1>{{{3}}}, 4, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Invalid dimension '1' in 1D]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ParallelSchwarz.OverlapHelpers.AssertDim",
                               "[Unit][ParallelAlgorithms][LinearSolver]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  overlap_num_points(Index<1>{{{3}}}, 0, 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

}  // namespace LinearSolver::Schwarz
