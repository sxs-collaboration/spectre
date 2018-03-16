// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ForConstexpr.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void single_loop() {
  /// [single_loop]
  constexpr size_t array_size = 3;
  std::array<size_t, array_size> values{{0, 0, 0}};
  for_constexpr<for_bounds<0, array_size>>([&values](auto i) { values[i]++; });
  for (size_t i = 0; i < values.size(); ++i) {
    INFO("i:" << i);
    CHECK(gsl::at(values, i) == 1);
  }
  /// [single_loop]
}

void double_loop() {
  constexpr size_t array_size = 3;
  std::array<std::array<size_t, array_size>, array_size> zero{
      {{{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}}}};
  {  // Check double loop, no symmetry
    /// [double_loop]
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>>(
        [&values](auto i, auto j) { values[i][j]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        INFO("i:" << i << " j:" << j);
        CHECK(gsl::at(gsl::at(values, i), j) == 1);
      }
    }
    /// [double_loop]
  }
  {  // Check double loop, lower symmetry
    /// [double_symm_lower_exclusive]
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0>>(
        [&values](auto i, auto j) { values[i][j]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        INFO("i:" << i << " j:" << j);
        if (j < i) {
          CHECK(gsl::at(gsl::at(values, i), j) == 1);
        } else {
          CHECK(gsl::at(gsl::at(values, i), j) == 0);
        }
      }
    }
    /// [double_symm_lower_exclusive]
  }
  {  // Check double loop, lower symmetry, offset = 1
     /// [double_symm_lower_inclusive]
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>>(
        [&values](auto i, auto j) { values[i][j]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        INFO("i:" << i << " j:" << j);
        if (j <= i) {
          CHECK(gsl::at(gsl::at(values, i), j) == 1);
        } else {
          CHECK(gsl::at(gsl::at(values, i), j) == 0);
        }
      }
    }
    /// [double_symm_lower_inclusive]
  }
  {  // Check double loop, upper symmetry
    /// [double_symm_upper]
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>>(
        [&values](auto i, auto j) { values[i][j]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        INFO("i:" << i << " j:" << j);
        if (j >= i) {
          CHECK(gsl::at(gsl::at(values, i), j) == 1);
        } else {
          CHECK(gsl::at(gsl::at(values, i), j) == 0);
        }
      }
    }
    /// [double_symm_upper]
  }
}

void triple_loop() {
  constexpr size_t array_size = 3;
  std::array<std::array<size_t, array_size>, array_size> zero_2d{
      {{{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}}}};
  std::array<std::array<std::array<size_t, array_size>, array_size>, array_size>
      zero{{zero_2d, zero_2d, zero_2d}};

  {  // no symmetry
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_bounds<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
        }
      }
    }
  }
  {  // upper symmetric last on first
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_symm_upper<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (k >= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // upper symmetric last on second
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_symm_upper<1, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (k >= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // upper symmetric second on first
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_bounds<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // upper symmetric second on first, last on first
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_symm_upper<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i) {
            if (k >= i) {
              CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
              continue;
            }
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // upper symmetric second on first, last on second
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_symm_upper<1, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i) {
            if (k >= j) {
              CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
              continue;
            }
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
}

void triple_loop_lower_symmetric() {
  constexpr size_t array_size = 3;
  std::array<std::array<size_t, array_size>, array_size> zero_2d{
      {{{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}}}};
  std::array<std::array<std::array<size_t, array_size>, array_size>, array_size>
      zero{{zero_2d, zero_2d, zero_2d}};

  {  // lower symmetric last on first, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_symm_lower<0, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (k < i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // lower symmetric last on first, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_symm_lower<0, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (k <= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // lower symmetric last on second, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_symm_lower<1, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (k < j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // lower symmetric last on second, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_bounds<0, array_size>,
                  for_symm_lower<1, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (k <= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // lower symmetric second on first, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0>,
                  for_bounds<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // lower symmetric second on first, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_bounds<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
          } else {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
          }
        }
      }
    }
  }
  {  // lower symmetric second on first, exclusive, last on first, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 0>,
                  for_symm_lower<0, 0, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i and k < i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower symmetric second on first, exclusive, last on first, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 0>,
                  for_symm_lower<0, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i and k <= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower symmetric second on first, inclusive, last on first, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_symm_lower<0, 0, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i and k < i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower symmetric second on first, inclusive, last on first, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_symm_lower<0, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i and k <= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }

  {  // lower symmetric second on first, exclusive, last on second, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 0>,
                  for_symm_lower<1, 0, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i and k < j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower symmetric second on first, exclusive, last on second, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 0>,
                  for_symm_lower<1, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i and k <= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower symmetric second on first, inclusive, last on second, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_symm_lower<1, 0, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i and k < j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower symmetric second on first, inclusive, last on second, inclusive
    /// [triple_symm_lower_lower]
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_symm_lower<1, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i and k <= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
    /// [triple_symm_lower_lower]
  }
}

void triple_loop_mixed() {
  constexpr size_t array_size = 3;
  std::array<std::array<size_t, array_size>, array_size> zero_2d{
      {{{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}}}};
  std::array<std::array<std::array<size_t, array_size>, array_size>, array_size>
      zero{{zero_2d, zero_2d, zero_2d}};
  {  // lower/upper symmetric second on first, exclusive, last on first, upper
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0>,
                  for_symm_upper<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i and k >= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower/upper symmetric second on first, inclusive, last on first, upper
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_symm_upper<0, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i and k >= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower/upper symmetric second on first, exclusive, last on second, upper
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0>,
                  for_symm_upper<1, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j < i and k >= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // lower/upper symmetric second on first, inclusive, last on second, upper
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_lower<0, 0, 1>,
                  for_symm_upper<1, array_size>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j <= i and k >= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // upper/lower symmetric second on first, upper, last on first, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_symm_lower<0, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i and k < i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // upper/lower symmetric second on first, upper, last on first, inclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_symm_lower<0, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i and k <= i) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // upper/lower symmetric second on first, upper, last on second, exclusive
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_symm_lower<1, 0>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i and k < j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
  }
  {  // upper/lower symmetric second on first, upper, last on second, inclusive
    /// [triple_symm_upper_lower]
    auto values = zero;
    for_constexpr<for_bounds<0, array_size>, for_symm_upper<0, array_size>,
                  for_symm_lower<1, 0, 1>>(
        [&values](auto i, auto j, auto k) { values[i][j][k]++; });
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values.size(); ++j) {
        for (size_t k = 0; k < values.size(); ++k) {
          INFO("i:" << i << " j:" << j << " k:" << k);
          if (j >= i and k <= j) {
            CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 1);
            continue;
          }
          CHECK(gsl::at(gsl::at(gsl::at(values, i), j), k) == 0);
        }
      }
    }
    /// [triple_symm_upper_lower]
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.ForConstexpr", "[Utilities][Unit]") {
  single_loop();
  double_loop();
  triple_loop();
  triple_loop_mixed();
  triple_loop_lower_symmetric();
}
