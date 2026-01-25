// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "Parallel/FifoCache.hpp"

namespace {
struct T0 {
  int a;
  float b;
  bool operator==(const T0& rhs) const = default;
};

void test_basic() {
#if defined(SPECTRE_DEBUG)
  {
    CHECK_THROWS_WITH(Parallel::FifoCache<T0>(0u),
                      Catch::Matchers::ContainsSubstring(
                          "Must have a positive capacity but got 0"));
  }
#endif
  Parallel::FifoCache<T0> fc{3u};
  CHECK_FALSE(
      fc.find([](const T0& t0) -> bool { return t0.a == 1; }).has_value());
  {
    INFO("Test push.");
    const auto t0a1 =
        fc.push(T0{1, 1.32}, [](const T0& t0) -> bool { return t0.a == 1; });
    REQUIRE(t0a1.has_value());
    CHECK(t0a1.value() == T0{1, 1.32});
  }
  {
    INFO("Test finding pushed value more than once.");
    const auto t0a1 = fc.find([](const T0& t0) -> bool { return t0.a == 1; });
    REQUIRE(t0a1.has_value());
    CHECK(t0a1.value() == T0{1, 1.32});
    const auto t0a1_2 = fc.find([](const T0& t0) -> bool { return t0.a == 1; });
    REQUIRE(t0a1_2.has_value());
    CHECK(std::addressof(t0a1) != std::addressof(t0a1_2));
    CHECK(std::addressof(t0a1.value()) == std::addressof(t0a1_2.value()));
    CHECK_FALSE(
        fc.find([](const T0& t0) -> bool { return t0.a == 2; }).has_value());
  }
  {
    INFO("Test pushing same value doesn't change exist object.");
    const auto t0a1 = fc.find([](const T0& t0) -> bool { return t0.a == 1; });
    REQUIRE(t0a1.has_value());
    CHECK(t0a1.value() == T0{1, 1.32});
    const auto t0a1_2 =
        fc.push(T0{1, 1.32}, [](const T0& t0) -> bool { return t0.a == 1; });
    REQUIRE(t0a1_2.has_value());
    CHECK(std::addressof(t0a1) != std::addressof(t0a1_2));
    CHECK(std::addressof(t0a1.value()) == std::addressof(t0a1_2.value()));
    CHECK_FALSE(
        fc.find([](const T0& t0) -> bool { return t0.a == 2; }).has_value());
  }
  {
    INFO("Insert to fill.");
    {
      const auto a2 =
          fc.push(T0{2, 2.32}, [](const T0& t0) -> bool { return t0.a == 2; });
      REQUIRE(a2.has_value());
      CHECK(a2.value() == T0{2, 2.32});
    }
    {
      const auto a1 = fc.find([](const T0& t0) -> bool { return t0.a == 1; });
      REQUIRE(a1.has_value());
      CHECK(a1.value() == T0{1, 1.32});
    }
    {
      const auto a3 =
          fc.push(T0{3, 3.32}, [](const T0& t0) -> bool { return t0.a == 3; });
      REQUIRE(a3.has_value());
      CHECK(a3.value() == T0{3, 3.32});
    }
    {
      const auto a1 = fc.find([](const T0& t0) -> bool { return t0.a == 1; });
      REQUIRE(a1.has_value());
      CHECK(a1.value() == T0{1, 1.32});
      const auto a2 = fc.find([](const T0& t0) -> bool { return t0.a == 2; });
      REQUIRE(a2.has_value());
      CHECK(a2.value() == T0{2, 2.32});
      const auto a3 = fc.find([](const T0& t0) -> bool { return t0.a == 3; });
      REQUIRE(a3.has_value());
      CHECK(a3.value() == T0{3, 3.32});
    }
    {
      const auto a4 =
          fc.push(T0{4, 4.32}, [](const T0& t0) -> bool { return t0.a == 4; });
      REQUIRE(a4.has_value());
      CHECK(a4.value() == T0{4, 4.32});
    }
    {
      const auto a1 = fc.find([](const T0& t0) -> bool { return t0.a == 1; });
      REQUIRE_FALSE(a1.has_value());
      const auto a2 = fc.find([](const T0& t0) -> bool { return t0.a == 2; });
      REQUIRE(a2.has_value());
      CHECK(a2.value() == T0{2, 2.32});
      const auto a3 = fc.find([](const T0& t0) -> bool { return t0.a == 3; });
      REQUIRE(a3.has_value());
      CHECK(a3.value() == T0{3, 3.32});
      const auto a4 = fc.find([](const T0& t0) -> bool { return t0.a == 4; });
      REQUIRE(a4.has_value());
      CHECK(a4.value() == T0{4, 4.32});
    }
  }
  {
    INFO("Compute call is lazily evaluated.");
    {
      bool invoked_compute_value = false;
      const auto compute_value = [&invoked_compute_value]() {
        CHECK_FALSE(invoked_compute_value);
        invoked_compute_value = true;
        return T0{5, 5.32};
      };
      const auto a5 = fc.push(compute_value,
                              [](const T0& t0) -> bool { return t0.a == 5; });
      REQUIRE(invoked_compute_value);
      REQUIRE(a5.has_value());
      CHECK(a5.value() == T0{5, 5.32});

      invoked_compute_value = false;
      const auto a5b = fc.push(compute_value,
                               [](const T0& t0) -> bool { return t0.a == 5; });
      REQUIRE_FALSE(invoked_compute_value);
      REQUIRE(a5b.has_value());
      CHECK(a5b.value() == T0{5, 5.32});

      CHECK(std::addressof(a5) != std::addressof(a5b));
      CHECK(std::addressof(a5.value()) == std::addressof(a5b.value()));
    }
  }
}

void test_non_copyable() {
  Parallel::FifoCache<std::unique_ptr<T0>> fc{3u};
  CHECK_FALSE(fc.find([](const std::unique_ptr<T0>& t0) -> bool {
                  return t0->a == 1;
                }).has_value());
  const auto t0a1 =
      fc.push(std::make_unique<T0>(T0{1, 1.32}),
              [](const std::unique_ptr<T0>& t0) -> bool { return t0->a == 1; });
  REQUIRE(t0a1.has_value());
  REQUIRE(t0a1.value() != nullptr);
  CHECK(*(t0a1.value()) == T0{1, 1.32});
}

void test_deadlock() {
  Parallel::FifoCache<T0> fc{3u};
  CHECK_FALSE(
      fc.find([](const T0& t0) -> bool { return t0.a == 1; }).has_value());
  auto t0a1 =
      fc.push(T0{1, 1.32}, [](const T0& t0) -> bool { return t0.a == 1; });
  REQUIRE(t0a1.has_value());
  CHECK(t0a1.value() == T0{1, 1.32});

  std::atomic<int> t0_state{0};
  std::thread thread0{[&fc, &t0_state]() {
    t0_state.store(1, std::memory_order_release);
    const auto t0a2 =
        fc.push(T0{2, 1.32}, [](const T0& t0) -> bool { return t0.a == 2; });
    t0_state.store(2, std::memory_order_release);
  }};

  while (t0_state.load(std::memory_order_acquire) < 1) {
  }
  REQUIRE(t0_state.load(std::memory_order_acquire) == 1);

  // Wait 1s to give other thread plenty of time to not be deadlocked.
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  REQUIRE(t0_state.load(std::memory_order_acquire) == 1);

  t0a1 = std::nullopt;
  thread0.join();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.FifoCache", "[Parallel][Unit]") {
  test_basic();
  test_non_copyable();
  test_deadlock();
}
