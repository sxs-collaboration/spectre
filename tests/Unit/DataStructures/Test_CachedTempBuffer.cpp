// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace Tags {
template <typename DataType>
struct Scalar1 : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct Scalar2 : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct Vector1 : db::SimpleTag {
  using type = tnsr::I<DataType, 2>;
};

template <typename DataType>
struct Vector2 : db::SimpleTag {
  using type = tnsr::I<DataType, 2>;
};

template <typename Tag>
struct Count {
  using type = size_t;
};
}  // namespace Tags

template <typename DataType>
struct Computer;

/// [alias]
template <typename DataType>
using Cache =
    CachedTempBuffer<Computer<DataType>, Tags::Scalar1<DataType>,
                     Tags::Scalar2<DataType>, Tags::Vector1<DataType>,
                     Tags::Vector2<DataType>>;
/// [alias]

template <typename DataType>
using Counter = tuples::TaggedTuple<
    Tags::Count<Tags::Scalar1<DataType>>, Tags::Count<Tags::Scalar2<DataType>>,
    Tags::Count<Tags::Vector1<DataType>>, Tags::Count<Tags::Vector2<DataType>>>;

template <typename DataType>
class Computer {
 public:
  explicit Computer(const gsl::not_null<Counter<DataType>*> counter) noexcept
      : counter_(counter) {}

  void operator()(const gsl::not_null<Scalar<DataType>*> scalar1,
                  const gsl::not_null<Cache<DataType>*> /*cache*/,
                  Tags::Scalar1<DataType> /*meta*/) const noexcept {
    ++get<Tags::Count<Tags::Scalar1<DataType>>>(*counter_);
    get(*scalar1) = 7.0;
  }

  /// [compute_func]
  void operator()(const gsl::not_null<Scalar<DataType>*> scalar2,
                  const gsl::not_null<Cache<DataType>*> cache,
                  Tags::Scalar2<DataType> /*meta*/) const noexcept {
    /// [compute_func]
    ++get<Tags::Count<Tags::Scalar2<DataType>>>(*counter_);
    const auto& vector1 = cache->get_var(Tags::Vector1<DataType>{});

    get(*scalar2) = get<1>(vector1);
  }

  void operator()(const gsl::not_null<tnsr::I<DataType, 2>*> vector1,
                  const gsl::not_null<Cache<DataType>*> /*cache*/,
                  Tags::Vector1<DataType> /*meta*/) const noexcept {
    ++get<Tags::Count<Tags::Vector1<DataType>>>(*counter_);
    get<0>(*vector1) = 10.0;
    get<1>(*vector1) = 11.0;
  }

  void operator()(const gsl::not_null<tnsr::I<DataType, 2>*> vector2,
                  const gsl::not_null<Cache<DataType>*> cache,
                  Tags::Vector2<DataType> /*meta*/) const noexcept {
    ++get<Tags::Count<Tags::Vector2<DataType>>>(*counter_);
    const auto& scalar2 = cache->get_var(Tags::Scalar2<DataType>{});
    const auto& vector1 = cache->get_var(Tags::Vector1<DataType>{});

    get<0>(*vector2) = get<0>(vector1) * get(scalar2);
    get<1>(*vector2) = get<1>(vector1) * get(scalar2);
  }

 private:
  gsl::not_null<Counter<DataType>*> counter_;
};

template <typename DataType>
void test_cached_temp_buffer(const DataType& used_for_size) noexcept {
  Counter<DataType> counter{};
  get<Tags::Count<Tags::Scalar1<DataType>>>(counter) = 0;
  get<Tags::Count<Tags::Scalar2<DataType>>>(counter) = 0;
  get<Tags::Count<Tags::Vector1<DataType>>>(counter) = 0;
  get<Tags::Count<Tags::Vector2<DataType>>>(counter) = 0;

  const auto check_counts = [&counter](
      const size_t scalar1, const size_t scalar2, const size_t vector1,
      const size_t vector2) noexcept {
    CHECK(get<Tags::Count<Tags::Scalar1<DataType>>>(counter) == scalar1);
    CHECK(get<Tags::Count<Tags::Scalar2<DataType>>>(counter) == scalar2);
    CHECK(get<Tags::Count<Tags::Vector1<DataType>>>(counter) == vector1);
    CHECK(get<Tags::Count<Tags::Vector2<DataType>>>(counter) == vector2);
  };

  Cache<DataType> cache(get_size(used_for_size), Computer<DataType>(&counter));
  check_counts(0, 0, 0, 0);
  CHECK(get(cache.get_var(Tags::Scalar1<DataType>{})) == 7.0);
  check_counts(1, 0, 0, 0);
  CHECK(get<0>(cache.get_var(Tags::Vector2<DataType>{})) == 110.0);
  check_counts(1, 1, 1, 1);
  CHECK(get<1>(cache.get_var(Tags::Vector2<DataType>{})) == 121.0);
  check_counts(1, 1, 1, 1);
  CHECK(get<0>(cache.get_var(Tags::Vector1<DataType>{})) == 10.0);
  check_counts(1, 1, 1, 1);

  CHECK(get_size(get(cache.get_var(Tags::Scalar1<DataType>{}))) ==
        get_size(used_for_size));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.CachedTempBuffer",
                  "[DataStructures][Unit]") {
  test_cached_temp_buffer(std::numeric_limits<double>::signaling_NaN());
  test_cached_temp_buffer(DataVector(5));
}
