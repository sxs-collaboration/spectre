// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDeviceCapable.hpp"

namespace {
struct Trivial {
  double a;
  int b;
};

struct Polymorphic {
  Polymorphic() = default;
  Polymorphic(const Polymorphic&) = default;
  Polymorphic& operator=(const Polymorphic&) = default;
  Polymorphic(Polymorphic&&) = default;
  Polymorphic& operator=(Polymorphic&&) = default;
  virtual ~Polymorphic() = default;
  virtual double f() const { return 1.0; }
};

struct OwnsMemory {
  std::unique_ptr<double> data{};
};

// A type that isn't trivially copyable but can be copied into kernels
struct OptedIn {
  OptedIn() = default;
  OptedIn(const OptedIn& rhs) : copies(rhs.copies + 1) {}
  OptedIn& operator=(const OptedIn& rhs) {
    copies = rhs.copies + 1;
    return *this;
  }
  OptedIn(OptedIn&&) = default;
  OptedIn& operator=(OptedIn&&) = default;
  ~OptedIn() = default;
  size_t copies = 0;
};

struct DoubleTag {
  using type = double;
};
struct OptedInTag {
  using type = OptedIn;
};
struct DataVectorTag {
  using type = DataVector;
};

// A header-defined template that is device-callable for device-capable types
// only, as described in the GPU support docs
namespace detail {
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <typename T>
KOKKOS_FUNCTION void scale_impl(const gsl::not_null<T*> value,
                                const double factor) {
  ASSERT_DEVICE_CAPABLE(T);
  *value *= factor;
}
}  // namespace detail

template <typename T>
  requires tt::is_device_capable_v<T>
KOKKOS_FUNCTION void scale(const gsl::not_null<T*> value, const double factor) {
  detail::scale_impl(value, factor);
}

template <typename T>
  requires(not tt::is_device_capable_v<T>)
void scale(const gsl::not_null<T*> value, const double factor) {
  detail::scale_impl(value, factor);
}
}  // namespace

template <>
struct tt::is_device_capable<OptedIn> : std::true_type {};

static_assert(tt::is_device_capable_v<double>);
static_assert(tt::is_device_capable_v<const double&>);
static_assert(tt::is_device_capable_v<size_t, int, bool, char>);
static_assert(tt::is_device_capable_v<Trivial>);
static_assert(tt::is_device_capable_v<Scalar<double>>);
static_assert(tt::is_device_capable_v<tnsr::ii<double, 3>>);
static_assert(tt::is_device_capable_v<Mesh<3>>);
static_assert(tt::is_device_capable_v<double*, const double*, void*>);
static_assert(tt::is_device_capable_v<gsl::not_null<Scalar<double>*>>);
static_assert(tt::is_device_capable_v<gsl::span<const double>>);
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
static_assert(tt::is_device_capable_v<double[3]>);
static_assert(tt::is_device_capable_v<std::array<double, 3>>);
static_assert(tt::is_device_capable_v<std::pair<double, size_t>>);
static_assert(tt::is_device_capable_v<std::tuple<double, size_t>>);
static_assert(tt::is_device_capable_v<std::optional<double>>);
static_assert(tt::is_device_capable_v<Scalar<OptedIn>>);
static_assert(tt::is_device_capable_v<OptedIn>);
static_assert(tt::is_device_capable_v<std::array<OptedIn, 2>>);
static_assert(tt::is_device_capable_v<std::tuple<OptedIn>>);
static_assert(tt::is_device_capable_v<tuples::TaggedTuple<>>);
static_assert(tt::is_device_capable_v<tuples::TaggedTuple<DoubleTag>>);
static_assert(
    tt::is_device_capable_v<tuples::TaggedTuple<DoubleTag, OptedInTag>>);
static_assert(
    tt::is_device_capable_v<
        TempBuffer<tmpl::list<::Tags::TempScalar<0, double>,
                              ::Tags::TempI<1, 3, Frame::Inertial, double>>>>);

static_assert(not tt::is_device_capable_v<DataVector>);
static_assert(not tt::is_device_capable_v<const DataVector&>);
static_assert(not tt::is_device_capable_v<double, DataVector>);
static_assert(not tt::is_device_capable_v<Scalar<DataVector>>);
static_assert(not tt::is_device_capable_v<tnsr::I<DataVector, 3>>);
static_assert(not tt::is_device_capable_v<std::vector<double>>);
static_assert(not tt::is_device_capable_v<std::string>);
static_assert(not tt::is_device_capable_v<Polymorphic>);
static_assert(not tt::is_device_capable_v<OwnsMemory>);
static_assert(not tt::is_device_capable_v<std::complex<double>>);
static_assert(not tt::is_device_capable_v<DataVector*>);
static_assert(not tt::is_device_capable_v<gsl::not_null<DataVector*>>);
static_assert(not tt::is_device_capable_v<gsl::span<DataVector>>);
static_assert(not tt::is_device_capable_v<Polymorphic*>);
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
static_assert(not tt::is_device_capable_v<DataVector[3]>);
static_assert(not tt::is_device_capable_v<std::array<DataVector, 3>>);
static_assert(not tt::is_device_capable_v<std::pair<double, DataVector>>);
static_assert(not tt::is_device_capable_v<std::tuple<double, DataVector>>);
static_assert(not tt::is_device_capable_v<std::optional<DataVector>>);
static_assert(not tt::is_device_capable_v<double (*)(double)>);
static_assert(
    not tt::is_device_capable_v<tuples::TaggedTuple<DoubleTag, DataVectorTag>>);
static_assert(not tt::is_device_capable_v<
              TempBuffer<tmpl::list<::Tags::TempScalar<0, DataVector>>>>);

SPECTRE_TEST_CASE("Unit.Utilities.TypeTraits.IsDeviceCapable",
                  "[Unit][Utilities]") {
  // The check does nothing on the host, also for host-only types
  double x = 1.0;
  scale(make_not_null(&x), 2.0);
  CHECK(x == 2.0);
  DataVector y{1.0, 2.0};
  scale(make_not_null(&y), 2.0);
  CHECK(y == DataVector{2.0, 4.0});
}
