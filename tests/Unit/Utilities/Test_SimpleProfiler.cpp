// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstdint>
#include <ctime>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/SimpleProfiler.hpp"

namespace {
// Extra time in milliseconds allowed in the timing results.
const std::uint64_t timing_slop = 5;

std::uint64_t nanosec(const std::uint64_t msec) { return 1'000'000u * msec; }

void msleep(const std::uint64_t msec) {
  // Test shouldn't be sleeping for a full second.
  const timespec spec{.tv_sec = 0, .tv_nsec = static_cast<long>(nanosec(msec))};
  nanosleep(&spec, nullptr);
}

struct Plain {
  using FileLabel = profiling::NoLabel;
  using ColumnLabel = profiling::NoLabel;

  static std::string make_file_name(const profiling::NoLabel& /*file_label*/) {
    return "Plain";
  }
};

// [labels]
struct File {
  using FileLabel = int;
  using ColumnLabel = profiling::NoLabel;

  static std::string make_file_name(const int file_label) {
    return "File" + std::to_string(file_label);
  }
};

enum class Column : std::uint8_t { A, B, C, AlsoA };

struct Columns {
  using FileLabel = profiling::NoLabel;
  using ColumnLabel = Column;

  static constexpr std::array allowed_column_labels{Column::A, Column::B,
                                                    Column::AlsoA};

  static std::string make_file_name(const profiling::NoLabel& /*file_label*/) {
    return "Columns";
  }

  static std::string make_column_name(const Column column_label) {
    switch (column_label) {
      case Column::A:
      case Column::AlsoA:
        return "A";
      case Column::B:
        return "B";
      default:
        ERROR(static_cast<int>(column_label));
    }
  }
};
// [labels]

struct Both {
  using FileLabel = int;
  using ColumnLabel = Column;

  static constexpr std::array allowed_column_labels{Column::A, Column::C};

  static std::string make_file_name(const int file_label) {
    return "Both" + std::to_string(file_label);
  }

  static std::string make_column_name(const Column column_label) {
    switch (column_label) {
      case Column::A:
        return "A";
      case Column::C:
        return "C";
      default:
        ERROR(static_cast<int>(column_label));
    }
  }
};

// Column name, count, exclusive, inclusive (milliseconds)
using ExpectedColumnData =
    std::tuple<std::string, std::uint64_t, std::uint64_t, std::uint64_t>;

void check_columns(const profiling::SimpleProfilerData::TimingData& data,
                   const std::string& file_name,
                   const std::vector<ExpectedColumnData>& expected_columns) {
  CAPTURE(file_name);
  const auto& file_data = data.at(file_name);
  CHECK(file_data.size() == expected_columns.size());
  for (const auto& expected : expected_columns) {
    CAPTURE(get<0>(expected));
    const auto& timing = file_data.at(get<0>(expected));
    CHECK(timing.call_count == get<1>(expected));
    if (get<2>(expected) == get<3>(expected)) {
      CHECK(timing.exclusive_nsec == timing.inclusive_nsec);
    }
    CHECK(timing.exclusive_nsec >= nanosec(get<2>(expected)));
    CHECK(timing.exclusive_nsec < nanosec(get<2>(expected) + timing_slop));
    CHECK(timing.inclusive_nsec >= nanosec(get<3>(expected)));
    CHECK(timing.inclusive_nsec < nanosec(get<3>(expected) + timing_slop));
  }
}

bool test(const bool enable) {
  std::uint64_t unprofiled = 0;
  std::uint64_t unprofiled_calls = 0;
  std::uint64_t plain = 0;
  std::uint64_t plain_calls = 0;
  std::uint64_t file3_ex = 0;
  std::uint64_t file3_in = 0;
  std::uint64_t file3_calls = 0;
  std::uint64_t colA_ex = 0;
  std::uint64_t colA_in = 0;
  std::uint64_t colA_calls = 0;
  std::uint64_t file7 = 0;
  std::uint64_t file7_calls = 0;
  std::uint64_t bothA_ex = 0;
  std::uint64_t bothA_in = 0;
  std::uint64_t bothA_calls = 0;
  std::uint64_t bothC = 0;
  std::uint64_t bothC_calls = 0;

  timespec test_start_time{};
  clock_gettime(CLOCK_MONOTONIC, &test_start_time);

  profiling::SimpleProfilerData profiler_data{};
  profiling::SimpleProfilerData* const data = enable ? &profiler_data : nullptr;
  ++unprofiled_calls;
  msleep(10);
  unprofiled += 10;
  {
    const profiling::SimpleProfiler<Plain> profiler(data);
    ++plain_calls;
    msleep(10);
    plain += 10;
  }
  ++unprofiled_calls;
  {
    const profiling::SimpleProfiler<File> prof1(data, 3);
    ++file3_calls;
    msleep(10);
    file3_ex += 10;
    file3_in += 10;
    const profiling::SimpleProfiler<Columns> prof2(data, Column::A);
    ++colA_calls;
    msleep(10);
    colA_ex += 10;
    colA_in += 10;
    file3_in += 10;
    {
      const profiling::SimpleProfiler<Columns> prof3(data, Column::A);
      ++colA_calls;
      msleep(10);
      colA_ex += 10;
      colA_in += 10;
      colA_in += 10;
      file3_in += 10;
    }
    {
      profiling::SimpleProfiler<File> prof4(data, 17);  // changed to 7
      ++file7_calls;
      msleep(10);
      file7 += 10;
      colA_in += 10;
      file3_in += 10;
      prof4.set_file_label(7);
    }
  }
  ++unprofiled_calls;
  {
    const profiling::SimpleProfiler<Columns> profiler(data, Column::AlsoA);
    ++colA_calls;
    msleep(10);
    colA_ex += 10;
    colA_in += 10;
  }

  ++unprofiled_calls;
  msleep(10);
  unprofiled += 10;

  const auto make_profiler = [&]() {
    profiling::SimpleProfiler<Both> prof(data, 2, Column::C);
    msleep(10);
    return prof;
  };
  {
    auto prof1 = make_profiler();  // BothC, changed to A
    ++bothA_calls;
    bothA_ex += 10;
    bothA_in += 10;
    msleep(10);
    bothA_ex += 10;
    bothA_in += 10;
    prof1.set_column_label(Column::A);
    auto prof2 = make_profiler();  // BothC
    ++bothC_calls;
    bothC += 10;
    bothA_in += 10;
    // Explicitly test moving a profiler, since the moves from
    // make_profiler() can be elided by RVO.
    auto prof2_moved = std::move(prof2);
    msleep(10);
    bothC += 10;
    bothA_in += 10;
    prof2_moved.end_profiling();
    msleep(10);
    bothA_ex += 10;
    bothA_in += 10;
  }

  ++unprofiled_calls;
  msleep(10);
  unprofiled += 10;

  if (not enable) {
    // Testing that the interface works passing a null pointer for the
    // data.  There is no actual output to test.
    return true;
  }

  const auto timing_data = data->timing_data();

  timespec test_end_time{};
  clock_gettime(CLOCK_MONOTONIC, &test_end_time);
  const std::uint64_t test_run_time =
      static_cast<std::uint64_t>(test_end_time.tv_sec -
                                 test_start_time.tv_sec) *
          1'000'000'000u +
      static_cast<std::uint64_t>(test_end_time.tv_nsec) -
      static_cast<std::uint64_t>(test_start_time.tv_nsec);

  const std::uint64_t expected_total_time =
      unprofiled + plain + file3_ex + colA_ex + file7 + bothA_ex + bothC;
  if (test_run_time > nanosec(expected_total_time + timing_slop)) {
    return false;
  }

  CHECK(timing_data.size() == 6);
  check_columns(timing_data, "Unprofiled",
                {{"", unprofiled_calls, unprofiled, unprofiled}});
  check_columns(timing_data, "Plain", {{"", plain_calls, plain, plain}});
  check_columns(timing_data, "File3", {{"", file3_calls, file3_ex, file3_in}});
  check_columns(timing_data, "File7", {{"", file7_calls, file7, file7}});
  check_columns(timing_data, "Columns",
                {{"A", colA_calls, colA_ex, colA_in}, {"B", 0, 0, 0}});
  check_columns(timing_data, "Both2",
                {{"A", bothA_calls, bothA_ex, bothA_in},
                 {"C", bothC_calls, bothC, bothC}});
  return true;
}

bool test_serialization() {
  std::uint64_t unprofiled = 0;
  std::uint64_t unprofiled_calls = 0;
  std::uint64_t plain = 0;
  std::uint64_t plain_calls = 0;

  timespec test_start_time{};
  clock_gettime(CLOCK_MONOTONIC, &test_start_time);

  profiling::SimpleProfilerData data{};
  ++unprofiled_calls;
  msleep(10);
  unprofiled += 10;
  {
    const profiling::SimpleProfiler<Plain> profiler(&data);
    ++plain_calls;
    msleep(10);
    plain += 10;
  }
  ++unprofiled_calls;
  msleep(10);
  unprofiled += 10;
  auto copy = serialize_and_deserialize(data);
  ++unprofiled_calls;
  msleep(10);
  unprofiled += 10;
  {
    const profiling::SimpleProfiler<Plain> profiler(&data);
    const profiling::SimpleProfiler<Plain> copy_profiler(&copy);
    ++plain_calls;
    msleep(10);
    plain += 10;
  }
  ++unprofiled_calls;

  const auto timing_data = data.timing_data();
  const auto copy_timing_data = copy.timing_data();

  timespec test_end_time{};
  clock_gettime(CLOCK_MONOTONIC, &test_end_time);
  const std::uint64_t test_run_time =
      static_cast<std::uint64_t>(test_end_time.tv_sec -
                                 test_start_time.tv_sec) *
          1'000'000'000u +
      static_cast<std::uint64_t>(test_end_time.tv_nsec) -
      static_cast<std::uint64_t>(test_start_time.tv_nsec);

  const std::uint64_t expected_total_time = unprofiled + plain;
  if (test_run_time > nanosec(expected_total_time + timing_slop)) {
    return false;
  }

  CHECK(timing_data.size() == 2);
  CHECK(copy_timing_data.size() == 2);
  check_columns(timing_data, "Unprofiled",
                {{"", unprofiled_calls, unprofiled, unprofiled}});
  check_columns(timing_data, "Plain", {{"", plain_calls, plain, plain}});
  check_columns(copy_timing_data, "Unprofiled",
                {{"", unprofiled_calls, unprofiled, unprofiled}});
  check_columns(copy_timing_data, "Plain", {{"", plain_calls, plain, plain}});

  // Test serializing empty data.  Early versions had a bug here that
  // would write the wrong amount of data to the serialization buffer
  // and corrupt other objects.
  {
    std::tuple<int, profiling::SimpleProfilerData, int> tuple{};
    get<0>(tuple) = 1234;
    get<2>(tuple) = 5678;
    const auto tuple_copy = serialize_and_deserialize(tuple);
    CHECK(get<0>(tuple_copy) == 1234);
    CHECK(get<2>(tuple_copy) == 5678);
  }
  return true;
}

SPECTRE_TEST_CASE("Unit.Utilities.SimpleProfiler", "[Utilities][Unit]") {
  // The test it timing based, so it will occasionally fail because
  // the system is busy.  Allow a few retries if the timed portion ran
  // for too long.
  for (size_t attempt = 0;; ++attempt) {
    if (test(true)) {
      break;
    }
    REQUIRE(attempt < 5);
  }
  for (size_t attempt = 0;; ++attempt) {
    if (test(false)) {
      break;
    }
    REQUIRE(attempt < 5);
  }
  for (size_t attempt = 0;; ++attempt) {
    if (test_serialization()) {
      break;
    }
    REQUIRE(attempt < 5);
  }
}
}  // namespace
