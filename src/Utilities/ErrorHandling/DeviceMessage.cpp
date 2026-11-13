// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/DeviceMessage.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#ifdef SPECTRE_KOKKOS
#include "Utilities/Kokkos/KokkosCore.hpp"
#endif  // SPECTRE_KOKKOS

void DeviceMessage::put(const char character) {
  if (truncated_) {
    return;
  }
  // Keep the last character for the null terminator
  if (size_ + 1 < capacity) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    buffer_[size_] = character;
    ++size_;
  } else {
    // Mark the truncation by replacing the last characters with "..."
    for (size_t i = size_ - 3; i < size_; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      buffer_[i] = '.';
    }
    truncated_ = true;
  }
}

void DeviceMessage::append(const char* str) {
  if (str == nullptr) {
    str = "(null)";
  }
  for (; *str != '\0'; ++str) {  // NOLINT(*-pro-bounds-pointer-arithmetic)
    put(*str);
  }
}

void DeviceMessage::append_unsigned(unsigned long long value,
                                    const size_t min_digits) {
  // Enough for the largest value
  std::array<char, std::numeric_limits<unsigned long long>::digits10 + 1>
      digits{};
  size_t number_of_digits = 0;
  do {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    digits[number_of_digits] = static_cast<char>('0' + value % 10);
    value /= 10;
    ++number_of_digits;
  } while (value != 0 or number_of_digits < min_digits);
  while (number_of_digits > 0) {
    --number_of_digits;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    put(digits[number_of_digits]);
  }
}

void DeviceMessage::append_signed(const long long value) {
  if (value < 0) {
    put('-');
    // Negate as unsigned so the most negative value doesn't overflow
    append_unsigned(0ULL - static_cast<unsigned long long>(value));
  } else {
    append_unsigned(static_cast<unsigned long long>(value));
  }
}

// Formats like "%.16e". Device code has no `snprintf`, so the digits are
// computed directly. Scaling to [1, 10) rounds, so the last digit can differ
// from the exact decimal representation
void DeviceMessage::append_double(double value) {
  if (std::isnan(value)) {
    append("nan");
    return;
  }
  if (std::signbit(value)) {
    put('-');
    value = -value;
  }
  if (std::isinf(value)) {
    append("inf");
    return;
  }
  // The 17 significant digits are `digits / 10^16`, '.', `digits % 10^16`
  constexpr unsigned long long ten_to_16 = 10'000'000'000'000'000ULL;
  int exponent = 0;
  unsigned long long digits = 0;
  if (value != 0.0) {
    exponent = static_cast<int>(std::floor(std::log10(value)));
    // Powers of ten up to 1e22 are exact, so for most values this is a single
    // rounding. Scale in two steps for denormals, where 10^-exponent
    // overflows
    double scaled = value;
    if (exponent >= 0) {
      scaled /= std::pow(10.0, exponent);
    } else if (exponent >= -300) {
      scaled *= std::pow(10.0, -exponent);
    } else {
      scaled = (scaled * 1.0e300) * std::pow(10.0, -(exponent + 300));
    }
    // `log10` can be off by one near powers of ten
    if (scaled >= 10.0) {
      scaled /= 10.0;
      ++exponent;
    } else if (scaled < 1.0) {
      scaled *= 10.0;
      --exponent;
    }
    digits = static_cast<unsigned long long>(
        std::llround(scaled * static_cast<double>(ten_to_16)));
    // Rounding can carry into the next decade, e.g. 9.99...95 -> 10.0
    if (digits >= 10 * ten_to_16) {
      digits /= 10;
      ++exponent;
    }
  }
  put(static_cast<char>('0' + digits / ten_to_16));
  put('.');
  append_unsigned(digits % ten_to_16, 16);
  put('e');
  put(exponent < 0 ? '-' : '+');
  append_unsigned(
      static_cast<unsigned long long>(exponent < 0 ? -exponent : exponent), 2);
}

#ifdef SPECTRE_KOKKOS
namespace DeviceMessage_detail {
namespace {
// `Kokkos::abort` isn't `[[noreturn]]` in some configurations (e.g. CUDA with
// bounds checking). It doesn't return on CUDA and HIP devices. (On SYCL
// devices it doesn't abort yet, which SpECTRE doesn't support)
[[noreturn]] KOKKOS_FUNCTION void abort_with_message(
    const char* const message) {
  ::Kokkos::abort(message);
  __builtin_unreachable();
}
}  // namespace

// The function-local statics below are shared by all threads of the device
// (and, separately, of the host). They are never reset because the program
// aborts

bool claim_failure_message() {
  static unsigned int number_of_failures = 0;
  return ::Kokkos::atomic_fetch_add(&number_of_failures, 1U) == 0;
}

DeviceMessage& start_failure_message(const char* const heading,
                                     const char* const description,
                                     const char* const file, const int line) {
  // Only the first failing thread writes to this buffer, so it doesn't take
  // memory in every GPU thread
  static DeviceMessage message{};
  // Start on a new line because CUDA and HIP print the message after the
  // block and thread indices
  message << '\n' << heading << " in " << file << ':' << line << '\n';
  if (*description != '\0') {
    message << description << '\n';
  }
  return message;
}

void finish_failure_message(DeviceMessage& message,
                            const char* const pretty_function) {
  // No newline at the end, so CUDA's and HIP's closing quote around the
  // message ends the last line
  message << "\nin " << pretty_function;
  // A single print, so the message doesn't interleave with other output
  abort_with_message(message.c_str());
}

// See SPECTRE_ABORT_ON_DEVICE dox
void end_other_failing_thread(const char* const abort_message) {
#ifdef __CUDA_ARCH__
  asm volatile("exit;");
#elif defined(__HIP_DEVICE_COMPILE__)
  for (unsigned int i = 0; i < (1U << 20); ++i) {
    __builtin_amdgcn_s_sleep(1);
  }
#endif
  abort_with_message(abort_message);
}
}  // namespace DeviceMessage_detail
#endif  // SPECTRE_KOKKOS
