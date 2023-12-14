// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Base64.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
const std::array<char, 64> alphabet{
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

const char padding_char = '=';
}  // namespace

std::string base64_encode(const std::vector<std::byte>& data) {
  const size_t encoded_size = (data.size() + 2) / 3 * 4;
  std::string encoded{};
  encoded.reserve(encoded_size);

  auto octet = data.begin();
  while (octet != data.end()) {
    size_t bits = 0;
    size_t octets_in_group = 0;
    while (octets_in_group < 3 and octet != data.end()) {
      bits |= static_cast<size_t>(*octet) << 8 * (2 - octets_in_group);
      ++octet;
      ++octets_in_group;
    }

    const size_t characters_to_output = octets_in_group + 1;

    for (size_t i = 0; i < characters_to_output; ++i) {
      const auto hextet = 0x3F & (bits >> 6 * (3 - i));
      encoded.push_back(gsl::at(alphabet, hextet));
    }

    if (characters_to_output < 4) {
      encoded.push_back(padding_char);
      if (characters_to_output < 3) {
        encoded.push_back(padding_char);
      }
      ASSERT(
          characters_to_output >= 2,
          "Output an invalid number of characters: " << characters_to_output);
    }
  }

  ASSERT(encoded.size() == encoded_size,
         "Expected " << encoded_size << " characters for " << data.size()
                     << " bytes, but wrote " << encoded.size());
  return encoded;
}

std::vector<std::byte> base64_decode(const std::string& encoded) {
  static const std::array<std::byte, 256> reverse_alphabet = []() {
    std::array<std::byte, 256> local_reverse_alphabet{};
    std::fill(local_reverse_alphabet.begin(), local_reverse_alphabet.end(),
              std::byte{255});
    for (size_t i = 0; i < alphabet.size(); ++i) {
      gsl::at(local_reverse_alphabet,
              static_cast<size_t>(gsl::at(alphabet, i))) =
          static_cast<std::byte>(i);
    }
    gsl::at(local_reverse_alphabet, static_cast<size_t>(padding_char)) =
        std::byte{0};
    return local_reverse_alphabet;
  }();

  if (encoded.size() % 4 != 0) {
    ERROR("base64 encoded data must have a multiple of 4 characters, not "
          << encoded.size());
  }

  // Might be slightly too large.
  const size_t decoded_size_bound = encoded.size() / 4 * 3;
  std::vector<std::byte> decoded{};
  decoded.reserve(decoded_size_bound);

  auto encoded_char = encoded.begin();
  while (encoded_char != encoded.end()) {
    size_t bits = 0;
    int padding = 0;
    for (size_t i = 0; i < 4; ++i) {
      const auto hextet =
          gsl::at(reverse_alphabet, static_cast<size_t>(*encoded_char));
      if (hextet == std::byte{255}) {
        ERROR("Invalid character in base64 encoded string: '" << *encoded_char
                                                              << "'");
      }
      bits <<= 6;
      bits |= static_cast<size_t>(hextet);
      if (padding > 0 and *encoded_char != padding_char) {
        ERROR("Misplaced padding character at position "
              << (encoded_char - encoded.begin() - padding)
              << " of base64 encoded string.");
      }
      if (*encoded_char == padding_char) {
        ++padding;
      }
      ++encoded_char;
    }

    if (padding > 2 or (padding != 0 and encoded_char != encoded.end())) {
      ERROR("Misplaced padding character at position "
            << (encoded_char - encoded.begin() - padding)
            << " of base64 encoded string.");
    }

    const int octets_to_output = 3 - padding;
    for (int i = 0; i < octets_to_output; ++i) {
      const auto octet = 0xFF & (bits >> 8 * (2 - i));
      decoded.push_back(static_cast<std::byte>(octet));
    }
  }

  ASSERT(decoded.size() <= decoded_size_bound and
             decoded.size() + 3 > decoded_size_bound,
         "Expected approximately "
             << decoded_size_bound << " bytes for " << encoded.size()
             << " characters, but wrote " << decoded.size());
  return decoded;
}
