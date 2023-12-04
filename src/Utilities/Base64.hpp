// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

/// Encode and decode data using the RFC4648 base64 format.
/// @{
std::string base64_encode(const std::vector<std::byte>& data);
std::vector<std::byte> base64_decode(const std::string& encoded);
/// @}
