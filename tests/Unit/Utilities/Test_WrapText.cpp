// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "Utilities/WrapText.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.WrapText", "[Utilities][Unit]") {
  CHECK(
      wrap_text(
          "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          "aa",
          50, "         ") ==
      std::string("         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-\n"
                  "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-\n"
                  "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-\n"
                  "         aaaaaaaaaaaaaaaaaa"));
  CHECK(wrap_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                  "aaaaaaaaaaaaaaaaaa",
                  50) ==
        std::string("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-\n"
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-\n"
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));

  CHECK(wrap_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  50) ==
        std::string("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-\n"
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));

  CHECK(wrap_text("thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "I want to wrap \n\nthis text  "
                  "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "after about 50 characters please.",
                  50, "   ") ==
        std::string("   thisisareallylongwordthatwouldcauseproblemssow-\n"
                    "   eneedtoerror I want to wrap \n   \n   "
                    "this text \n"
                    "   thisisareallylongwordthatwouldcauseproblemssow-\n"
                    "   eneedtoerror after about 50 characters please."));
  CHECK(wrap_text("thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "I want to wrap \n\nthis text  "
                  "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "after about 50 characters please.",
                  50) ==
        std::string("thisisareallylongwordthatwouldcauseproblemssowene-\n"
                    "edtoerror I want to wrap \n\n"
                    "this text \n"
                    "thisisareallylongwordthatwouldcauseproblemssowene-\n"
                    "edtoerror after about 50 characters please."));
  CHECK(
      wrap_text("   thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                "I want to wrap \n\nthis text  "
                "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                "after about 50 characters please.",
                50) ==
      std::string("   thisisareallylongwordthatwouldcauseproblemssow-\n"
                  "eneedtoerror I want to wrap \n\n"
                  "this text \n"
                  "thisisareallylongwordthatwouldcauseproblemssowene-\n"
                  "edtoerror after about 50 characters please."));
  CHECK(wrap_text("I want to wrap \n\nthis text  "
                  "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "after about 50 characters please.",
                  50) ==
        std::string("I want to wrap \n\n"
                    "this text \n"
                    "thisisareallylongwordthatwouldcauseproblemssowene-\n"
                    "edtoerror after about 50 characters please."));
  CHECK(wrap_text("I want to wrap \n\nthis text  "
                  "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "after about 20 characters please.",
                  20) ==
        std::string("I want to wrap \n\nthis text \n"
                    "thisisareallylongwo-\nrdthatwouldcausepro-\n"
                    "blemssoweneedtoerror\nafter about 20\n"
                    "characters please."));
  CHECK(
      wrap_text("I want to wrap \n\nthis text\n"
                "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                "after about 20 characters please.",
                20) ==
      std::string("I want to wrap \n\nthis text\nthisisareallylongwo-\n"
                  "rdthatwouldcausepro-\nblemssoweneedtoerror\nafter about 20\n"
                  "characters please."));
  CHECK(wrap_text("I want to wrap \n\nthis text\nand this text\n"
                  "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "after about 50 characters please.",
                  50, "    ") ==
        std::string(
            "    I want to wrap \n    \n    this text\n    and this text\n"
            "    thisisareallylongwordthatwouldcauseproblemsso-\n"
            "    weneedtoerror after about 50 characters\n"
            "    please."));
  CHECK(wrap_text("I want to wrap \n\na\nand this text\n"
                  "thisisareallylongwordthatwouldcauseproblemssoweneedtoerror "
                  "after about 50 characters please.",
                  50, "    ") ==
        std::string(
            "    I want to wrap \n    \n    a\n    and this text\n"
            "    thisisareallylongwordthatwouldcauseproblemsso-\n"
            "    weneedtoerror after about 50 characters\n"
            "    please."));
}
