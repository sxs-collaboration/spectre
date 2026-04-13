# Distributed under the MIT License.
# See LICENSE.txt for details.

import importlib.util
import shutil
import textwrap
import unittest
from pathlib import Path

from spectre.Informer import unit_test_build_path, unit_test_src_path


def load_postprocess_docs_module():
    module_path = (
        Path(unit_test_src_path()).parents[1]
        / "docs/config/postprocess_docs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "postprocess_docs", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


postprocess_docs = load_postprocess_docs_module()


class TestPostprocessDocs(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(unit_test_build_path(), "tools/PostprocessDocs")
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.html_dir = self.test_dir / "html"
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.citelist_path = self.html_dir / "citelist.html"
        self.references_path = self.test_dir / "References.bib"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fix_empty_bibliography_list_and_append_eprint(self):
        self.citelist_path.write_text(
            textwrap.dedent("""\
                <html>
                <body>
                <dl>
                <dt><a class="anchor" id="CITEREF_Ayachour2003"></a>[1]</dt>
                <dd><p class="startdd">E.H. Ayachour. <a href="http://www.sciencedirect.com/science/article/pii/S037704270300534X">A fast implementation for GMRES method</a>. <em>Journal of Computational and Applied Mathematics</em>, 159(2):269 &ndash; 283,</p><ol type="1"><li></li></ol><p class="enddd"></p></dd>
                <dt>
                <a class="anchor" id="CITEREF_Alcubierre2003pc"></a>[2]
                </dt>
                <dd><p class="startdd">Miguel Alcubierre and others. <a href="https://doi.org/10.1088/0264-9381/21/2/019">Toward standard testbeds for numerical relativity</a>. <em>Class. Quant. Grav.</em>, 21(2):589&ndash;613, 2004.</p><p class="enddd"></p></dd>
                </dl>
                </body>
                </html>
                """),
            encoding="utf-8",
        )
        self.references_path.write_text(
            textwrap.dedent("""\
                @article{Ayachour2003,
                  author  = "Ayachour, E.H.",
                  title   = "A fast implementation for {GMRES} method",
                  journal =
                    "Journal of Computational and Applied Mathematics",
                  volume  = "159",
                  number  = "2",
                  pages   = "269 - 283",
                  year    = "2003",
                  doi     = "10.1016/S0377-0427(03)00534-X",
                  url     = "http://www.sciencedirect.com/science/article/pii/S037704270300534X"
                }

                @article{Alcubierre2003pc,
                  author        = "Alcubierre, Miguel and others",
                  title         =
                    "Toward standard testbeds for numerical relativity",
                  journal       = "Class. Quant. Grav.",
                  volume        = "21",
                  number        = "2",
                  pages         = "589--613",
                  year          = "2004",
                  archivePrefix = "arXiv",
                  eprint        = "gr-qc/0305023",
                  doi           = "10.1088/0264-9381/21/2/019",
                  url           = "https://doi.org/10.1088/0264-9381/21/2/019"
                }
                """),
            encoding="utf-8",
        )

        postprocess_docs.append_eprint_links_to_citelist(
            str(self.html_dir), [str(self.references_path)]
        )
        postprocess_docs.append_eprint_links_to_citelist(
            str(self.html_dir), [str(self.references_path)]
        )

        processed_citelist = self.citelist_path.read_text(encoding="utf-8")
        self.assertNotIn('<ol type="1">', processed_citelist)
        self.assertIn("159(2):269 – 283, 2003.", processed_citelist)
        self.assertIn(
            (
                '2004. <a href="https://arxiv.org/abs/gr-qc/0305023">'
                "arXiv:gr-qc/0305023</a>."
            ),
            processed_citelist,
        )
        self.assertEqual(
            processed_citelist.count(
                'href="https://arxiv.org/abs/gr-qc/0305023"'
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
