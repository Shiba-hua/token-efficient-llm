import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import integrate_readings as publisher

class PublicationIntegrity(unittest.TestCase):
    def test_skip_published_never_reseals_changed_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for part in ['sources','meta','assets/papers/p1','docs/01-data/reference','scratch']:
                (root/part).mkdir(parents=True,exist_ok=True)
            paper={'paper_id':'p1','primary_chapter':'01-data','status':'published','artifact':'docs/01-data/reference/p1.md'}
            asset={'paper_id':'p1','file':'assets/papers/p1/figure.png','preview':'assets/papers/p1/figure.png','sha256':'previously-verified-source','preview_sha256':'previously-verified-preview'}
            (root/'sources/papers.json').write_text(json.dumps({'papers':[paper]}))
            (root/'meta/paper-assets.json').write_text(json.dumps([asset]))
            (root/'meta/pending-figures.json').write_text('[]')
            (root/asset['file']).write_bytes(b'changed bytes must remain detectable')
            (root/paper['artifact']).write_text('current mother text')
            with patch.object(publisher,'ROOT',root),patch.object(sys,'argv',['publish','--scratch',str(root/'scratch')]),contextlib.redirect_stdout(io.StringIO()):
                publisher.main()
            self.assertEqual(json.loads((root/'meta/paper-assets.json').read_text()),[asset])
            self.assertEqual((root/paper['artifact']).read_text(),'current mother text')
if __name__=='__main__':unittest.main()
