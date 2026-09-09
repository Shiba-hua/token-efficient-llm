import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_math import extract,inspect,fragment_errors

class MathMarkupTests(unittest.TestCase):
    def test_negative_controls_are_rejected(self):
        for source in [r'\operatorname{softmax}(x)',r'\operatorname*{argmax}_x x',r'\definitelyUnsupportedMacro{x}']:
            self.assertTrue(inspect(source),source)
    def test_mixed_document_excludes_program_code(self):
        text='`$not_math$`\n```python\nprint("$not_math$")\n```\n$`x+1`$\n```math\ny=2\n```\n$$z=3$$\n'
        self.assertEqual([x.tex.strip() for x in extract(text)],['x+1','y=2','z=3'])
    def test_denominator_missing_brace_is_rejected(self):
        self.assertIn('unbalanced_braces',inspect(r'\frac{x}{ab\cdot\max_s c.'))
        self.assertEqual(inspect(r'\frac{x}{ab\cdot\max_s c}'),[])
    def test_literal_set_braces_are_not_grouping(self):
        self.assertEqual(inspect(r'c\in\{0,1\}'),[])
    def test_transcription_and_literal_text_differ(self):
        self.assertTrue(inspect(r'x_1,ldots,x_n'))
        self.assertEqual(inspect(r'\text{hat}\quad\hat x'),[])
    def test_github_safe_operator_keeps_limit(self):
        self.assertEqual(inspect(r'\underset{y\in\mathcal C_x}{\mathrm{arg\,min}}\ell(y)'),[])
    def test_source_location_and_unprotected_dollars(self):
        f=extract('heading\n$plain$ and $`proper`$')[0]
        self.assertEqual((f.line,f.kind),(2,'unprotected_inline'))
    def test_quoted_math_fence_is_checked(self):
        fragments=extract('> ```math\n> \\operatorname{x}\n> ```\n')
        self.assertEqual(len(fragments),1)
        self.assertTrue(fragment_errors(fragments[0]))
    def test_unclosed_explicit_delimiters_fail(self):
        for source in ['$`x`','$$x','\\[x','```math\nx']:
            fragments=extract(source)
            self.assertTrue(any(fragment_errors(f) for f in fragments),source)
    def test_parenthesis_math_cannot_silently_disappear(self):
        fragments=extract('\\(x\\)')
        self.assertEqual(len(fragments),1)
        self.assertIn('unprotected_math_markup',fragment_errors(fragments[0]))

if __name__=='__main__':unittest.main()
