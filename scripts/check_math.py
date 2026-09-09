"""Inspect mathematical source independently of link checking.

A curated macro surface rejects unknown or known-incompatible commands. This is
not a TeX proof checker or a replacement for actual GitHub rendering. The JSON
inventory gives each source expression a stable content hash and exact location.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
# Kept deliberately explicit: an unknown macro must be examined, not silently accepted.
ALLOWED=set('''alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac dfrac tfrac sqrt sum prod int lim max min sup inf log ln exp sin cos tan det Pr gcd arg
left right middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr
begin end text textrm textbf textit texttt mathrm mathbf mathit mathsf mathtt mathcal mathbb boldsymbol
hat widehat bar overline underline tilde widetilde vec dot ddot underbrace overbrace overset underset
le leq leqslant ge geq geqslant lt gt ne neq approx sim simeq equiv propto to mapsto rightarrow leftarrow Rightarrow Leftarrow Leftrightarrow leftrightarrow longrightarrow Longrightarrow iff implies
in notin ni subset subseteq supset supseteq cup cap emptyset varnothing forall exists neg land lor lnot
cdot times div pm mp circ otimes oplus perp parallel mid vert Vert lvert rvert lVert rVert langle rangle lceil rceil lfloor rfloor
infty partial nabla ell hbar prime top bot ast star colon ldots cdots vdots ddots dots quad qquad thinspace displaystyle textstyle scriptstyle scriptscriptstyle limits nolimits substack binom dbinom tbinom cases
tag ll gg pmod triangleq rm dagger odot bigcup setminus over Longleftrightarrow nsubseteq mathbin mbox succ underleftarrow underrightarrow cancel phantom vphantom hphantom color boxed textnormal'''.split())
SUSPECT=re.compile(r'(?<!\\)\b(ldots|cdots|mathrm|mathcal|mathbb|mathbf|hat|tilde)\b')
@dataclass
class Fragment:
    start:int
    end:int
    tex:str
    kind:str
    line:int

def extract(text:str)->list[Fragment]:
    found=[];masked=list(text)
    def hide(a,b):
        for i in range(a,b):
            if masked[i]!='\n':masked[i]=' '
    def add(a,b,kind):
        found.append(Fragment(a,b,text[a:b],kind,text.count('\n',0,a)+1))
    # Match entire fences first, so example code cannot be mistaken for formulas.
    fence_view=re.sub(r'(?m)^ {0,3}(?:> ?)+',lambda m:' '*len(m[0]),text)
    pattern=re.compile(r'^[ \t]*(`{3,}|~{3,})([^\n]*)\n',re.M)
    cursor=0
    while (m:=pattern.search(fence_view,cursor)):
        close=re.search(r'^[ \t]*'+re.escape(m[1][0])+'{'+str(len(m[1]))+r',}\s*$',fence_view[m.end():],re.M)
        if not close:
            if m[2].strip()=='math':add(m.end(),len(text),'unclosed_math_fence')
            hide(m.start(),len(text));break
        a=m.end()+close.start();z=m.end()+close.end()
        if m[2].strip()=='math':add(m.end(),a,'display_fence')
        hide(m.start(),z);cursor=z
    for pat,kind in [(r'\$`([^`]*?)`\$','protected_inline'),(r'(?<!\\)\$\$(.*?)\$\$','display_dollars'),(r'\\\[(.*?)\\\]','display_brackets'),(r'\\\((.*?)\\\)','inline_parentheses')]:
        for m in re.finditer(pat,''.join(masked),re.S):add(m.start(1),m.end(1),kind);hide(m.start(),m.end())
    # Preserve a backtick span immediately opened by a dollar: it may be an
    # unterminated protected formula. Other ordinary code remains excluded.
    for m in re.finditer(r'(`+)(.+?)\1',''.join(masked),re.S):
        if m.start() and masked[m.start()-1]=='$':
            continue
        hide(m.start(),m.end())
    # These explicit delimiters must not disappear as ordinary code spans.
    for m in re.finditer(r'(?<!\\)(?:\$`|`\$|\$\$)|\\[\[\](]', ''.join(masked)):
        add(m.start(),m.end(),'unclosed_math_delimiter');hide(m.start(),m.end())
    for m in re.finditer(r'(?<![\\$])\$(?!\$)([^$\n]+)(?<!\\)\$(?!\$)',''.join(masked)):
        add(m.start(1),m.end(1),'unprotected_inline')
    return sorted(found,key=lambda f:f.start)

def inspect(tex:str)->list[str]:
    errors=[]
    for command in sorted(set(re.findall(r'\\([A-Za-z]+)',tex))):
        if command not in ALLOWED:errors.append('unsupported_or_unknown_macro: '+command)
    plain=re.sub(r'\\(?:text\w*|mathrm|mathsf|mathtt)\{[^{}]*\}','',tex)
    for m in SUSPECT.finditer(plain):errors.append('missing_command_backslash: '+m[1])
    depth=0
    for m in re.finditer(r'(?<!\\)[{}]',tex):
        depth+=1 if m[0]=='{' else -1
        if depth<0:errors.append('unmatched_closing_brace');break
    if depth:errors.append('unbalanced_braces')
    return list(dict.fromkeys(errors))

def fragment_errors(fragment:Fragment)->list[str]:
    errors=inspect(fragment.tex)
    if fragment.kind.startswith('unclosed_'):errors.append(fragment.kind)
    if fragment.kind in ('unprotected_inline','inline_parentheses','display_brackets'):
        errors.append('unprotected_math_markup')
    return errors

def collect():
    rows=[]
    for p in sorted((ROOT/'docs').rglob('*.md')):
        text=p.read_text()
        for n,f in enumerate(extract(text),1):
            rows.append({'file':str(p.relative_to(ROOT)),'line':f.line,'ordinal':n,'kind':f.kind,'sha256':hashlib.sha256(f.tex.encode()).hexdigest(),'tex':f.tex,'macros':sorted(set(re.findall(r'\\([A-Za-z]+)',f.tex))),'errors':fragment_errors(f)})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path);args=p.parse_args();rows=collect()
    if args.inventory:args.inventory.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    bad=[x for x in rows if x['errors']]
    for x in bad:print(f"{x['file']}:{x['line']}: {'; '.join(x['errors'])}")
    print(json.dumps({'math_fragments':len(rows),'pages_with_math':len({x['file'] for x in rows}),'source_errors':len(bad),'actual_github_rendering':'separate check required'}))
    raise SystemExit(bool(bad))
if __name__=='__main__':main()
