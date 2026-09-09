// Read-only DOM probe for the browser skill's tab.playwright.evaluate().
// A present math element alone is insufficient: inspect errors and pending work.
() => {
  const article = document.querySelector('article.markdown-body') || document.querySelector('article');
  if (!article) return {url:location.href,article:false,ready:false};
  const renderers = [...article.querySelectorAll('math-renderer')];
  const pattern = /following macros are not allowed|Unable to render|Undefined control sequence|KaTeX parse error|Error parsing|Extra close brace|Missing close brace|unknown command|unknown environment/i;
  const failures = [...article.querySelectorAll('math-renderer, .flash-error, .katex-error, merror, mjx-merror')]
    .filter(x => /\b(flash-error|katex-error)\b/.test(x.getAttribute('class') || '') || /^(merror|mjx-merror)$/i.test(x.tagName) || (!x.querySelector('math') && pattern.test(x.textContent || '')))
    .map(x=>({tag:x.tagName,text:(x.textContent||'').slice(0,900)}));
  const pending = renderers.filter(x=>!x.querySelector('math') && !x.querySelector('.flash-error, .katex-error, merror, mjx-merror') && !pattern.test(x.textContent||''));
  const unparsed=[];
  for (const element of article.querySelectorAll('p,li,td,th,h1,h2,h3,h4')) {
    let value=element.textContent || '';
    for (const code of element.querySelectorAll('code,pre,math-renderer')) value=value.replace(code.textContent || '', '');
    if (/\$`|`\$|\$\$/.test(value)) unparsed.push(value.slice(0,200));
  }
  return {url:location.href,article:true,title:article.querySelector('h1')?.textContent,
    renderers:renderers.length,mathml:article.querySelectorAll('math').length,
    failures,unparsed,pending:pending.length,ready:pending.length===0,
    images:[...article.querySelectorAll('img')].map(x=>({alt:x.alt,complete:x.complete,width:x.naturalWidth}))};
}
