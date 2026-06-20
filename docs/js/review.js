// Status-marking widget. Inert unless the local review server is running:
//   python tools/review_server.py    (writes back to docs/_authored/, then regenerates)
// With `mkdocs serve` also running, a click live-reloads the page with the new badge.
(function () {
  var API = "http://127.0.0.1:8017/status";
  var STATES = [["approved", "✅ approved"], ["review", "🟡 in review"],
                ["draft", "⬜ draft"]];
  function render(el) {
    var node = el.dataset.node, cat = el.dataset.cat, cur = el.dataset.status;
    el.innerHTML = "";
    var bar = document.createElement("div");
    bar.style.cssText = "display:flex;gap:.4em;align-items:center;flex-wrap:wrap;margin:.4em 0";
    var tag = document.createElement("span");
    tag.textContent = "Mark status:";
    tag.style.cssText = "font-size:.8em;opacity:.7";
    bar.appendChild(tag);
    STATES.forEach(function (s) {
      var b = document.createElement("button");
      b.textContent = s[1];
      b.style.cssText = "cursor:pointer;border:1px solid var(--md-default-fg-color--lightest);" +
        "border-radius:.5em;padding:.1em .5em;font-size:.8em;background:" +
        (s[0] === cur ? "var(--md-accent-fg-color);color:#fff" : "transparent");
      b.onclick = function () {
        fetch(API, {method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({cat: cat, node: node, status: s[0]})})
          .then(function (r) { if (!r.ok) throw new Error(); el.dataset.status = s[0]; render(el);
                               note(el, "saved — mkdocs will reload"); })
          .catch(function () { note(el, "start: python tools/review_server.py"); });
      };
      bar.appendChild(b);
    });
    el.appendChild(bar);
  }
  function note(el, msg) {
    var n = el.querySelector(".gm-note") || document.createElement("div");
    n.className = "gm-note";
    n.style.cssText = "font-size:.75em;opacity:.7;margin-top:.2em";
    n.textContent = msg;
    el.appendChild(n);
  }
  function init() { document.querySelectorAll(".gm-approval").forEach(render); }
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
  // Material instant-navigation hook (no-op if absent).
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
})();
