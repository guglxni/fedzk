/** FEDzk agent-kit machine hooks (HTML-first). */
(function () {
  // Copy-to-clipboard for dispatch / role briefs
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sel = btn.getAttribute("data-copy");
      const el = document.querySelector(sel);
      if (!el) return;
      const text = el.innerText || el.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        const prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = prev; }, 1200);
      } catch (_) {
        btn.textContent = "Copy failed";
      }
    });
  });

  // Inject page TOC from h2[id] when #page-toc exists
  const tocHost = document.getElementById("page-toc");
  if (tocHost) {
    const heads = [...document.querySelectorAll("main h2[id]")];
    if (heads.length) {
      const ol = document.createElement("ol");
      heads.forEach((h) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#" + h.id;
        a.textContent = h.textContent.replace(/\s*—.*$/, "").trim();
        li.appendChild(a);
        ol.appendChild(li);
      });
      tocHost.appendChild(ol);
    }
  }

  // Expose load hints for agents scraping the DOM
  document.documentElement.dataset.kitFormat = "html-first";
  document.documentElement.dataset.forbidUpstreamMd = "vendor/aidlc-workflows/**/*.md";
})();
