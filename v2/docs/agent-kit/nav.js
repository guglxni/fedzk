/** Shared sidebar for FEDzk agent-kit (HTML-first, AIDLC methodology). */
(function () {
  const pages = [
    { g: "Start", items: [
      ["index.html", "Hub / Index"],
      ["MANIFEST.html", "MANIFEST (boot)"],
      ["FORMAT.html", "FORMAT (HTML-first)"],
      ["CONCEPTS.html", "CONCEPTS (core)"],
      ["AIDLC.html", "AIDLC method map"],
      ["LOOPS.html", "LOOPS / conductor"],
      ["AGENTS.html", "AGENTS"],
      ["CLAUDE.html", "CLAUDE"],
      ["SKILL.html", "SKILL"],
    ]},
    { g: "AIDLC ops", items: [
      ["ROSTER.html", "ROSTER (personas)"],
      ["MEMORY.html", "MEMORY / rules"],
      ["rules/org.html", "rules/org"],
      ["rules/team.html", "rules/team"],
      ["rules/project.html", "rules/project"],
    ]},
    { g: "Product", items: [
      ["prd.html", "PRD"],
      ["PROPOSAL.html", "PROPOSAL"],
      ["REQUIREMENTS.html", "REQUIREMENTS"],
      ["QUALITY.html", "QUALITY / OSS"],
    ]},
    { g: "Design", items: [
      ["architecture.html", "Architecture"],
      ["DESIGN.html", "DESIGN"],
      ["STACK.html", "STACK (Py+Rust)"],
      ["MATH.html", "MATH"],
      ["RESEARCH.html", "RESEARCH"],
      ["PAPER.html", "PAPER plan"],
    ]},
    { g: "Build", items: [
      ["PLAN.html", "PLAN"],
      ["ARTIFACTS.html", "ARTIFACTS (freeze)"],
      ["REPOS.html", "REPOS + gh"],
      ["INTEGRATION.html", "INTEGRATION"],
      ["scratchpad.html", "scratchpad (live)"],
    ]},
  ];

  const path = location.pathname.replace(/\\/g, "/");
  const kitIdx = path.lastIndexOf("/agent-kit/");
  const afterKit = kitIdx >= 0 ? path.slice(kitIdx + "/agent-kit/".length) : (path.split("/").pop() || "index.html");
  const depth = (afterKit.match(/\//g) || []).length;
  const prefix = depth > 0 ? "../".repeat(depth) : "";

  const here = afterKit || "index.html";
  let html = `<div class="brand">FEDzk Agent Kit</div>
    <div class="brand-sub">HTML-first · AI-DLC method</div>`;
  for (const group of pages) {
    html += `<div class="group">${group.g}</div>`;
    for (const [href, label] of group.items) {
      const cls = href === here ? " active" : "";
      html += `<a class="${cls.trim()}" href="${prefix}${href}">${label}</a>`;
    }
  }
  const el = document.getElementById("sidebar");
  if (el) el.innerHTML = html;
})();
