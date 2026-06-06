const sampleResponse = {
  status: "success",
  problem:
    "傾角 $30^\\circ$ のなめらかな斜面上に質量 $m$ の小物体を静かに置く。重力加速度を $g$ とする。斜面方向の加速度を求めよ。",
  solution:
    "斜面方向の重力成分は $mg\\sin 30^\\circ$ である。運動方程式より $ma=mg\\sin 30^\\circ$。したがって、$a=g\\sin 30^\\circ=\\frac{g}{2}$ である。",
  diagram: {
    type: "svg",
    description: "傾角30度の斜面上の物体",
    spec: {
      kind: "inclined_plane",
      angle_deg: 30,
      block_label: "m",
      forces: ["mg", "N"]
    },
    svg:
      '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220" viewBox="0 0 360 220" role="img" aria-label="傾角30度の斜面上の物体"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#222"/></marker><style>.line{stroke:#222;stroke-width:3;fill:none}.thin{stroke:#555;stroke-width:2;fill:none}.label{font:16px sans-serif;fill:#111}</style></defs><path class="line" d="M40 170 L320 170 L320 70 Z"/><g transform="translate(190 110) rotate(-30)"><rect x="-26" y="-18" width="52" height="36" fill="#f5f5f5" stroke="#222" stroke-width="2"/><text x="-7" y="6" class="label">m</text></g><line x1="190" y1="110" x2="190" y2="158" class="thin" marker-end="url(#arrow)"/><text x="198" y="150" class="label">mg</text><line x1="190" y1="110" x2="160" y2="80" class="thin" marker-end="url(#arrow)"/><text x="142" y="78" class="label">N</text><path class="thin" d="M265 170 A45 45 0 0 0 249 137"/><text x="252" y="160" class="label">30°</text></svg>'
  }
};

const jsonInput = document.getElementById("jsonInput");
const fileInput = document.getElementById("fileInput");
const renderButton = document.getElementById("renderButton");
const sampleButton = document.getElementById("sampleButton");
const statusMessage = document.getElementById("statusMessage");
const problemOutput = document.getElementById("problemOutput");
const solutionOutput = document.getElementById("solutionOutput");
const diagramMeta = document.getElementById("diagramMeta");
const diagramOutput = document.getElementById("diagramOutput");

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`.trim();
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function appendTextWithBreaks(container, value) {
  clearNode(container);
  container.classList.remove("empty");

  const text = String(value || "").trim();
  if (!text) {
    container.textContent = "No content loaded.";
    container.classList.add("empty");
    return;
  }

  const blocks = text.split(/\n{2,}/);
  blocks.forEach((block) => {
    const paragraph = document.createElement("p");
    const lines = block.split(/\n/);
    lines.forEach((line, index) => {
      if (index > 0) {
        paragraph.appendChild(document.createElement("br"));
      }
      paragraph.appendChild(document.createTextNode(line));
    });
    container.appendChild(paragraph);
  });
}

function renderDiagram(diagram) {
  clearNode(diagramOutput);
  diagramMeta.textContent = "No diagram loaded.";

  if (!diagram || !diagram.svg) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No diagram in this response.";
    diagramOutput.appendChild(empty);
    return;
  }

  diagramMeta.textContent = [
    diagram.type || "svg",
    diagram.description || "",
    diagram.spec && diagram.spec.kind ? `kind: ${diagram.spec.kind}` : ""
  ]
    .filter(Boolean)
    .join(" / ");

  const iframe = document.createElement("iframe");
  iframe.setAttribute("title", diagram.description || "Generated diagram");
  iframe.setAttribute("sandbox", "");
  iframe.srcdoc = `<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;display:grid;place-items:center;min-height:100vh;background:#fff}svg{max-width:100%;height:auto}</style></head><body>${diagram.svg}</body></html>`;
  diagramOutput.appendChild(iframe);
}

async function typeset() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    await window.MathJax.typesetPromise([problemOutput, solutionOutput]);
  }
}

async function renderFromInput() {
  let data;
  try {
    data = JSON.parse(jsonInput.value);
  } catch (error) {
    setStatus(`Invalid JSON: ${error.message}`, "error");
    return;
  }

  if (data.status && data.status !== "success") {
    setStatus(`Response status is "${data.status}". Rendering available fields.`, "error");
  } else {
    setStatus("Rendered successfully.", "ok");
  }

  appendTextWithBreaks(problemOutput, data.problem);
  appendTextWithBreaks(solutionOutput, data.solution);
  renderDiagram(data.diagram);

  try {
    await typeset();
  } catch (error) {
    setStatus(`Rendered text, but LaTeX typesetting failed: ${error.message}`, "error");
  }
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    return;
  }
  jsonInput.value = await file.text();
  renderFromInput();
});

sampleButton.addEventListener("click", () => {
  jsonInput.value = JSON.stringify(sampleResponse, null, 2);
  renderFromInput();
});

renderButton.addEventListener("click", renderFromInput);

jsonInput.value = JSON.stringify(sampleResponse, null, 2);
renderFromInput();
