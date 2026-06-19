const sampleResponse = {
  status: "success",
  problem: "二次関数 $y=x^2-2x$ の最小値を求めよ。",
  solution:
    "\\textbf{(1)} 平方完成すると $y=(x-1)^2-1$ である。\n\n\\begin{enumerate}\n\\item 頂点は $(1,-1)$ である。\n\\item よって最小値は $-1$ である。\n\\end{enumerate}"
};

const form = document.getElementById("generateForm");
const subjectInput = document.getElementById("subjectInput");
const levelInput = document.getElementById("levelInput");
const topicInput = document.getElementById("topicInput");
const apiUrlInput = document.getElementById("apiUrlInput");
const visualApiUrlInput = document.getElementById("visualApiUrlInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const autoVisualInput = document.getElementById("autoVisualInput");
const renderOutputInput = document.getElementById("renderOutputInput");
const sampleButton = document.getElementById("sampleButton");
const visualButton = document.getElementById("visualButton");
const generateButton = document.getElementById("generateButton");
const statusMessage = document.getElementById("statusMessage");
const problemOutput = document.getElementById("problemOutput");
const solutionOutput = document.getElementById("solutionOutput");
const visualOutput = document.getElementById("visualOutput");
const renderedOutput = document.getElementById("renderedOutput");
const processOutput = document.getElementById("processOutput");
const rawOutput = document.getElementById("rawOutput");
const processSection = document.querySelector(".process-section");
const renderToggle = renderOutputInput ? renderOutputInput.closest(".visual-toggle") : null;
const renderSection = document.querySelector(".render-section");
let currentTextResponse = null;
let currentVisualResponse = null;

function appConfig() {
  return window.GEN_STUDY_CONFIG || {};
}

function configFlag(name, defaultValue = false) {
  const value = appConfig()[name];
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
  }
  return defaultValue;
}

const defaultTopicsBySubject = {
  mathematics: "Probability Recurrence Relations",
  physics: "Mechanics",
  chemistry: "Acid-Base Titration"
};

const topicOptionsBySubject = {
  mathematics: [
    "Probability Recurrence Relations",
    "Integer Problems",
    "Calculus",
    "Coordinate Geometry"
  ],
  physics: [
    "Mechanics",
    "Electromagnetism",
    "Waves",
    "Thermodynamics"
  ],
  chemistry: [
    "Acid-Base Titration",
    "Chemical Equilibrium",
    "Organic Chemistry",
    "Oxidation-Reduction"
  ]
};

function defaultApiUrl() {
  const configuredApiUrl = appConfig().apiUrl;
  if (configuredApiUrl) {
    return configuredApiUrl;
  }
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return `${window.location.origin}/generate`;
  }
  return "http://127.0.0.1:8000/generate";
}

function defaultVisualApiUrl() {
  const configuredApiUrl = appConfig().visualApiUrl;
  if (configuredApiUrl) {
    return configuredApiUrl;
  }
  const apiUrl = defaultApiUrl();
  if (apiUrl.endsWith("/generate")) {
    return `${apiUrl.slice(0, -"/generate".length)}/visuals/generate`;
  }
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return `${window.location.origin}/visuals/generate`;
  }
  return "http://127.0.0.1:8000/visuals/generate";
}

function setTopicOptions(subject) {
  const topicOptions = document.getElementById("topicOptions");
  clearNode(topicOptions);
  topicOptionsBySubject[subject].forEach((topic) => {
    const option = document.createElement("option");
    option.value = topic;
    topicOptions.appendChild(option);
  });
}

function initializeRequestDefaults() {
  const subject = subjectInput.value || "mathematics";
  setTopicOptions(subject);
  topicInput.value = defaultTopicsBySubject[subject];
}

function syncTopicForSubject() {
  const subject = subjectInput.value;
  setTopicOptions(subject);
  topicInput.value = defaultTopicsBySubject[subject];
}

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`.trim();
}

function setLoading(isLoading) {
  generateButton.disabled = isLoading;
  visualButton.disabled = isLoading || !currentTextResponse;
  sampleButton.disabled = isLoading;
  generateButton.textContent = isLoading ? "Generating..." : "Generate";
}

function setVisualLoading(isLoading) {
  generateButton.disabled = isLoading;
  sampleButton.disabled = isLoading;
  visualButton.disabled = isLoading || !currentTextResponse;
  visualButton.textContent = isLoading ? "Generating Visuals..." : "Generate Visuals";
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function normalizeTextLatex(value) {
  return String(value || "")
    .replace(/(?:\$\$|\\\[)\s*(\\begin\{(enumerate|itemize)\}[\s\S]*?\\end\{\2\})\s*(?:\$\$|\\\])/g, "$1")
    .replace(/\\\((\\begin\{(enumerate|itemize)\}[\s\S]*?\\end\{\2\})\\\)/g, "$1");
}

function appendInlineLatex(container, value) {
  const text = String(value || "");
  // Math spans:
  //  - $...$  (display, may span newlines)
  //  - \[...\]  (display, may span newlines)
  //  - \(...\)  (inline, may span newlines)
  //  - $...$    (inline, single line only)
  // The inline $...$ branch uses a negative lookbehind on both delimiters so
  // that an escaped "\$" (e.g. "\$100") is never treated as a math boundary,
  // and an escaped char inside (\.) is consumed so math containing \$ or \\
  // does not split mid-expression. Newlines are disallowed inside $...$ so
  // two separate inline expressions on different lines are not merged.
  const mathPattern = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<!\\)\$(?:[^$\\\n]|\\.)*?(?<!\\)\$)/g;
  let cursor = 0;
  let match;

  while ((match = mathPattern.exec(text)) !== null) {
    appendTextCommands(container, text.slice(cursor, match.index));
    container.appendChild(document.createTextNode(match[0]));
    cursor = match.index + match[0].length;
  }

  appendTextCommands(container, text.slice(cursor));
}

function appendTextCommands(container, value) {
  const text = String(value || "");
  // Match \textbf{...} / \emph{...} with balanced, nestable braces so that
  // e.g. "\textbf{a \emph{b} c}" renders correctly. The previous regex
  // (\\(textbf|emph)\{([^{}]*)\}) stopped at the first inner brace and
  // truncated the command. A brace-counting scanner is used instead: it
  // honors escaped braces (\{, \}) and recurses into the captured body so
  // nested \textbf/\emph are styled too.
  const commandStart = /\\(textbf|emph)\{/g;
  let cursor = 0;
  let match;

  while ((match = commandStart.exec(text)) !== null) {
    const name = match[1];
    const bodyStart = match.index + match[0].length;
    const bodyEnd = findMatchingBrace(text, bodyStart);
    if (bodyEnd === -1) {
      // Unbalanced: leave the rest untouched rather than emitting a broken
      // command. MathJax will still typeset any math that follows.
      continue;
    }
    appendPlainTextWithLineBreaks(container, text.slice(cursor, match.index));
    const element = document.createElement(name === "textbf" ? "strong" : "em");
    appendTextCommands(element, text.slice(bodyStart, bodyEnd));
    container.appendChild(element);
    cursor = bodyEnd + 1;
    commandStart.lastIndex = cursor;
  }

  appendPlainTextWithLineBreaks(container, text.slice(cursor));
}

// Given text and an index pointing just after an opening "{", return the index
// of the matching "}", honoring escaped braces (\{, \}) and nested braces.
// Returns -1 if no matching brace is found.
function findMatchingBrace(text, afterOpenIndex) {
  let depth = 1;
  let i = afterOpenIndex;
  while (i < text.length) {
    const ch = text[i];
    if (ch === "\\") {
      i += 2; // skip escaped char (\{, \}, \\, etc.)
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
    i += 1;
  }
  return -1;
}

function appendPlainTextWithLineBreaks(container, value) {
  const lines = String(value || "").split(/\n/);
  lines.forEach((line, index) => {
    if (index > 0) {
      container.appendChild(document.createElement("br"));
    }
    if (line) {
      container.appendChild(document.createTextNode(line));
    }
  });
}

function appendParagraph(container, value) {
  const text = String(value || "").trim();
  if (!text) {
    return;
  }
  const paragraph = document.createElement("p");
  appendInlineLatex(paragraph, text);
  container.appendChild(paragraph);
}

function appendList(container, kind, value) {
  const list = document.createElement(kind === "enumerate" ? "ol" : "ul");
  list.className = "latex-list";
  const items = value.split(/\\item\b/).map((item) => item.trim()).filter(Boolean);

  items.forEach((item) => {
    const listItem = document.createElement("li");
    appendInlineLatex(listItem, item);
    list.appendChild(listItem);
  });

  if (list.children.length > 0) {
    container.appendChild(list);
  }
}

function appendRichText(container, value) {
  clearNode(container);
  container.classList.remove("empty");

  const text = normalizeTextLatex(value).trim();
  if (!text) {
    container.textContent = "No content loaded.";
    container.classList.add("empty");
    return;
  }

  const environmentPattern = /\\begin\{(enumerate|itemize)\}([\s\S]*?)\\end\{\1\}/g;
  let cursor = 0;
  let match;

  while ((match = environmentPattern.exec(text)) !== null) {
    text.slice(cursor, match.index).split(/\n{2,}/).forEach((block) => {
      appendParagraph(container, block);
    });
    appendList(container, match[1], match[2]);
    cursor = match.index + match[0].length;
  }

  text.slice(cursor).split(/\n{2,}/).forEach((block) => {
    appendParagraph(container, block);
  });
}

function resetVisualOutput(message = "No visuals loaded.") {
  clearNode(visualOutput);
  visualOutput.textContent = message;
  visualOutput.classList.add("empty");
  currentVisualResponse = null;
}

function resetRenderedOutput(message = "No rendered PNG loaded.") {
  clearNode(renderedOutput);
  renderedOutput.textContent = message;
  renderedOutput.classList.add("empty");
}

function resetProcessOutput(message = "No process loaded.") {
  if (!configFlag("showProcess")) {
    return;
  }
  clearNode(processOutput);
  processOutput.textContent = message;
  processOutput.classList.add("empty");
}

function renderTraceRecords(records, title, append = true) {
  if (!configFlag("showProcess")) {
    return;
  }
  if (!append) {
    clearNode(processOutput);
  }
  if (!Array.isArray(records) || records.length === 0) {
    if (!append || processOutput.classList.contains("empty")) {
      processOutput.textContent = "No process trace was returned.";
      processOutput.classList.add("empty");
    }
    return;
  }

  if (processOutput.classList.contains("empty")) {
    clearNode(processOutput);
    processOutput.classList.remove("empty");
  }

  const group = document.createElement("details");
  group.className = "process-group";
  group.open = true;
  const summary = document.createElement("summary");
  summary.textContent = title;
  group.appendChild(summary);

  records.forEach((record, index) => {
    const item = document.createElement("details");
    item.className = "process-item";
    const itemSummary = document.createElement("summary");
    itemSummary.textContent = traceTitle(record, index + 1);
    const body = document.createElement("pre");
    body.textContent = record;
    item.appendChild(itemSummary);
    item.appendChild(body);
    group.appendChild(item);
  });

  processOutput.appendChild(group);
}

function traceTitle(record, index) {
  const firstLine = String(record || "").trim().split("\n")[0];
  return firstLine.replace(/^[-=\s]+|[-=\s]+$/g, "") || `Step ${index}`;
}

function renderVisualResponse(data) {
  currentVisualResponse = data;
  clearNode(visualOutput);
  visualOutput.classList.remove("empty");

  if (data.status !== "success") {
    visualOutput.textContent = data.detail || data.reason || data.error || "Visual generation failed.";
    visualOutput.classList.add("empty");
    return;
  }

  if (!data.visual_needed) {
    visualOutput.textContent = "No visual was needed for this problem.";
    visualOutput.classList.add("empty");
    return;
  }

  const assets = Array.isArray(data.visual_assets) ? data.visual_assets : [];
  if (assets.length === 0) {
    visualOutput.textContent = "No visual asset was returned.";
    visualOutput.classList.add("empty");
    return;
  }

  const status = document.createElement("div");
  status.className = `visual-status ${data.visual_validation_status || ""}`.trim();
  status.textContent = `Validation: ${data.visual_validation_status || "unknown"}`;
  visualOutput.appendChild(status);

  const grid = document.createElement("div");
  grid.className = "visual-grid";
  assets.forEach((asset, index) => {
    const figure = document.createElement("figure");
    figure.className = "visual-figure";

    if (asset.png_base64) {
      const image = document.createElement("img");
      image.className = "visual-image";
      image.alt = asset.id || `visual_${index + 1}`;
      image.src = `data:${asset.mime_type || "image/png"};base64,${asset.png_base64}`;
      figure.appendChild(image);
    }

    const caption = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = asset.id || `visual_${index + 1}`;
    caption.appendChild(title);
    caption.appendChild(document.createTextNode(` · ${asset.placement || "unknown"} · ${asset.purpose || "visual"}`));
    figure.appendChild(caption);

    const sourceDetails = document.createElement("details");
    sourceDetails.className = "source-details";
    const sourceSummary = document.createElement("summary");
    sourceSummary.textContent = "TikZ";
    const source = document.createElement("pre");
    source.className = "tikz-source";
    source.textContent = asset.tikz_source || "No TikZ source was returned.";
    sourceDetails.appendChild(sourceSummary);
    sourceDetails.appendChild(source);
    figure.appendChild(sourceDetails);

    if (Array.isArray(asset.compile_transcript) && asset.compile_transcript.length > 0) {
      const compileDetails = document.createElement("details");
      compileDetails.className = "compile-details";
      const compileSummary = document.createElement("summary");
      compileSummary.textContent = `Compile: ${asset.compile_status || "unknown"}`;
      const compileLog = document.createElement("pre");
      compileLog.textContent = JSON.stringify(asset.compile_transcript, null, 2);
      compileDetails.appendChild(compileSummary);
      compileDetails.appendChild(compileLog);
      figure.appendChild(compileDetails);
    }

    if (asset.prompt) {
      const details = document.createElement("details");
      details.className = "prompt-details";
      const summary = document.createElement("summary");
      summary.textContent = "Prompt";
      const prompt = document.createElement("p");
      prompt.textContent = asset.prompt;
      details.appendChild(summary);
      details.appendChild(prompt);
      figure.appendChild(details);
    }

    grid.appendChild(figure);
  });
  visualOutput.appendChild(grid);
}

function renderProblemImages(data) {
  clearNode(renderedOutput);
  renderedOutput.classList.remove("empty");

  const output = data.rendered_output;
  const assets = output && Array.isArray(output.assets) ? output.assets : [];
  if (assets.length === 0) {
    renderedOutput.textContent = "No rendered PNG was returned.";
    renderedOutput.classList.add("empty");
    return;
  }

  const grid = document.createElement("div");
  grid.className = "rendered-grid";
  assets.forEach((asset, index) => {
    const figure = document.createElement("figure");
    figure.className = "rendered-figure";

    const image = document.createElement("img");
    image.className = "rendered-image";
    image.alt = `${asset.title || asset.kind || "rendered"} page ${asset.page || index + 1}`;
    image.src = `data:${asset.mime_type || "image/png"};base64,${asset.png_base64}`;
    figure.appendChild(image);

    const caption = document.createElement("figcaption");
    caption.textContent = `${asset.title || asset.kind || "Rendered"} · page ${asset.page || index + 1}`;
    figure.appendChild(caption);
    grid.appendChild(figure);
  });

  renderedOutput.appendChild(grid);
}

async function typeset() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    await window.MathJax.typesetPromise([problemOutput, solutionOutput]);
  }
}

async function renderResponse(data) {
  renderRawResponse(data);

  if (data.status === "success") {
    currentTextResponse = data;
    appendRichText(problemOutput, data.problem);
    appendRichText(solutionOutput, data.solution);
    if (data.rendered_output && data.rendered_output.status === "compiled") {
      renderProblemImages(data);
    } else {
      resetRenderedOutput();
    }
    renderTraceRecords(data.trace, "Problem Generation", false);
    visualButton.disabled = false;
    setStatus("Generation completed.", "ok");
  } else {
    currentTextResponse = null;
    visualButton.disabled = true;
    const detail = data.detail || data.reason || data.error || "Generation failed.";
    appendRichText(problemOutput, data.problem);
    appendRichText(solutionOutput, data.solution);
    resetRenderedOutput();
    setStatus(detail, "error");
  }

  try {
    await typeset();
  } catch (error) {
    setStatus(`Rendered text, but LaTeX typesetting failed: ${error.message}`, "error");
  }
}

function requestPayload() {
  const payload = {
    subject: subjectInput.value,
    level: levelInput.value.trim(),
    topic: topicInput.value.trim()
  };
  if (configFlag("enableRenderOutput") && renderOutputInput && renderOutputInput.checked) {
    payload.render_output = true;
  }
  if (configFlag("includeTrace")) {
    payload.trace = true;
  }
  return payload;
}

async function generateProblem(event) {
  event.preventDefault();

  const payload = requestPayload();
  if (!payload.level || !payload.topic) {
    setStatus("Level and topic are required.", "error");
    return;
  }

  const headers = {
    "Content-Type": "application/json"
  };
  const apiKey = apiKeyInput.value.trim();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  setLoading(true);
  setStatus("Generating. This can take several minutes.", "");
  resetVisualOutput();
  resetRenderedOutput(renderOutputInput.checked ? "Rendering PNG..." : "No rendered PNG loaded.");
  resetProcessOutput("Generating...");

  try {
    const data = await fetchJsonResult(
      apiUrlInput.value.trim(),
      headers,
      payload,
      "API"
    );
    await renderResponse(data);
    if (data.status === "success" && autoVisualInput.checked) {
      await generateVisuals();
    }
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setLoading(false);
  }
}

function visualPayload() {
  if (!currentTextResponse) {
    return null;
  }
  const payload = {
    subject: subjectInput.value,
    level: levelInput.value.trim(),
    topic: topicInput.value.trim(),
    problem: currentTextResponse.problem,
    solution: currentTextResponse.solution
  };
  if (configFlag("includeTrace")) {
    payload.trace = true;
  }
  return payload;
}

async function generateVisuals() {
  const payload = visualPayload();
  if (!payload) {
    setStatus("Generate a problem before generating visuals.", "error");
    return;
  }

  const headers = {
    "Content-Type": "application/json"
  };
  const apiKey = apiKeyInput.value.trim();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  setVisualLoading(true);
  setStatus("Generating visuals. This can take several minutes.", "");
  resetVisualOutput("Generating visuals...");

  try {
    const data = await fetchJsonResult(
      visualApiUrlInput.value.trim(),
      headers,
      payload,
      "Visual API"
    );
    renderVisualResponse(data);
    renderTraceRecords(data.trace, "Visual Generation", true);
    renderRawResponse(configFlag("showFullRawResponse") ? { text: currentTextResponse, visual: data } : currentTextResponse);
    if (data.status === "success") {
      setStatus("Visual generation completed.", "ok");
    } else {
      setStatus(data.detail || data.reason || data.error || "Visual generation failed.", "error");
    }
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setVisualLoading(false);
  }
}

async function fetchJsonResult(apiUrl, headers, payload, label) {
  const response = await fetch(apiUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(payload)
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (error) {
    throw new Error(formatNonJsonResponse(label, apiUrl, response, text));
  }
  return data;
}

function formatNonJsonResponse(label, apiUrl, response, text) {
  const contentType = response.headers.get("Content-Type") || "unknown content type";
  const excerpt = text ? text.slice(0, 160).replace(/\s+/g, " ").trim() : "empty response";
  const looksLikeHtml = /<\/?[a-z][\s\S]*>/i.test(text);
  const hint = looksLikeHtml
    ? " The request probably reached an HTML page, proxy error page, or static frontend instead of the JSON API."
    : "";
  return `${label} returned non-JSON response from ${apiUrl} ` +
    `(HTTP ${response.status}, ${contentType}): ${excerpt}.${hint}`;
}

function loadSample() {
  currentTextResponse = sampleResponse;
  resetVisualOutput();
  resetRenderedOutput();
  resetProcessOutput();
  renderResponse(sampleResponse);
}

function renderRawResponse(value) {
  if (configFlag("showFullRawResponse")) {
    rawOutput.textContent = JSON.stringify(redactRawResponse(value), null, 2);
    return;
  }
  rawOutput.textContent = plainTextResponse(value);
}

function plainTextResponse(value) {
  if (!value || typeof value !== "object") {
    return "";
  }
  const problem = String(value.problem || "").trim();
  const solution = String(value.solution || "").trim();
  if (!problem && !solution) {
    return "";
  }
  return `=== Problem ===\n${problem}\n\n=== Solution ===\n${solution}`;
}

function redactRawResponse(value) {
  if (Array.isArray(value)) {
    return value.map(redactRawResponse);
  }
  if (value && typeof value === "object") {
    const redacted = {};
    Object.entries(value).forEach(([key, item]) => {
      if (key === "base64" && typeof item === "string") {
        redacted[key] = `<base64 omitted: ${item.length} chars>`;
      } else {
        redacted[key] = redactRawResponse(item);
      }
    });
    return redacted;
  }
  return value;
}

apiUrlInput.value = defaultApiUrl();
visualApiUrlInput.value = defaultVisualApiUrl();
if (processSection && !configFlag("showProcess")) {
  processSection.hidden = true;
}
if (!configFlag("enableRenderOutput")) {
  if (renderToggle) {
    renderToggle.hidden = true;
  }
  if (renderOutputInput) {
    renderOutputInput.checked = false;
    renderOutputInput.disabled = true;
  }
  if (renderSection) {
    renderSection.hidden = true;
  }
}
initializeRequestDefaults();
form.addEventListener("submit", generateProblem);
subjectInput.addEventListener("change", syncTopicForSubject);
sampleButton.addEventListener("click", loadSample);
visualButton.addEventListener("click", generateVisuals);
loadSample();
