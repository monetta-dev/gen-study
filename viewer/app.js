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
const sampleButton = document.getElementById("sampleButton");
const visualButton = document.getElementById("visualButton");
const generateButton = document.getElementById("generateButton");
const statusMessage = document.getElementById("statusMessage");
const problemOutput = document.getElementById("problemOutput");
const solutionOutput = document.getElementById("solutionOutput");
const visualOutput = document.getElementById("visualOutput");
const processOutput = document.getElementById("processOutput");
const rawOutput = document.getElementById("rawOutput");
const processSection = document.querySelector(".process-section");
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

function streamUrlFor(apiUrl) {
  if (apiUrl.endsWith("/generate")) {
    return `${apiUrl}/stream`;
  }
  if (apiUrl.endsWith("/visuals/generate")) {
    return `${apiUrl}/stream`;
  }
  return "";
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
  const mathPattern = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$[^$]*?\$)/g;
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
  const commandPattern = /\\(textbf|emph)\{([^{}]*)\}/g;
  let cursor = 0;
  let match;

  while ((match = commandPattern.exec(value)) !== null) {
    appendPlainTextWithLineBreaks(container, value.slice(cursor, match.index));
    const element = document.createElement(match[1] === "textbf" ? "strong" : "em");
    element.textContent = match[2];
    container.appendChild(element);
    cursor = match.index + match[0].length;
  }

  appendPlainTextWithLineBreaks(container, value.slice(cursor));
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

    const source = document.createElement("pre");
    source.className = "tikz-source";
    source.textContent = asset.tikz_source || "No TikZ source was returned.";
    figure.appendChild(source);

    const caption = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = asset.id || `visual_${index + 1}`;
    caption.appendChild(title);
    caption.appendChild(document.createTextNode(` · ${asset.placement || "unknown"} · ${asset.purpose || "visual"}`));
    figure.appendChild(caption);

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
    renderTraceRecords(data.trace, "Problem Generation", false);
    visualButton.disabled = false;
    setStatus("Generation completed.", "ok");
  } else {
    currentTextResponse = null;
    visualButton.disabled = true;
    const detail = data.detail || data.reason || data.error || "Generation failed.";
    appendRichText(problemOutput, data.problem);
    appendRichText(solutionOutput, data.solution);
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
  resetProcessOutput("Generating...");

  try {
    const data = await fetchJsonOrStream(
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
    const data = await fetchJsonOrStream(
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

async function fetchJsonOrStream(apiUrl, headers, payload, label) {
  if (configFlag("streamProgress", true)) {
    const streamUrl = streamUrlFor(apiUrl);
    if (streamUrl && window.ReadableStream) {
      try {
        return await fetchStreamedResult(streamUrl, headers, payload, label);
      } catch (error) {
        setStatus(`${error.message} Retrying without progress stream.`, "error");
      }
    }
  }
  return fetchJsonResult(apiUrl, headers, payload, label);
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
    throw new Error(`${label} returned non-JSON response: ${text.slice(0, 160)}`);
  }
  return data;
}

async function fetchStreamedResult(apiUrl, headers, payload, label) {
  const response = await fetch(apiUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(payload)
  });
  if (!response.body) {
    throw new Error(`${label} progress stream is unavailable.`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\n/);
    buffer = lines.pop() || "";
    lines.forEach((line) => {
      const event = parseStreamEvent(line, label);
      if (!event) {
        return;
      }
      if (event.type === "progress" && event.message) {
        setStatus(event.message, "");
      }
      if (event.type === "result") {
        finalData = event.data;
      }
    });
  }

  buffer += decoder.decode();
  const trailingEvent = parseStreamEvent(buffer, label);
  if (trailingEvent && trailingEvent.type === "result") {
    finalData = trailingEvent.data;
  }
  if (!finalData) {
    throw new Error(`${label} progress stream ended without a final result.`);
  }
  return finalData;
}

function parseStreamEvent(line, label) {
  const text = String(line || "").trim();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} returned an invalid progress event: ${text.slice(0, 120)}`);
  }
}

function loadSample() {
  currentTextResponse = sampleResponse;
  resetVisualOutput();
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
initializeRequestDefaults();
form.addEventListener("submit", generateProblem);
subjectInput.addEventListener("change", syncTopicForSubject);
sampleButton.addEventListener("click", loadSample);
visualButton.addEventListener("click", generateVisuals);
loadSample();
