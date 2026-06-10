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
const apiKeyInput = document.getElementById("apiKeyInput");
const sampleButton = document.getElementById("sampleButton");
const generateButton = document.getElementById("generateButton");
const statusMessage = document.getElementById("statusMessage");
const problemOutput = document.getElementById("problemOutput");
const solutionOutput = document.getElementById("solutionOutput");
const rawOutput = document.getElementById("rawOutput");

function defaultApiUrl() {
  const configuredApiUrl = window.GEN_STUDY_CONFIG && window.GEN_STUDY_CONFIG.apiUrl;
  if (configuredApiUrl) {
    return configuredApiUrl;
  }
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return `${window.location.origin}/generate`;
  }
  return "http://127.0.0.1:8000/generate";
}

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`.trim();
}

function setLoading(isLoading) {
  generateButton.disabled = isLoading;
  generateButton.textContent = isLoading ? "Generating..." : "Generate";
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

async function typeset() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    await window.MathJax.typesetPromise([problemOutput, solutionOutput]);
  }
}

async function renderResponse(data) {
  rawOutput.textContent = JSON.stringify(data, null, 2);

  if (data.status === "success") {
    appendRichText(problemOutput, data.problem);
    appendRichText(solutionOutput, data.solution);
    setStatus("Generation completed.", "ok");
  } else {
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
  return {
    subject: subjectInput.value,
    level: levelInput.value.trim(),
    topic: topicInput.value.trim()
  };
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

  try {
    const response = await fetch(apiUrlInput.value.trim(), {
      method: "POST",
      headers,
      body: JSON.stringify(payload)
    });
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error(`API returned non-JSON response: ${text.slice(0, 160)}`);
    }
    if (!response.ok && data.status !== "failed") {
      await renderResponse(data);
      return;
    }
    await renderResponse(data);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setLoading(false);
  }
}

function loadSample() {
  renderResponse(sampleResponse);
}

apiUrlInput.value = defaultApiUrl();
form.addEventListener("submit", generateProblem);
sampleButton.addEventListener("click", loadSample);
loadSample();
