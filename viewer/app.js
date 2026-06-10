const sampleResponse = {
  status: "success",
  problem: "二次関数 $y=x^2-2x$ の最小値を求めよ。",
  solution: "平方完成すると $y=(x-1)^2-1$ である。よって最小値は $-1$ である。"
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

async function typeset() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    await window.MathJax.typesetPromise([problemOutput, solutionOutput]);
  }
}

async function renderResponse(data) {
  rawOutput.textContent = JSON.stringify(data, null, 2);

  if (data.status === "success") {
    appendTextWithBreaks(problemOutput, data.problem);
    appendTextWithBreaks(solutionOutput, data.solution);
    setStatus("Generation completed.", "ok");
  } else {
    const detail = data.detail || data.reason || data.error || "Generation failed.";
    appendTextWithBreaks(problemOutput, data.problem);
    appendTextWithBreaks(solutionOutput, data.solution);
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
