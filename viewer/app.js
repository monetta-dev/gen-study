const sampleResponse = {
  status: "success",
  problem:
    "二次関数 $y=x^2-2x$ の最小値を求めよ。",
  solution:
    "平方完成すると $y=(x-1)^2-1$ である。よって最小値は $-1$ である。"
};

const jsonInput = document.getElementById("jsonInput");
const fileInput = document.getElementById("fileInput");
const renderButton = document.getElementById("renderButton");
const sampleButton = document.getElementById("sampleButton");
const statusMessage = document.getElementById("statusMessage");
const problemOutput = document.getElementById("problemOutput");
const solutionOutput = document.getElementById("solutionOutput");

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
