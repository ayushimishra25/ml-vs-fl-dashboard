const state = {
  lastEventId: 0,
  polling: null,
  classNames: [],
  classDisplayNames: {},
};

const elements = {
  statusPill: document.getElementById("status-pill"),
  eventCounter: document.getElementById("event-counter"),
  mlAccuracy: document.getElementById("ml-accuracy"),
  flAccuracy: document.getElementById("fl-accuracy"),
  agreementRate: document.getElementById("agreement-rate"),
  classesSeen: document.getElementById("classes-seen"),
  truthLabel: document.getElementById("truth-label"),
  mlLabel: document.getElementById("ml-label"),
  flLabel: document.getElementById("fl-label"),
  agreeLabel: document.getElementById("agree-label"),
  currentTime: document.getElementById("current-time"),
  mlBar: document.getElementById("ml-bar"),
  flBar: document.getElementById("fl-bar"),
  featureList: document.getElementById("feature-list"),
  distributionList: document.getElementById("class-distribution"),
  streamBody: document.getElementById("stream-body"),
  trendChart: document.getElementById("trend-chart"),
  startButton: document.getElementById("start-button"),
  stopButton: document.getElementById("stop-button"),
  mode: document.getElementById("mode"),
  samplingStrategy: document.getElementById("sampling-strategy"),
  focusClass: document.getElementById("focus-class"),
  events: document.getElementById("events"),
  interval: document.getElementById("interval"),
  noiseScale: document.getElementById("noise-scale"),
  seed: document.getElementById("seed"),
};

function setStatus(running) {
  elements.statusPill.textContent = running ? "Streaming" : "Idle";
  elements.statusPill.className = `pill ${running ? "live" : "idle"}`;
}

function updateClassOptions(classNames, classDisplayNames) {
  if (
    JSON.stringify(classNames) === JSON.stringify(state.classNames) &&
    JSON.stringify(classDisplayNames) === JSON.stringify(state.classDisplayNames)
  ) {
    return;
  }

  state.classNames = classNames;
  state.classDisplayNames = classDisplayNames;

  const previous = elements.focusClass.value || "all";
  elements.focusClass.innerHTML = "";

  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All Classes";
  elements.focusClass.appendChild(allOption);

  classNames.forEach((label) => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = classDisplayNames[label] || label;
    elements.focusClass.appendChild(option);
  });

  elements.focusClass.value = classNames.includes(previous) || previous === "all" ? previous : "all";
}

function updateMetrics(stats) {
  elements.eventCounter.textContent = `${stats.total_events || 0} events`;
  elements.mlAccuracy.textContent = (stats.ml_accuracy || 0).toFixed(4);
  elements.flAccuracy.textContent = (stats.fl_accuracy || 0).toFixed(4);
  elements.agreementRate.textContent = (stats.agreement_rate || 0).toFixed(4);
  elements.classesSeen.textContent = `${stats.classes_seen || 0} / ${stats.class_count || 0}`;

  const counts = stats.true_counts || {};
  elements.distributionList.innerHTML = "";
  state.classNames.forEach((label) => {
    const chip = document.createElement("div");
    chip.className = "distribution-chip";
    chip.innerHTML = `
      <span>${state.classDisplayNames[label] || label}</span>
      <strong>${counts[label] || 0}</strong>
    `;
    elements.distributionList.appendChild(chip);
  });
}

function renderCurrent(event, previewFields) {
  if (!event) {
    elements.currentTime.textContent = "Waiting for stream";
    elements.truthLabel.textContent = "-";
    elements.mlLabel.textContent = "-";
    elements.flLabel.textContent = "-";
    elements.agreeLabel.textContent = "-";
    elements.mlBar.style.width = "0%";
    elements.flBar.style.width = "0%";
    elements.featureList.innerHTML = "";
    return;
  }

  elements.currentTime.textContent = event.timestamp;
  elements.truthLabel.textContent = event.true_label_name;
  elements.mlLabel.textContent = `${event.ml_label_name} (${event.ml_confidence.toFixed(3)})`;
  elements.flLabel.textContent = `${event.fl_label_name} (${event.fl_confidence.toFixed(3)})`;
  elements.agreeLabel.textContent = event.agree ? "yes" : "no";
  elements.mlBar.style.width = `${event.ml_confidence * 100}%`;
  elements.flBar.style.width = `${event.fl_confidence * 100}%`;

  elements.featureList.innerHTML = "";
  previewFields.forEach((field) => {
    const wrapper = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = field;
    detail.textContent = event.preview[field];
    wrapper.appendChild(term);
    wrapper.appendChild(detail);
    elements.featureList.appendChild(wrapper);
  });
}

function prependEvents(events) {
  if (!events.length) {
    return;
  }

  const fragment = document.createDocumentFragment();
  const reversed = [...events].reverse();

  reversed.forEach((event) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${event.timestamp}</td>
      <td>${event.true_label_name}</td>
      <td>${event.ml_label_name} (${event.ml_confidence.toFixed(3)})</td>
      <td>${event.fl_label_name} (${event.fl_confidence.toFixed(3)})</td>
      <td class="${event.agree ? "tag-yes" : "tag-no"}">${event.agree ? "yes" : "no"}</td>
    `;
    fragment.appendChild(row);
  });

  elements.streamBody.prepend(fragment);
  while (elements.streamBody.children.length > 300) {
    elements.streamBody.removeChild(elements.streamBody.lastChild);
  }
}

function drawTrendChart(points) {
  const canvas = elements.trendChart;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fcfbf8";
  ctx.fillRect(0, 0, width, height);

  const pad = { top: 16, right: 16, bottom: 24, left: 28 };
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;

  ctx.strokeStyle = "#d7d0c2";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  [0, 0.25, 0.5, 0.75, 1].forEach((ratio) => {
    const y = pad.top + innerHeight * ratio;
    ctx.strokeStyle = "#ece6d8";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  });

  if (!points.length) {
    ctx.fillStyle = "#7a8076";
    ctx.font = "13px SF Pro Text, Segoe UI, sans-serif";
    ctx.fillText("Start a stream to see confusion trend", pad.left + 8, pad.top + 28);
    return;
  }

  const maxEvent = Math.max(...points.map((point) => point.event_id), 1);
  const xAt = (eventId) => pad.left + (innerWidth * (eventId - 1)) / Math.max(maxEvent - 1, 1);
  const yAt = (value) => pad.top + innerHeight * Math.min(Math.max(value, 0), 1);

  const drawLine = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xAt(point.event_id);
      const y = yAt(point[key]);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  };

  drawLine("ml_confusion_rate", "#2e7d5d");
  drawLine("fl_confusion_rate", "#8a6432");
}

async function fetchState() {
  const response = await fetch(`/api/state?after=${state.lastEventId}`);
  const payload = await response.json();
  setStatus(payload.running);
  updateClassOptions(payload.class_names, payload.class_display_names);
  updateMetrics(payload.stats);
  renderCurrent(payload.current_event, payload.preview_fields);
  drawTrendChart(payload.trend_points || []);
  prependEvents(payload.events);

  if (payload.events.length) {
    state.lastEventId = payload.events[payload.events.length - 1].event_id;
  }
}

async function startStream() {
  state.lastEventId = 0;
  elements.streamBody.innerHTML = "";

  const response = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: elements.mode.value,
      sampling_strategy: elements.samplingStrategy.value,
      focus_class: elements.focusClass.value,
      events: Number(elements.events.value),
      interval: Number(elements.interval.value),
      noise_scale: Number(elements.noiseScale.value),
      seed: Number(elements.seed.value),
    }),
  });

  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || "Could not start the stream.");
  }

  await fetchState();
}

async function stopStream() {
  await fetch("/api/stop", { method: "POST" });
  await fetchState();
}

elements.startButton.addEventListener("click", async () => {
  elements.startButton.disabled = true;
  try {
    await startStream();
  } catch (error) {
    window.alert(error.message);
  } finally {
    elements.startButton.disabled = false;
  }
});

elements.stopButton.addEventListener("click", async () => {
  await stopStream();
});

state.polling = window.setInterval(() => {
  fetchState().catch(() => {});
}, 500);

fetchState().catch(() => {});
