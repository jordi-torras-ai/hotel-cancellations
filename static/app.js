/* Plain browser JavaScript: no build step, CDN or external service. */
"use strict";
const report = JSON.parse(document.getElementById("initial-report").textContent);
const $ = (selector) => document.querySelector(selector);
const number = (value, decimals = 0) => new Intl.NumberFormat("es-ES", {maximumFractionDigits: decimals, minimumFractionDigits: decimals}).format(value);
const percentage = (value, decimals = 1) => `${number(value * 100, decimals)}%`;
const money = (value) => new Intl.NumberFormat("es-ES", {style: "currency", currency: "EUR", maximumFractionDigits: 2}).format(value);
const textSize = $("#text-size");
textSize.value = document.documentElement.dataset.textSize || "normal";
textSize.addEventListener("change", () => {
  document.documentElement.dataset.textSize = textSize.value;
  try { localStorage.setItem("inn-text-size", textSize.value); } catch (_) {}
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
async function checked(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `La solicitud no se ha completado (${response.status}).`);
  }
  return response;
}
function showMessage(selector, message) {
  const node = $(selector);
  node.textContent = message;
  node.hidden = !message;
}
function horizontalChart(selector, items, max = 1) {
  const target = $(selector);
  if (!target) return;
  for (const item of items) {
    const row = element("div", "horizontal-row");
    row.append(element("span", "bar-label", item.label));
    const track = element("div", "bar-track");
    const fill = element("div", "bar-fill");
    fill.style.width = `${Math.min(100, item.value / max * 100)}%`;
    track.append(fill);
    row.append(track, element("strong", "", percentage(item.value)));
    if (item.bookings !== undefined) row.title = `${number(item.bookings)} reservas; ${percentage(item.value)} canceladas`;
    target.append(row);
  }
}
function monthlyChart(items) {
  const target = $("#monthly-chart");
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  const chartWidth = Math.max(36 * parseFloat(getComputedStyle(document.documentElement).fontSize), target.clientWidth);
  svg.setAttribute("viewBox", `0 0 ${chartWidth} 300`);
  svg.style.width = `${chartWidth}px`;
  const left = 62, top = 20, height = 230, width = chartWidth - 80;
  const max = Math.max(4, Math.ceil(Math.max(...items.map(item => item.bookings)) / 1000) * 1000);
  function draw(tag, attrs, text) {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  }
  for (let step = 0; step <= 4; step++) {
    const value = max * step / 4;
    const y = top + height - height * step / 4;
    draw("line", {x1: left, x2: left + width, y1: y, y2: y, class: "chart-grid"});
    draw("text", {x: left - 10, y: y + 3, "text-anchor": "end", class: "chart-text"}, value >= 1000 ? `${number(value / 1000, value % 1000 ? 1 : 0)}k` : String(value));
  }
  items.forEach((item, index) => {
    const x = left + index * width / 12 + 12;
    const barWidth = width / 12 - 23;
    const keptHeight = (item.bookings - item.cancelled) / max * height;
    const cancelledHeight = item.cancelled / max * height;
    const group = draw("g", {});
    const title = document.createElementNS(svgNS, "title");
    title.textContent = `${item.label}: ${number(item.bookings)} reservas, ${number(item.cancelled)} canceladas (${percentage(item.rate)})`;
    group.append(title);
    for (const [y, h, color] of [[top + height - keptHeight, keptHeight, "#6f8965"], [top + height - keptHeight - cancelledHeight, cancelledHeight, "#e1c8a3"]]) {
      const rect = document.createElementNS(svgNS, "rect");
      for (const [key, value] of Object.entries({x, y, width: barWidth, height: h, rx: 2, fill: color})) rect.setAttribute(key, value);
      group.append(rect);
    }
    draw("text", {x: x + barWidth / 2, y: 282, "text-anchor": "middle", class: "chart-text"}, item.label);
  });
  target.append(svg);
  target.setAttribute("aria-label", items.map(item => `${item.label}: ${item.bookings} reservas, ${item.cancelled} canceladas`).join(". "));
}

if (document.body.dataset.page === "dashboard") {
  monthlyChart(report.summary.monthly);
  const redrawMonthly = () => {
    $("#monthly-chart").replaceChildren();
    monthlyChart(report.summary.monthly);
  };
  window.addEventListener("resize", redrawMonthly);
  textSize.addEventListener("change", redrawMonthly);
  horizontalChart("#lead-chart", report.summary.lead_time.map(item => ({...item, value: item.rate})), 1);
  for (const item of report.summary.segments) {
    const card = element("div", `segment-card ${item.label.toLowerCase()}`);
    card.append(element("span", "", item.label), element("strong", "", percentage(item.rate)), element("small", "", `cancelación · ${number(item.bookings)} reservas`));
    const track = element("div", "bar-track"), fill = element("div", "bar-fill");
    fill.style.width = `${item.rate * 100}%`;
    track.append(fill); card.append(track); $("#segment-chart").append(card);
  }
}

if (document.body.dataset.page === "data") {
  let page = 1, controller;
  const columns = Array.from(document.querySelectorAll("#data-table th")).map(node => node.textContent.trim());
  async function loadRows(requestedPage = 1) {
    controller?.abort(); controller = new AbortController();
    const params = new URLSearchParams({page: requestedPage, month: $("#data-month").value, status: $("#data-status").value});
    try {
      showMessage("#data-error", "");
      const response = await checked(await fetch(`/api/data?${params}`, {signal: controller.signal}));
      const result = await response.json();
      page = result.page;
      $("#data-table tbody").replaceChildren();
      for (const row of result.rows) {
        const tr = element("tr");
        for (const key of columns) {
          const td = element("td");
          if (key === "booking_status") {
            td.append(element("span", `booking-status ${row[key] === "Canceled" ? "cancelled" : ""}`, row[key] === "Canceled" ? "Cancelada" : "No cancelada"));
          } else td.textContent = key === "avg_price_per_room" ? number(row[key], 2) : row[key];
          tr.append(td);
        }
        $("#data-table tbody").append(tr);
      }
      $("#data-count").textContent = `${number(result.total)} reservas · página ${page} de ${number(result.pages)}`;
      $("#previous-page").disabled = page <= 1;
      $("#next-page").disabled = page >= result.pages;
    } catch (error) { if (error.name !== "AbortError") showMessage("#data-error", error.message); }
  }
  $("#data-month").addEventListener("change", () => loadRows());
  $("#data-status").addEventListener("change", () => loadRows());
  $("#previous-page").addEventListener("click", () => loadRows(page - 1));
  $("#next-page").addEventListener("click", () => loadRows(page + 1));
  loadRows();
}

if (document.body.dataset.page === "model") {
  horizontalChart("#importance-chart", report.importance.map(item => ({label: item.label, value: item.value})), Math.max(...report.importance.map(item => item.value)));
  for (const [key, label] of [["baseline", "Árbol inicial"], ["tuned", "Árbol ajustado"]]) {
    const group = element("div", "comparison-group"); group.append(element("span", "", label));
    for (const split of ["train", "test"]) {
      const row = element("div", `comparison-line ${split}`), track = element("div", "bar-track"), fill = element("div", "bar-fill");
      fill.style.width = `${report[key][split].f1 * 100}%`;
      track.append(fill); row.append(track, element("span", "", percentage(report[key][split].f1))); group.append(row);
    }
    $("#comparison-chart").append(group);
  }
  $("#retrain").addEventListener("click", async () => {
    const button = $("#retrain"); button.disabled = true;
    showMessage("#train-message", "Entrenando el árbol inicial y evaluando las cuatro combinaciones de parámetros…");
    try {
      await checked(await fetch("/api/train", {method: "POST"}));
      sessionStorage.setItem("trained", "1"); location.reload();
    } catch (error) { showMessage("#train-message", error.message); button.disabled = false; }
  });
  if (sessionStorage.getItem("trained")) {
    sessionStorage.removeItem("trained");
    showMessage("#train-message", "Entrenamiento completado. Las métricas y el modelo guardado se han actualizado.");
  }
}

if (document.body.dataset.page === "predict") {
  const form = $("#prediction-form");
  let requestId = 0;
  function clearResult() {
    requestId++;
    $("#result-content").hidden = true; $("#result-empty").hidden = false;
    showMessage("#prediction-error", "");
  }
  form.addEventListener("input", clearResult);
  const examples = {
    low: {lead_time: 5, avg_price_per_room: 106.68, market_segment_type: "Online", arrival_month: 11, no_of_adults: 2, no_of_special_requests: 1, no_of_week_nights: 3, no_of_weekend_nights: 2, required_car_parking_space: false},
    high: {lead_time: 211, avg_price_per_room: 100, market_segment_type: "Online", arrival_month: 5, no_of_adults: 2, no_of_special_requests: 0, no_of_week_nights: 2, no_of_weekend_nights: 0, required_car_parking_space: false}
  };
  document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => {
    for (const [key, value] of Object.entries(examples[button.dataset.example])) {
      if (form.elements[key].type === "checkbox") form.elements[key].checked = value;
      else form.elements[key].value = value;
    }
    clearResult();
  }));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const currentId = ++requestId, button = form.querySelector("[type=submit]");
    button.disabled = true; showMessage("#prediction-error", "");
    const values = Object.fromEntries(new FormData(form));
    values.required_car_parking_space = form.elements.required_car_parking_space.checked ? 1 : 0;
    for (const key of Object.keys(values)) if (key !== "market_segment_type") values[key] = Number(values[key]);
    try {
      const response = await checked(await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(values)}));
      const result = await response.json();
      if (currentId !== requestId) return;
      $("#result-empty").hidden = true; $("#result-content").hidden = false;
      $("#probability-number").textContent = percentage(result.probability);
      $("#probability-ring").style.setProperty("--probability", `${result.probability * 100}%`);
      $("#probability-ring").style.setProperty("--ring-color", result.cancelled ? "#c39a6d" : "#829c6e");
      $("#result-verdict").textContent = result.cancelled ? "Predicción: cancelación" : "Predicción: no cancelación";
      $("#result-verdict").classList.toggle("high", result.cancelled);
      $("#result-action").textContent = result.cancelled ? "Prioriza una revisión de esta reserva y valora un seguimiento con el huésped." : "Mantén el seguimiento habitual. Una probabilidad baja no descarta una cancelación.";
      $("#booking-value").textContent = money(result.booking_value);
      $("#expected-exposure").textContent = money(result.expected_exposure);
      $("#decision-rules").replaceChildren(...result.rules.map(rule => element("li", "", `${rule.label}: ${number(rule.value, 2)} ${rule.operator} ${number(rule.threshold, 2)}`)));
      $("#leaf-samples").textContent = `Esta hoja contiene ${number(result.leaf_samples)} reservas de entrenamiento. La probabilidad utiliza pesos de clase equilibrados.`;
      showMessage("#range-warnings", result.warnings.join(" "));
    } catch (error) { if (currentId === requestId) showMessage("#prediction-error", error.message); }
    finally { button.disabled = false; }
  });
  $("#batch-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.currentTarget.querySelector("button"); button.disabled = true;
    showMessage("#batch-message", "Calculando las predicciones…");
    try {
      const response = await checked(await fetch("/api/batch", {method: "POST", body: new FormData(event.currentTarget)}));
      const url = URL.createObjectURL(await response.blob());
      const link = element("a"); link.href = url; link.download = "predicciones.csv"; document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      showMessage("#batch-message", "Predicciones listas. Se ha descargado predicciones.csv.");
    } catch (error) { showMessage("#batch-message", error.message); }
    finally { button.disabled = false; }
  });
}

if (document.body.dataset.page === "guide") {
  function calculate() {
    const riskInput = $("#scenario-risk"), costInput = $("#scenario-cost");
    const recovery = Number($("#scenario-recovery").value);
    $("#recovery-label").textContent = `${recovery}%`;
    $("#scenario-result").textContent = riskInput.checkValidity() && costInput.checkValidity() && riskInput.value && costInput.value
      ? money(Number(riskInput.value) * recovery / 100 - Number(costInput.value)) : "Revisa los importes";
  }
  for (const id of ["#scenario-risk", "#scenario-cost", "#scenario-recovery"]) $(id).addEventListener("input", calculate);
  calculate();
}
