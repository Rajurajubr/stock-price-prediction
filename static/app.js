let chartInstance = null;
const $ = (id) => document.getElementById(id);

$("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("error").classList.add("hidden");
  $("result").classList.add("hidden");
  $("loading").classList.remove("hidden");
  $("predict-btn").disabled = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({symbol: $("symbol").value.trim()})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    $("direction").textContent = data.direction;
    $("direction-pill").textContent = data.direction;
    $("direction-pill").className = "direction-pill " + data.direction.toLowerCase();
    $("predicted").textContent = data.predicted_close.toLocaleString(undefined, {maximumFractionDigits: 2});
    $("previous").textContent = data.previous_close.toLocaleString(undefined, {maximumFractionDigits: 2});
    $("change").textContent = (data.change_pct > 0 ? "+" : "") + data.change_pct + "% estimated change";
    $("rows").textContent = data.rows;
    $("mae").textContent = data.mae === null ? "N/A" : data.mae;
    $("date").textContent = data.last_date;
    $("candle-note").textContent = data.candle_note;
    $("disclaimer").textContent = data.disclaimer;
    $("result").classList.remove("hidden");
    $("chart-empty").classList.add("hidden");
    drawChart(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").classList.remove("hidden");
  } finally {
    $("loading").classList.add("hidden");
    $("predict-btn").disabled = false;
  }
});

function drawChart(data) {
  const ctx = $("prediction-chart").getContext("2d");
  if (chartInstance) chartInstance.destroy();
  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Latest close", "Estimated next close"],
      datasets: [{
        data: [data.previous_close, data.predicted_close],
        backgroundColor: ["rgba(117,169,255,.8)", "rgba(73,224,160,.8)"],
        borderRadius: 8,
        maxBarThickness: 75
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {legend: {display: false}},
      scales: {
        x: {ticks: {color: "#9aabc2"}, grid: {display: false}},
        y: {ticks: {color: "#9aabc2"}, grid: {color: "rgba(160,185,220,.1)"}}
      }
    }
  });
}

$("image-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("image-error").classList.add("hidden");
  $("image-result").classList.add("hidden");
  const file = $("chart-file").files[0];
  if (!file) return;
  const body = new FormData();
  body.append("chart", file);
  try {
    const response = await fetch("/api/image", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Image analysis failed.");
    $("uploaded-preview").src = data.image_url;
    $("visual-bias").textContent = data.visual_bias;
    $("visual-explanation").textContent = data.explanation;
    $("visual-limitation").textContent = data.limitation;
    $("image-result").classList.remove("hidden");
  } catch (err) {
    $("image-error").textContent = err.message;
    $("image-error").classList.remove("hidden");
  }
});
