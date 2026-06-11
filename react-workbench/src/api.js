const API_BASE = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    throw new Error(`无法连接后端服务 ${API_BASE}，请确认 FastAPI 已启动。`);
  }

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

export async function uploadForeground(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/upload_foreground", { method: "POST", body: form });
}

export async function uploadBackground(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/upload_background", { method: "POST", body: form });
}

export async function composeScene(payload) {
  return request("/api/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function scoreCandidates(payload) {
  return request("/api/score_candidates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function generateHeatmap(payload) {
  return request("/api/generate_heatmap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function exportReport(payload) {
  return request("/api/export_report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export { API_BASE };
