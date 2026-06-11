import { useEffect, useMemo, useState } from "react";
import {
  API_BASE,
  composeScene,
  exportReport,
  generateHeatmap,
  scoreCandidates,
  uploadBackground,
  uploadForeground
} from "./api";
import AssetPanel from "./components/AssetPanel";
import DragCanvas from "./components/DragCanvas";
import Magnet from "./components/Magnet";
import ParameterPanel from "./components/ParameterPanel";
import ResultsTabs from "./components/ResultsTabs";
import SpotlightCard from "./components/SpotlightCard";

const initialParams = {
  score_backend: "handin_opa_subprocess",
  mask_mode: "自动判断",
  white_bg_threshold: 38,
  scale: 0.25,
  top_k: 3,
  filter_out_of_bounds: true,
  enable_explanation: false,
  enable_saliency: false,
  enable_feature_analysis: false,
  enable_libcom_suite: false,
  libcom_suite_models: ["fopa", "fos", "harmony"],
  lbm_steps: 4,
  lbm_resolution: 768,
  occlusion_patch_size: 96,
  occlusion_stride: 96
};

export default function App() {
  const [foreground, setForeground] = useState(null);
  const [background, setBackground] = useState(null);
  const [scene, setScene] = useState(null);
  const [position, setPosition] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [params, setParams] = useState(initialParams);
  const [result, setResult] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [candidatePulse, setCandidatePulse] = useState(false);
  const [readyPulse, setReadyPulse] = useState(false);
  const [reportPulse, setReportPulse] = useState(false);

  useEffect(() => {
    if (!candidates.length) return;
    setCandidatePulse(true);
    const timer = window.setTimeout(() => setCandidatePulse(false), 720);
    return () => window.clearTimeout(timer);
  }, [candidates.length]);

  useEffect(() => {
    if (!scene?.session_id) return;
    setReadyPulse(true);
    const timer = window.setTimeout(() => setReadyPulse(false), 900);
    return () => window.clearTimeout(timer);
  }, [scene?.session_id]);

  const candidateTable = useMemo(
    () => candidates.map((item) => ({ ...item, scale: Number(item.scale).toFixed(2) })),
    [candidates]
  );
  const activePosition = position || scene?.initial_state || null;

  async function handleUpload(kind, file) {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const data = kind === "foreground" ? await uploadForeground(file) : await uploadBackground(file);
      if (kind === "foreground") setForeground(data);
      else setBackground(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleCompose() {
    if (!foreground?.path || !background?.path) {
      setError("请先上传前景图和背景图。");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await composeScene({
        background_path: background.path,
        foreground_path: foreground.path,
        mask_mode: params.mask_mode,
        white_bg_threshold: params.white_bg_threshold,
        scale: params.scale
      });
      setScene({
        ...data,
        background: { ...data.background, url: `${API_BASE}${data.background.url}` },
        foreground: { ...data.foreground, url: `${API_BASE}${data.foreground.url}` },
        mask_preview_url: data.mask_preview_url ? `${API_BASE}${data.mask_preview_url}` : null
      });
      setPosition(data.initial_state);
      setCandidates([]);
      setResult(null);
      setHeatmap(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleRecordCandidate() {
    if (!position) return;
    setCandidates((prev) => [
      ...prev,
      { id: prev.length + 1, x: position.x, y: position.y, scale: position.scale }
    ]);
  }

  function handleClearCandidates() {
    setCandidates([]);
  }

  async function handleScore() {
    if (!scene?.session_id || candidates.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const payload = await scoreCandidates({
        session_id: scene.session_id,
        candidate_points: candidates,
        top_k: params.top_k,
        filter_out_of_bounds: params.filter_out_of_bounds,
        enable_explanation: params.enable_explanation,
        enable_saliency: params.enable_saliency,
        enable_feature_analysis: params.enable_feature_analysis,
        occlusion_patch_size: params.occlusion_patch_size,
        occlusion_stride: params.occlusion_stride,
        enable_libcom_suite: params.enable_libcom_suite,
        libcom_suite_models: params.libcom_suite_models,
        lbm_steps: params.lbm_steps,
        lbm_resolution: params.lbm_resolution,
        case_name: "SmartPlace Studio Case",
        background_note: "",
        foreground_note: "",
        manual_label: "",
        manual_reason: "",
        drag_mode_state: "React drag canvas",
        score_backend: params.score_backend
      });
      const normalizeUrls = (items = []) =>
        items.map((item) => ({
          ...item,
          image_url: item.image_url ? `${API_BASE}${item.image_url}` : null
        }));
      setResult({
        ...payload,
        ranked: normalizeUrls(payload.ranked),
        results: normalizeUrls(payload.results),
        explanations: Object.fromEntries(
          Object.entries(payload.explanations || {}).map(([key, value]) => [key, value ? `${API_BASE}${value}` : null])
        ),
        libcom_suite: {
          ...(payload.libcom_suite || {}),
          gallery: (payload.libcom_suite?.gallery || []).map((item) => ({
            ...item,
            url: item.url ? `${API_BASE}${item.url}` : null,
          })),
        },
        exports: Object.fromEntries(
          Object.entries(payload.exports || {}).map(([key, value]) => [key, value ? `${API_BASE}${value}` : null])
        )
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerateHeatmap() {
    if (!result?.run_id) return;
    setLoading(true);
    try {
      const payload = await generateHeatmap({
        run_id: result.run_id,
        patch_size: params.occlusion_patch_size,
        stride: params.occlusion_stride,
        score_backend: params.score_backend
      });
      setHeatmap({
        ...payload,
        overlay_url: `${API_BASE}${payload.overlay_url}`
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleExportReport() {
    if (!result?.run_id) return;
    setLoading(true);
    try {
      const payload = await exportReport({ run_id: result.run_id });
      setResult((prev) => ({
        ...prev,
        exports: Object.fromEntries(
          Object.entries(payload.exports || {}).map(([key, value]) => [key, value ? `${API_BASE}${value}` : null])
        )
      }));
      setReportPulse(true);
      window.setTimeout(() => setReportPulse(false), 720);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="app-backdrop" />
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark">SP</div>
          <div className="brand-copy">
            <h1>SmartPlace Studio</h1>
            <p>面向图像合成的智能物体放置工作台</p>
          </div>
        </div>

        <div className="header-actions">
          <span className="status-pill">交互式评估</span>
          <span className={`status-pill ${readyPulse ? "is-ring" : ""}`}>推理服务就绪</span>
          <span className="status-pill">LibCom OPA</span>
          <span className={`status-pill ${candidatePulse ? "is-ring" : ""}`}>候选位置 {candidates.length}</span>
          <Magnet padding={42} magnetStrength={18}>
            <button
              className={`primary pill-button ${reportPulse ? "has-splash" : ""}`}
              onClick={handleExportReport}
              disabled={!result?.run_id || loading}
            >
              导出报告
            </button>
          </Magnet>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="workbench-grid">
        <AssetPanel
          foreground={foreground}
          background={background}
          scene={scene}
          onUpload={handleUpload}
        />

        <section className="panel card canvas-panel">
          <div className="panel-heading canvas-panel-heading">
            <div>
              <span className="eyebrow">Workbench</span>
              <h3>拖拽画布</h3>
            </div>
            <div className={`candidate-count ${candidatePulse ? "is-ring" : ""}`}>候选位置 {candidates.length}</div>
          </div>

          <SpotlightCard className="canvas-stage-card" spotlightColor="rgba(96, 214, 255, 0.18)">
            <DragCanvas scene={scene} position={position} onChangePosition={setPosition} />

            <div className="canvas-floating-toolbar">
              <div className="canvas-stats">
                <div className="canvas-stat">
                  <span>x</span>
                  <strong>{activePosition?.x ?? "--"}</strong>
                </div>
                <div className="canvas-stat">
                  <span>y</span>
                  <strong>{activePosition?.y ?? "--"}</strong>
                </div>
                <div className="canvas-stat">
                  <span>scale</span>
                  <strong>{activePosition ? Number(activePosition.scale).toFixed(2) : "--"}</strong>
                </div>
              </div>

              <div className="canvas-action-group">
                <Magnet padding={30} magnetStrength={22}>
                  <button onClick={handleCompose} disabled={loading || !foreground || !background}>
                    加载画布
                  </button>
                </Magnet>
                <Magnet padding={30} magnetStrength={22}>
                  <button onClick={handleRecordCandidate} disabled={loading || !activePosition}>
                    记录当前位置
                  </button>
                </Magnet>
                <button className="ghost-button" onClick={handleClearCandidates} disabled={loading || candidates.length === 0}>
                  清空候选
                </button>
                <Magnet padding={34} magnetStrength={18}>
                  <button className="primary has-splash" onClick={handleScore} disabled={loading || candidates.length === 0}>
                    批量评分
                  </button>
                </Magnet>
              </div>
            </div>
          </SpotlightCard>

          <div className="candidate-list">
            <div className="candidate-list__head">
              <div>
                <h4>候选列表</h4>
                <p>记录当前拖拽位置，随后进行批量评分与推荐排序。</p>
              </div>
              <span className="mini-pill">{candidateTable.length} items</span>
            </div>
            <div className="table-scroll">
              <table className="candidate-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>X</th>
                    <th>Y</th>
                    <th>Scale</th>
                  </tr>
                </thead>
                <tbody>
                  {candidateTable.map((row, index) => (
                    <tr
                      key={row.id}
                      className="candidate-row-enter"
                      style={{ animationDelay: `${Math.min(index, 8) * 45}ms` }}
                    >
                      <td>{row.id}</td>
                      <td>{row.x}</td>
                      <td>{row.y}</td>
                      <td>{row.scale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <ParameterPanel
          params={params}
          setParams={setParams}
          loading={loading}
        />
      </main>

      <ResultsTabs
        result={result}
        heatmap={heatmap}
        onGenerateHeatmap={handleGenerateHeatmap}
        candidates={candidateTable}
      />
    </div>
  );
}
