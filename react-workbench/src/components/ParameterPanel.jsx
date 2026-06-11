const sliderFields = [
  ["white_bg_threshold", "去底阈值", 10, 100, 2],
  ["scale", "前景缩放比例", 0.05, 0.8, 0.05],
  ["top_k", "Top-K 数量", 1, 5, 1]
];

const advancedLibcomFields = [
  ["lbm_steps", "LBM 步数", 1, 8, 1],
  ["lbm_resolution", "LBM 分辨率", 512, 1024, 128]
];

const advancedExplainFields = [
  ["occlusion_patch_size", "遮挡块大小", 48, 160, 16],
  ["occlusion_stride", "遮挡滑动步长", 32, 128, 16]
];

function formatValue(value) {
  return Number.isInteger(value) ? value : Number(value).toFixed(2);
}

function SliderField({ label, value, min, max, step, onChange }) {
  const rangeStyle = {
    "--min": min,
    "--max": max,
    "--value": value
  };

  return (
    <div className="field slider-field">
      <div className="field-head">
        <span>{label}</span>
        <strong>{formatValue(value)}</strong>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={rangeStyle}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="field-meta">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export default function ParameterPanel({ params, setParams, loading }) {
  const update = (key, value) => setParams((prev) => ({ ...prev, [key]: value }));

  return (
    <aside className="panel card parameter-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Controls</span>
          <h3>模型与参数</h3>
        </div>
        <span className="mini-pill">Live</span>
      </div>

      <label className="field">
        <span>评分后端</span>
        <select value={params.score_backend} onChange={(e) => update("score_backend", e.target.value)}>
          <option value="handin_opa_subprocess">handin OPA + SmartPlace 校准</option>
          <option value="libcom_opa_subprocess">LibCom OPA + SmartPlace 校准</option>
        </select>
      </label>

      <label className="field">
        <span>前景处理模式</span>
        <select value={params.mask_mode} onChange={(e) => update("mask_mode", e.target.value)}>
          <option value="自动判断">自动判断</option>
          <option value="透明 PNG Alpha">透明 PNG Alpha</option>
          <option value="浅色/纯色背景去除">浅色/纯色背景去除</option>
          <option value="U2Net 自动抠图">U2Net 自动抠图</option>
          <option value="不处理">不处理</option>
        </select>
      </label>

      <div className="slider-stack">
        {sliderFields.map(([key, label, min, max, step]) => (
          <SliderField
            key={key}
            label={label}
            value={params[key]}
            min={min}
            max={max}
            step={step}
            onChange={(value) => update(key, value)}
          />
        ))}
      </div>

      <div className="toggle-stack">
        {[
          ["filter_out_of_bounds", "过滤越界候选"],
          ["enable_explanation", "启用热力图解释"],
          ["enable_saliency", "启用显著性图"],
          ["enable_feature_analysis", "启用中间特征分析"],
          ["enable_libcom_suite", "启用 LibCom 增强模型"]
        ].map(([key, label]) => (
          <label className="toggle-row" key={key}>
            <div className="toggle-copy">
              <span>{label}</span>
            </div>
            <input
              type="checkbox"
              checked={params[key]}
              onChange={(e) => update(key, e.target.checked)}
            />
          </label>
        ))}
      </div>

      <div className="field">
        <span>增强模型</span>
        <div className="chip-grid">
          {["fopa", "fos", "harmony", "pctnet", "lbm"].map((name) => {
            const active = params.libcom_suite_models.includes(name);
            return (
              <button
                key={name}
                type="button"
                className={`chip ${active ? "active" : ""}`}
                onClick={() =>
                  update(
                    "libcom_suite_models",
                    active
                      ? params.libcom_suite_models.filter((item) => item !== name)
                      : [...params.libcom_suite_models, name]
                  )
                }
              >
                {name.toUpperCase()}
              </button>
            );
          })}
        </div>
      </div>

      <details className="details-card">
        <summary>LibCom 高级参数</summary>
        <div className="slider-stack details-stack">
          {advancedLibcomFields.map(([key, label, min, max, step]) => (
            <SliderField
              key={key}
              label={label}
              value={params[key]}
              min={min}
              max={max}
              step={step}
              onChange={(value) => update(key, value)}
            />
          ))}
        </div>
      </details>

      <details className="details-card">
        <summary>解释图高级参数</summary>
        <div className="slider-stack details-stack">
          {advancedExplainFields.map(([key, label, min, max, step]) => (
            <SliderField
              key={key}
              label={label}
              value={params[key]}
              min={min}
              max={max}
              step={step}
              onChange={(value) => update(key, value)}
            />
          ))}
        </div>
      </details>
    </aside>
  );
}
