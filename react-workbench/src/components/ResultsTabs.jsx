import { useMemo, useState } from "react";

function formatScale(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "--";
}

function getMiniSummary(resultCount) {
  return resultCount > 0 ? `Top-${resultCount}` : "等待结果";
}

export default function ResultsTabs({ result, heatmap, onGenerateHeatmap, candidates = [] }) {
  const [tab, setTab] = useState("candidates");

  const ranked = result?.ranked || [];
  const scoreTable = result?.score_table || [];
  const topRecommendation = ranked[0] || null;
  const explanationImages = [
    result?.explanations?.occlusion_url
      ? { key: "occlusion", title: "遮挡热力图", url: result.explanations.occlusion_url }
      : null,
    result?.explanations?.saliency_url
      ? { key: "saliency", title: "显著性图", url: result.explanations.saliency_url }
      : null,
    result?.explanations?.feature_plot_url
      ? { key: "feature", title: "中间特征分析", url: result.explanations.feature_plot_url }
      : null,
  ].filter(Boolean);
  const libcomGallery = result?.libcom_suite?.gallery || [];
  const libcomText = result?.libcom_suite?.text || "";

  const candidateRows = useMemo(
    () =>
      candidates.map((row) => ({
        ...row,
        scale: formatScale(row.scale),
      })),
    [candidates]
  );

  void onGenerateHeatmap;
  void heatmap;

  return (
    <div className="panel card tabs-panel results-workbench">
      <div className="results-header">
        <div>
          <span className="eyebrow">Results</span>
          <h3>结果面板</h3>
        </div>
        <span className="mini-pill">{getMiniSummary(ranked.length)}</span>
      </div>

      <div className="tab-bar card-tabs">
        {[
          ["candidates", "Candidates"],
          ["topk", "Top-K Results"],
          ["analysis", "Explainability"],
          ["exports", "Report"],
        ].map(([id, label]) => (
          <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {tab === "candidates" && (
        <div className="results-section">
          <div className="section-copy">
            <h4>候选列表</h4>
            <p>保留当前所有拖拽候选，作为评分前的轻量候选清单和详细数据入口。</p>
          </div>

          <div className="animated-candidate-list">
            {candidateRows.length === 0 && <div className="empty-state-card">还没有记录候选位置。</div>}
            {candidateRows.map((row, index) => (
              <article
                key={row.id}
                className="candidate-item-card"
                style={{ animationDelay: `${Math.min(index, 10) * 50}ms` }}
              >
                <div className="candidate-item-card__head">
                  <strong>候选 {row.id}</strong>
                  <span className="mini-pill">scale {row.scale}</span>
                </div>
                <div className="candidate-item-card__meta">
                  <span>x {row.x}</span>
                  <span>y {row.y}</span>
                </div>
              </article>
            ))}
          </div>

          {candidateRows.length > 0 && (
            <div className="table-scroll result-table-shell">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>X</th>
                    <th>Y</th>
                    <th>Scale</th>
                  </tr>
                </thead>
                <tbody>
                  {candidateRows.map((row) => (
                    <tr key={row.id}>
                      <td>{row.id}</td>
                      <td>{row.x}</td>
                      <td>{row.y}</td>
                      <td>{row.scale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {tab === "topk" && (
        <div className="results-section">
          {topRecommendation && (
            <article className="top-recommendation-card">
              <div className="top-recommendation-card__header">
                <div>
                  <span className="eyebrow">Top-1</span>
                  <h4>最佳推荐位置</h4>
                </div>
                <span className="score-pill">{topRecommendation.score}</span>
              </div>

              <div className="top-recommendation-card__grid">
                <div className="top-recommendation-card__preview">
                  {topRecommendation.image_url && (
                    <img src={topRecommendation.image_url} alt={`top-candidate-${topRecommendation.id}`} />
                  )}
                </div>

                <div className="top-recommendation-card__body">
                  <div className="top-recommendation-card__stats">
                    <div><span>x</span><strong>{topRecommendation.x ?? "--"}</strong></div>
                    <div><span>y</span><strong>{topRecommendation.y ?? "--"}</strong></div>
                    <div><span>scale</span><strong>{formatScale(topRecommendation.scale)}</strong></div>
                    <div><span>rank</span><strong>#{topRecommendation.rank}</strong></div>
                  </div>

                  <div className="top-recommendation-card__reason">
                    <span>推荐理由</span>
                    <p>{topRecommendation.reason || "当前结果没有返回额外解释文本。"}</p>
                  </div>
                </div>
              </div>
            </article>
          )}

          <div className="gallery-grid result-gallery-grid">
            {ranked.length === 0 && <div className="empty-state-card">先完成批量评分，Top-K 推荐会显示在这里。</div>}
            {ranked.map((item) => (
              <figure key={item.id} className="result-card result-card-nav">
                {item.image_url && <img src={item.image_url} alt={`candidate-${item.id}`} />}
                <figcaption>
                  <span>Top {item.rank}</span>
                  <strong>候选 {item.id}</strong>
                  <em>{item.score}</em>
                </figcaption>
              </figure>
            ))}
          </div>

          {scoreTable.length > 0 && (
            <div className="table-scroll result-table-shell">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Rank</th>
                    <th>X</th>
                    <th>Y</th>
                    <th>Scale</th>
                    <th>Score</th>
                    <th>Label</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {scoreTable.map((row) => (
                    <tr key={row.candidate_id}>
                      <td>{row.candidate_id}</td>
                      <td>{row.rank}</td>
                      <td>{row.x}</td>
                      <td>{row.y}</td>
                      <td>{row.scale}</td>
                      <td>{row.score}</td>
                      <td>{row.label}</td>
                      <td>{row.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {tab === "analysis" && (
        <div className="results-section analysis-section">
          <div className="section-copy">
            <h4>解释性结果</h4>
            <p>这里会集中展示热力图、显著性图、中间特征分析，以及 LibCom 增强模型输出。</p>
          </div>

          <div className="analysis-panel">
            {explanationImages.length > 0 && (
              <div className="analysis-gallery">
                {explanationImages.map((item) => (
                  <figure key={item.key} className="analysis-card">
                    <img src={item.url} alt={item.title} className="analysis-image" />
                    <figcaption>{item.title}</figcaption>
                  </figure>
                ))}
              </div>
            )}

            <pre>{heatmap?.explanation || result?.run_analysis_text || "暂无解释结果"}</pre>

            {(libcomText || libcomGallery.length > 0) && (
              <section className="libcom-suite-panel">
                <div className="section-copy">
                  <h4>LibCom 增强结果</h4>
                  <p>增强模型的附加评估与生成结果会集中显示在这里。</p>
                </div>

                {libcomText && <pre>{libcomText}</pre>}

                {libcomGallery.length > 0 && (
                  <div className="analysis-gallery">
                    {libcomGallery.map((item, index) => (
                      <figure key={`${item.url}-${index}`} className="analysis-card">
                        {item.url && <img src={item.url} alt={item.caption || `libcom-${index}`} className="analysis-image" />}
                        <figcaption>{item.caption || `LibCom 输出 ${index + 1}`}</figcaption>
                      </figure>
                    ))}
                  </div>
                )}
              </section>
            )}
          </div>
        </div>
      )}

      {tab === "exports" && (
        <div className="results-section">
          <div className="section-copy">
            <h4>报告导出</h4>
            <p>导出评分结果、运行日志和报告文件，作为答辩或复现实验材料。</p>
          </div>

          <div className="exports-list">
            {Object.keys(result?.exports || {}).length === 0 && (
              <div className="empty-state-card">完成评分后，导出链接会出现在这里。</div>
            )}
            {Object.entries(result?.exports || {}).map(([key, value]) => (
              <a key={key} href={value} target="_blank" rel="noreferrer">
                {key}
              </a>
            ))}
            {result?.explanations?.report_url && (
              <a href={result.explanations.report_url} target="_blank" rel="noreferrer">
                解释报告
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
