import { useRef, useState } from "react";
import { API_BASE } from "../api";
import Magnet from "./Magnet";
import SpotlightCard from "./SpotlightCard";

function TiltCard({
  className = "",
  children,
  spotlightColor = "rgba(108, 204, 255, 0.16)",
  onClick
}) {
  const ref = useRef(null);

  function handleMouseMove(event) {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    const rotateY = (px - 0.5) * 6;
    const rotateX = (0.5 - py) * 6;
    ref.current.style.setProperty("--tilt-rotate-x", `${rotateX.toFixed(2)}deg`);
    ref.current.style.setProperty("--tilt-rotate-y", `${rotateY.toFixed(2)}deg`);
  }

  function handleMouseLeave() {
    if (!ref.current) return;
    ref.current.style.setProperty("--tilt-rotate-x", "0deg");
    ref.current.style.setProperty("--tilt-rotate-y", "0deg");
  }

  return (
    <SpotlightCard className={`tilted-card ${className}`} spotlightColor={spotlightColor}>
      <div
        ref={ref}
        className="tilted-card__inner"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={onClick}
      >
        {children}
      </div>
    </SpotlightCard>
  );
}

function friendlyAssetName(asset, displayName, fallback) {
  if (displayName) return displayName;
  if (asset?.name) return asset.name;
  return fallback;
}

function AssetDropzone({
  type,
  asset,
  displayName,
  inputRef,
  onUploadFile
}) {
  const [dragState, setDragState] = useState("idle");
  const [pulse, setPulse] = useState(false);
  const ready = Boolean(asset);
  const isBackground = type === "background";
  const title = isBackground ? "Background" : "Foreground";
  const emptyTitle = isBackground ? "Drop background image here" : "Drop foreground image here";
  const releaseText = isBackground ? "Release to upload background" : "Release to upload foreground";
  const uploadText = isBackground ? "Upload Background" : "Upload Foreground";
  const friendlyName = friendlyAssetName(asset, displayName, title);

  async function acceptFile(file) {
    if (!file) return;
    await onUploadFile(file);
    setPulse(true);
    window.setTimeout(() => setPulse(false), 280);
  }

  function handleDragOver(event) {
    event.preventDefault();
    if (event.dataTransfer?.types?.includes("Files")) {
      event.dataTransfer.dropEffect = "copy";
      setDragState("valid");
    }
  }

  function handleDragLeave(event) {
    event.preventDefault();
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setDragState("idle");
    }
  }

  async function handleDrop(event) {
    event.preventDefault();
    setDragState("idle");
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      await acceptFile(file);
    }
  }

  async function handleInputChange(event) {
    await acceptFile(event.target.files?.[0]);
  }

  return (
    <SpotlightCard
      className={`asset-dropzone-card ${type} ${ready ? "is-ready" : "is-empty"} is-${dragState} ${pulse ? "is-pulse" : ""}`}
      spotlightColor="rgba(102, 208, 255, 0.16)"
    >
      <div
        className="asset-dropzone"
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className="asset-hidden-input"
          type="file"
          accept="image/*"
          onChange={handleInputChange}
        />

        {ready ? (
          <>
            <div className="asset-dropzone__head">
              <div className="asset-dropzone__title">
                <span>{title}</span>
                <strong title={friendlyName}>{friendlyName}</strong>
              </div>
              <Magnet padding={36} magnetStrength={20}>
                <button type="button" className="asset-replace-button" onClick={() => inputRef.current?.click()}>
                  Change
                </button>
              </Magnet>
            </div>

            <div className={`asset-preview-frame ${isBackground ? "is-wide" : ""}`}>
              <img className="asset-preview-frame__image" src={`${API_BASE}${asset.url}`} alt={`${type}-preview`} />
            </div>
          </>
        ) : (
          <div
            className="asset-dropzone__empty"
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
          >
            <div className="asset-dropzone__empty-copy">
              <strong>{dragState === "valid" ? releaseText : emptyTitle}</strong>
              <span>PNG / JPG supported</span>
            </div>
            <Magnet padding={36} magnetStrength={20}>
              <button type="button" className="asset-upload-button">
                {uploadText}
              </button>
            </Magnet>
          </div>
        )}
      </div>
    </SpotlightCard>
  );
}

export default function AssetPanel({ foreground, background, scene, onUpload }) {
  const foregroundInputRef = useRef(null);
  const backgroundInputRef = useRef(null);
  const [displayNames, setDisplayNames] = useState({ foreground: "", background: "" });

  async function handleDirectUpload(kind, file) {
    if (!file) return;
    await onUpload(kind, file);
    setDisplayNames((prev) => ({ ...prev, [kind]: file.name }));
  }

  return (
    <section className="panel card asset-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Assets</span>
          <h3>素材区</h3>
        </div>
        <span className={`mini-pill ${foreground && background ? "is-ring" : ""}`}>
          {foreground && background ? "就绪" : "待上传"}
        </span>
      </div>

      <div className="asset-flow">
        <div className="asset-block">
          <div className="asset-block__label">Foreground</div>
          <AssetDropzone
            type="foreground"
            asset={foreground}
            displayName={displayNames.foreground}
            inputRef={foregroundInputRef}
            onUploadFile={(file) => handleDirectUpload("foreground", file)}
          />
        </div>

        <div className="asset-block">
          <div className="asset-block__label">Background</div>
          <AssetDropzone
            type="background"
            asset={background}
            displayName={displayNames.background}
            inputRef={backgroundInputRef}
            onUploadFile={(file) => handleDirectUpload("background", file)}
          />
        </div>
      </div>

      {scene?.mask_preview_url && (
        <TiltCard className="asset-mask-card" spotlightColor="rgba(117, 230, 206, 0.14)">
          <div className="asset-mask-card__head">
            <span>Mask Preview</span>
            <strong>Generated</strong>
          </div>
          <div className="asset-mask-card__frame">
            <img className="asset-preview-frame__image" src={scene.mask_preview_url} alt="mask-preview" />
          </div>
        </TiltCard>
      )}
    </section>
  );
}
