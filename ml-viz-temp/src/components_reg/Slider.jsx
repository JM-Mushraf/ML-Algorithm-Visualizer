import "./Slider.css";

export function Slider({
  value,
  onValueChange,
  min,
  max,
  step = 1,
  className = "",
}) {
  const percentage = ((value[0] - min) / (max - min)) * 100;

  return (
    <div className={`slider-wrapper ${className}`}>
      <div className="slider-track">
        <div className="slider-fill" style={{ width: `${percentage}%` }} />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value[0]}
        onChange={(e) => onValueChange([Number(e.target.value)])}
        className="slider-input"
      />
      <div
        className="slider-thumb"
        style={{ left: `calc(${percentage}% - 8px)` }}
      />
    </div>
  );
}
