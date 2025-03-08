import "./Button.css"

export function Button({ children, className = "", variant = "default", size = "default", ...props }) {
  const variantClass =
    variant === "default"
      ? "btn-default"
      : variant === "outline"
        ? "btn-outline"
        : variant === "ghost"
          ? "btn-ghost"
          : "btn-default"

  const sizeClass =
    size === "default"
      ? "btn-size-default"
      : size === "sm"
        ? "btn-size-sm"
        : size === "lg"
          ? "btn-size-lg"
          : size === "icon"
            ? "btn-size-icon"
            : "btn-size-default"

  return (
    <button className={`btn ${variantClass} ${sizeClass} ${className}`} {...props}>
      {children}
    </button>
  )
}
