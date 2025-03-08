import "./Badge.css";

export function Badge({
  variant = "default",
  className = "",
  children,
  ...props
}) {
  const variantClass =
    variant === "default"
      ? "badge-default"
      : variant === "secondary"
      ? "badge-secondary"
      : variant === "outline"
      ? "badge-outline"
      : "badge-default";

  return (
    <div className={`badge ${variantClass} ${className}`} {...props}>
      {children}
    </div>
  );
}
