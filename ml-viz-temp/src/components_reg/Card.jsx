import "./Card.css"

export function Card({ className = "", children, ...props }) {
  return (
    <div className={`card-t ${className}`} {...props}>
      {children}
    </div>
  )
}

export function CardContent({ className = "", children, ...props }) {
  return (
    <div className={`card-content-t ${className}`} {...props}>
      {children}
    </div>
  )
}