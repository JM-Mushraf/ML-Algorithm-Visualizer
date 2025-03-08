"use client"

import { useState, useRef, useEffect } from "react"
import { ChevronDown } from "./Icons"
import "./Select.css"

export function Select({ value, onValueChange, options, placeholder = "Select option" }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (ref.current && !ref.current.contains(event.target)) {
        setOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  const selectedOption = options.find((option) => option.value === value)

  return (
    <div className="select-container" ref={ref}>
      <SelectTrigger onClick={() => setOpen(!open)}>
        <SelectValue>{selectedOption?.label || placeholder}</SelectValue>
        <ChevronDown className="select-icon" />
      </SelectTrigger>

      {open && (
        <SelectContent>
          {options.map((option) => (
            <SelectItem
              key={option.value}
              onClick={() => {
                onValueChange(option.value)
                setOpen(false)
              }}
              className={value === option.value ? "selected" : ""}
              value={option.value}
            >
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      )}
    </div>
  )
}

export function SelectGroup({ className = "", children, ...props }) {
  return (
    <div className={`select-group ${className}`} {...props}>
      {children}
    </div>
  )
}

export function SelectTrigger({ className = "", children, onClick, ...props }) {
  return (
    <button className={`select-trigger ${className}`} onClick={onClick} {...props}>
      {children}
    </button>
  )
}

export function SelectValue({ className = "", children, ...props }) {
  return (
    <span className={`select-value ${className}`} {...props}>
      {children}
    </span>
  )
}

export function SelectContent({ className = "", children, ...props }) {
  return (
    <div className={`select-content ${className}`} {...props}>
      {children}
    </div>
  )
}

export function SelectLabel({ className = "", children, ...props }) {
  return (
    <label className={`select-label ${className}`} {...props}>
      {children}
    </label>
  )
}

export function SelectItem({ className = "", children, value, ...props }) {
  return (
    <div className={`select-item ${className}`} data-value={value} {...props}>
      {children}
    </div>
  )
}

export function SelectSeparator({ className = "", ...props }) {
  return <div className={`select-separator ${className}`} {...props} />
}

export function SelectScrollUpButton({ className = "", ...props }) {
  return (
    <div className={`select-scroll-button ${className}`} {...props}>
      <ChevronDown className="select-scroll-icon rotate" />
    </div>
  )
}

export function SelectScrollDownButton({ className = "", ...props }) {
  return (
    <div className={`select-scroll-button ${className}`} {...props}>
      <ChevronDown className="select-scroll-icon" />
    </div>
  )
}

