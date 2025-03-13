"use client"

import { useState, useRef, useEffect } from "react"
import "./Chatbot.css"

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)

  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Function to handle sending a message
  const handleSendMessage = async () => {
    if (inputText.trim()) {
      // Add user message to the chat
      setMessages([...messages, { text: inputText, sender: "user" }])
      setInputText("")
      setIsTyping(true)

      try {
        // Send the user message to the backend
        const response = await fetch("http://localhost:5000/chatbot", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: inputText }),
        })

        if (!response.ok) {
          throw new Error("Network response was not ok")
        }

        const data = await response.json()

        // Add chatbot response to the chat after a small delay to simulate typing
        setTimeout(() => {
          setIsTyping(false)
          setMessages((prevMessages) => [...prevMessages, { text: data.response, sender: "bot" }])
        }, 1000)
      } catch (error) {
        console.error("Error:", error)
        // Add an error message to the chat
        setTimeout(() => {
          setIsTyping(false)
          setMessages((prevMessages) => [
            ...prevMessages,
            { text: "Sorry, something went wrong. Please try again.", sender: "bot" },
          ])
        }, 1000)
      }
    }
  }

  return (
    <div className={`chatbot-container ${isOpen ? "open" : ""}`}>
      {/* Chatbot Interface */}
      <div className="chatbot-interface">
        <div className="chatbot-header">
          <span>Chat Assistant</span>
        </div>
        <div className="chatbot-body">
          <div className="chatbot-messages">
            {messages.length === 0 && <div className="message bot">Hi there! How can I help you today?</div>}
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.sender === "user" ? "user" : "bot"}`}>
                {message.text}
              </div>
            ))}
            {isTyping && (
              <div className="message bot typing">
                <span className="typing-dot"></span>
                <span className="typing-dot"></span>
                <span className="typing-dot"></span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type a message..."
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            />
            <button onClick={handleSendMessage}></button>
          </div>
        </div>
      </div>

      {/* Toggle Button */}
      <div className="chatbot-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? "✕" : "💬"}
      </div>
    </div>
  )
}

export default Chatbot;