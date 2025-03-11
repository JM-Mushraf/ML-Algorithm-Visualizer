import React, { useState } from "react";
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false); // State to manage chatbot visibility
  const [messages, setMessages] = useState([]); // State to store chat messages
  const [inputText, setInputText] = useState(""); // State to manage user input

  // Function to handle sending a message
  const handleSendMessage = async () => {
    if (inputText.trim()) {
      // Add user message to the chat
      setMessages([...messages, { text: inputText, sender: "user" }]);
      setInputText("");

      try {
        // Send the user message to the backend
        const response = await fetch('http://localhost:5000/chatbot', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: inputText }),
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Add chatbot response to the chat
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: data.response, sender: "bot" },
        ]);
      } catch (error) {
        console.error('Error:', error);
        // Add an error message to the chat
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: "Sorry, something went wrong. Please try again.", sender: "bot" },
        ]);
      }
    }
  };

  return (
    <div className={`chatbot-container ${isOpen ? "open" : ""}`}>
      {/* Toggle Button */}
      <div className="chatbot-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? "✕" : "💬"}
      </div>

      {/* Chatbot Interface */}
      <div className="chatbot-interface">
        <div className="chatbot-header">
          <span>Chatbot</span>
        </div>
        <div className="chatbot-body">
          <div className="chatbot-messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.sender === "user" ? "user" : "bot"}`}
              >
                {message.text}
              </div>
            ))}
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type a message..."
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            />
            <button onClick={handleSendMessage}>Send</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;