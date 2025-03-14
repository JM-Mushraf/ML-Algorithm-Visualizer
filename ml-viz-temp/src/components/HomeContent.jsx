"use client";

import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from 'react-router-dom';
import "./HomeContent.css";

function HomeContent() {
  const [isMobile, setIsMobile] = useState(false);
  const barHeights = [30, 60, 90, 40, 80, 50, 70];
  const navigate = useNavigate();

  const decisionTreeData = [
    { id: 1, x: 150, y: 50, parent: null },
    { id: 2, x: 50, y: 150, parent: 1 },
    { id: 3, x: 250, y: 150, parent: 1 },
    { id: 4, x: 0, y: 250, parent: 2 },
    { id: 5, x: 100, y: 250, parent: 2 },
    { id: 6, x: 200, y: 250, parent: 3 },
    { id: 7, x: 300, y: 250, parent: 3 },
  ];

  const neuralConnections = [
    { x1: 50, y1: 50, x2: 150, y2: 100 },
    { x1: 150, y1: 100, x2: 250, y2: 50 },
    { x1: 50, y1: 150, x2: 150, y2: 100 },
    { x1: 150, y1: 100, x2: 250, y2: 150 },
  ];

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 480);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="home-content"
    >
      <motion.h1
        className="home-title gradient-text purple-to-pink"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.7, ease: "easeOut" }}
      >
        ML Visualizer
      </motion.h1>

      <motion.p
        className="home-subtitle"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7 }}
      >
        Explore machine learning algorithms through interactive visualizations
      </motion.p>

      <motion.div
        className="visualization-container"
        style={{
          display: "flex",
          flexDirection: isMobile ? "column" : "row",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "20px",
          width: "100%",
          maxWidth: "1200px",
          margin: "0 auto",
        }}
      >
        {!isMobile && (
          <motion.div
            className="bar-chart"
            style={{
              display: "flex",
              gap: "10px",
              alignItems: "flex-end",
              height: "200px",
              width: "30%",
            }}
          >
            {barHeights.map((height, index) => (
              <motion.div
                key={index}
                style={{
                  width: "30px",
                  backgroundColor: "#6b5b95",
                  borderRadius: "5px",
                }}
                animate={{
                  height: [`${height}px`, `${height + 20}px`, `${height}px`],
                }}
                transition={{
                  repeat: Infinity,
                  repeatType: "reverse",
                  duration: 1.5,
                  ease: "easeInOut",
                  delay: index * 0.1,
                }}
              />
            ))}
          </motion.div>
        )}

        {!isMobile && (
          <motion.div
            className="decision-tree"
            style={{
              position: "relative",
              width: "30%",
              height: "300px",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            {decisionTreeData.map((node) => {
              const parentNode = decisionTreeData.find(
                (n) => n.id === node.parent
              );
              return (
                <>
                  {parentNode && (
                    <motion.div
                      key={`line-${node.id}`}
                      style={{
                        position: "absolute",
                        width: `${Math.sqrt(
                          (node.x - parentNode.x) ** 2 +
                            (node.y - parentNode.y) ** 2
                        )}px`,
                        height: "2px",
                        backgroundColor: "#6b5b95",
                        top: `${parentNode.y}px`,
                        left: `${parentNode.x}px`,
                        transformOrigin: "0% 0%",
                        transform: `rotate(${Math.atan2(
                          node.y - parentNode.y,
                          node.x - parentNode.x
                        )}rad)`,
                      }}
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{
                        repeat: Infinity,
                        duration: 1.5,
                        ease: "easeInOut",
                      }}
                    />
                  )}
                  <motion.div
                    key={`node-${node.id}`}
                    style={{
                      position: "absolute",
                      width: "20px",
                      height: "20px",
                      backgroundColor: "#ff6f61",
                      borderRadius: "50%",
                      top: `${node.y}px`,
                      left: `${node.x}px`,
                    }}
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{
                      repeat: Infinity,
                      duration: 1.5,
                      ease: "easeInOut",
                      delay: node.id * 0.2,
                    }}
                  />
                </>
              );
            })}
          </motion.div>
        )}

        <motion.div
          className="neural-network"
          style={{
            position: "relative",
            width: isMobile ? "100%" : "30%",
            height: "200px",
          }}
        >
          {[50, 150, 250].map((x, i) => (
            <motion.div
              key={i}
              style={{
                position: "absolute",
                width: "20px",
                height: "20px",
                backgroundColor: "#ff6f61",
                borderRadius: "50%",
                top: i % 2 === 0 ? "50px" : "100px",
                left: `${x}px`,
              }}
              animate={{ y: [0, -10, 0] }}
              transition={{
                repeat: Infinity,
                duration: 1.5,
                ease: "easeInOut",
                delay: i * 0.2,
              }}
            />
          ))}

          {neuralConnections.map((conn, i) => (
            <motion.div
              key={i}
              style={{
                position: "absolute",
                width: `${Math.sqrt(
                  (conn.x2 - conn.x1) ** 2 + (conn.y2 - conn.y1) ** 2
                )}px`,
                height: "2px",
                backgroundColor: "#6b5b95",
                top: `${conn.y1}px`,
                left: `${conn.x1}px`,
                transformOrigin: "0% 0%",
                transform: `rotate(${Math.atan2(
                  conn.y2 - conn.y1,
                  conn.x2 - conn.x1
                )}rad)`,
              }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{
                repeat: Infinity,
                duration: 1.5,
                ease: "easeInOut",
                delay: i * 0.1,
              }}
            />
          ))}
        </motion.div>
      </motion.div>

      <motion.div
        className="algorithm-tags"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.7 }}
      >
        <motion.div
          className="algorithm-tag"
          whileHover={{ scale: 1.1, backgroundColor: "#6b5b95" }}
          transition={{ duration: 0.3 }}
          onClick={() => navigate('/regression')}
        >
          Regression
        </motion.div>
        <motion.div
          className="algorithm-tag"
          whileHover={{ scale: 1.1, backgroundColor: "#6b5b95" }}
          transition={{ duration: 0.3 }}
          onClick={() => navigate('/classification')}
        >
          Classification
        </motion.div>
      </motion.div>
    </motion.div>
  );
}

export default HomeContent;