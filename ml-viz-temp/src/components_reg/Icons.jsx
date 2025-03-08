// Helper function to create icon components
const createIcon = (path, viewBox = "0 0 24 24") => {
  return ({ className = "", ...props }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox={viewBox}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`icon ${className}`}
      {...props}
    >
      {path}
    </svg>
  );
};

export const LineChart = createIcon(
  <>
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <polyline points="8 5 3 12 8 19"></polyline>
    <polyline points="16 5 21 12 16 19"></polyline>
  </>
);

export const ScatterChart = createIcon(
  <>
    <circle cx="7.5" cy="7.5" r="2"></circle>
    <circle cx="16.5" cy="7.5" r="2"></circle>
    <circle cx="7.5" cy="16.5" r="2"></circle>
    <circle cx="16.5" cy="16.5" r="2"></circle>
  </>
);

export const Brain = createIcon(
  <>
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.54Z"></path>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.54Z"></path>
  </>
);

export const Layers = createIcon(
  <>
    <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
    <polyline points="2 17 12 22 22 17"></polyline>
    <polyline points="2 12 12 17 22 12"></polyline>
  </>
);

export const Cpu = createIcon(
  <>
    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
    <rect x="9" y="9" width="6" height="6"></rect>
    <line x1="9" y1="2" x2="9" y2="4"></line>
    <line x1="15" y1="2" x2="15" y2="4"></line>
    <line x1="9" y1="20" x2="9" y2="22"></line>
    <line x1="15" y1="20" x2="15" y2="22"></line>
    <line x1="20" y1="9" x2="22" y2="9"></line>
    <line x1="20" y1="14" x2="22" y2="14"></line>
    <line x1="2" y1="9" x2="4" y2="9"></line>
    <line x1="2" y1="14" x2="4" y2="14"></line>
  </>
);

export const BarChart4 = createIcon(
  <>
    <path d="M3 3v18h18"></path>
    <path d="M7 16v-3"></path>
    <path d="M11 16v-8"></path>
    <path d="M15 16v-5"></path>
    <path d="M19 16v-2"></path>
  </>
);

export const GitBranch = createIcon(
  <>
    <line x1="6" y1="3" x2="6" y2="15"></line>
    <circle cx="18" cy="6" r="3"></circle>
    <circle cx="6" cy="18" r="3"></circle>
    <path d="M18 9a9 9 0 0 1-9 9"></path>
  </>
);

export const MessageSquare = createIcon(
  <>
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
  </>
);

export const ChevronRight = createIcon(
  <>
    <polyline points="9 18 15 12 9 6"></polyline>
  </>
);

export const ChevronDown = createIcon(
  <>
    <polyline points="6 9 12 15 18 9"></polyline>
  </>
);

export const Sparkles = createIcon(
  <>
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>
  </>
);