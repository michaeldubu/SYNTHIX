import React, { useEffect, useState } from 'react';
import './banner.css';

const ascii = [
  "   ██████╗   █████╗ ███╗   ███╗",
  "  ██╔════╝  ██╔══██╗████╗ ████║",
  "  ╚█████╗   ███████║██╔████╔██║",
  "   ╚═══██╗  ██╔══██║██║╚██╔╝██║",
  "  ██████╔╝  ██║  ██║██║ ╚═╝ ██║",
  "  ╚═════╝   ╚═╝  ╚═╝╚═╝     ╚═╝"
];

export default function Banner() {
  const [rendered, setRendered] = useState([]);

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setRendered(prev => [...prev, ascii[i]]);
      i++;
      if (i >= ascii.length) clearInterval(interval);
    }, 100); // 100ms per line
    return () => clearInterval(interval);
  }, []);

  return (
    <pre className="ascii-banner">
      {rendered.map((line, index) => (
        <div key={index}>{line}</div>
      ))}
    </pre>
  );
}
